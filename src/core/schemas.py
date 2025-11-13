"""
DTF Core Data Contracts
Strict tensor definitions for the Design Tensor Framework.
All code must use Apple Silicon MPS device when available.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np


def get_device() -> torch.device:
    """
    Global device context: prioritize MPS (Apple Silicon), fallback to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Global device
DEVICE = get_device()


@dataclass
class TensorShapes:
    """Strict tensor shape definitions"""
    IMAGE_SIZE: int = 256
    P_IMAGE_SHAPE: Tuple[int, int, int] = (3, 256, 256)  # [C, H, W]
    F_TENSOR_SHAPE: Tuple[int, int, int] = (4, 256, 256)  # [K, H, W]
    V_GRAMMAR_DIM: int = 4  # [Alignment, Contrast, Whitespace, Hierarchy]

    # F_Tensor channel indices
    F_TEXT_MASK: int = 0      # K=0: Text Mask (Binary)
    F_IMAGE_MASK: int = 1     # K=1: Image/Shape Mask (Binary)
    F_COLOR_ID: int = 2       # K=2: Color ID Map (Integer/Long)
    F_HIERARCHY: int = 3      # K=3: Hierarchy Map (Float [0,1])


class P_Image:
    """
    Rendered Design Image
    Shape: [B, 3, 256, 256]
    Type: FloatTensor
    Range: [0, 1] normalized
    """

    @staticmethod
    def validate(tensor: torch.Tensor) -> bool:
        """Validate P_Image tensor"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dtype != torch.float32:
            return False
        if len(tensor.shape) != 4:  # [B, C, H, W]
            return False
        if tensor.shape[1:] != (3, 256, 256):
            return False
        if tensor.min() < 0 or tensor.max() > 1:
            return False
        return True

    @staticmethod
    def create(batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create empty P_Image tensor"""
        if device is None:
            device = DEVICE
        return torch.zeros(batch_size, 3, 256, 256, dtype=torch.float32, device=device)

    @staticmethod
    def from_numpy(img_array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert numpy array to P_Image tensor
        Args:
            img_array: numpy array of shape [H, W, 3] or [B, H, W, 3], range [0, 255]
        """
        if device is None:
            device = DEVICE

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Convert to tensor and transpose to [B, C, H, W]
        if len(img_array.shape) == 3:  # [H, W, C]
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        else:  # [B, H, W, C]
            tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2)

        return tensor.to(device)


class F_Tensor:
    """
    Structural Feature Tensor
    Shape: [B, 4, 256, 256]
    Channels:
        K=0: Text Mask (Binary, float)
        K=1: Image/Shape Mask (Binary, float)
        K=2: Color ID Map (Integer/Long, discrete classes)
        K=3: Hierarchy Map (Float [0, 1])
    """

    @staticmethod
    def validate(tensor: torch.Tensor) -> bool:
        """Validate F_Tensor tensor"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if len(tensor.shape) != 4:  # [B, K, H, W]
            return False
        if tensor.shape[1:] != (4, 256, 256):
            return False
        return True

    @staticmethod
    def create(batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create empty F_Tensor"""
        if device is None:
            device = DEVICE
        return torch.zeros(batch_size, 4, 256, 256, dtype=torch.float32, device=device)

    @staticmethod
    def validate_channels(tensor: torch.Tensor) -> Dict[str, bool]:
        """
        Validate individual channel constraints
        Returns dict of validation results per channel
        """
        results = {}

        # K=0: Text Mask should be binary
        text_mask = tensor[:, TensorShapes.F_TEXT_MASK, :, :]
        results['text_mask_binary'] = torch.all((text_mask == 0) | (text_mask == 1)).item()

        # K=1: Image Mask should be binary
        image_mask = tensor[:, TensorShapes.F_IMAGE_MASK, :, :]
        results['image_mask_binary'] = torch.all((image_mask == 0) | (image_mask == 1)).item()

        # K=2: Color ID should be integers
        color_id = tensor[:, TensorShapes.F_COLOR_ID, :, :]
        results['color_id_discrete'] = torch.all(color_id == color_id.long().float()).item()

        # K=3: Hierarchy should be in [0, 1]
        hierarchy = tensor[:, TensorShapes.F_HIERARCHY, :, :]
        results['hierarchy_range'] = (hierarchy.min() >= 0 and hierarchy.max() <= 1)

        return results


class V_Meta:
    """
    Semantic Metadata Dictionary
    Contains:
        - v_Goal: int/one-hot (design goal/purpose)
        - v_Tone: float (emotional tone)
        - v_Content: embedding/str (content description)
        - v_Format: int (format type: poster, social, etc.)
    """

    @staticmethod
    def create_empty() -> Dict[str, Any]:
        """Create empty V_Meta dictionary"""
        return {
            'v_Goal': 0,
            'v_Tone': 0.5,
            'v_Content': "",
            'v_Format': 0
        }

    @staticmethod
    def validate(meta: Dict[str, Any]) -> bool:
        """Validate V_Meta dictionary structure"""
        required_keys = {'v_Goal', 'v_Tone', 'v_Content', 'v_Format'}
        if not all(k in meta for k in required_keys):
            return False

        # Type checks
        if not isinstance(meta['v_Goal'], (int, np.integer)):
            return False
        if not isinstance(meta['v_Tone'], (float, np.floating)):
            return False
        if not isinstance(meta['v_Content'], str):
            return False
        if not isinstance(meta['v_Format'], (int, np.integer)):
            return False

        return True

    @staticmethod
    def to_tensor(meta: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert V_Meta to flat tensor for model conditioning
        Returns: [v_Goal (1-hot, 10 classes), v_Tone (1), v_Format (1-hot, 5 classes)]
        Total dim: 10 + 1 + 5 = 16
        """
        if device is None:
            device = DEVICE

        # One-hot encode v_Goal (assuming 10 classes)
        v_goal_onehot = torch.zeros(10, device=device)
        v_goal_onehot[meta['v_Goal'] % 10] = 1.0

        # v_Tone as single value
        v_tone = torch.tensor([meta['v_Tone']], device=device)

        # One-hot encode v_Format (assuming 5 classes)
        v_format_onehot = torch.zeros(5, device=device)
        v_format_onehot[meta['v_Format'] % 5] = 1.0

        # Concatenate
        return torch.cat([v_goal_onehot, v_tone, v_format_onehot], dim=0)


class V_Grammar:
    """
    Design Grammar Scores
    Shape: [B, 4]
    Type: FloatTensor [0, 1]
    Order: [Alignment, Contrast, Whitespace, Hierarchy]
    """

    DIMENSION_NAMES = ['Alignment', 'Contrast', 'Whitespace', 'Hierarchy']

    @staticmethod
    def validate(tensor: torch.Tensor) -> bool:
        """Validate V_Grammar tensor"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if len(tensor.shape) != 2:  # [B, 4]
            return False
        if tensor.shape[1] != 4:
            return False
        if tensor.min() < 0 or tensor.max() > 1:
            return False
        return True

    @staticmethod
    def create(batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create empty V_Grammar tensor"""
        if device is None:
            device = DEVICE
        return torch.zeros(batch_size, 4, dtype=torch.float32, device=device)

    @staticmethod
    def from_values(alignment: float, contrast: float,
                   whitespace: float, hierarchy: float,
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """Create V_Grammar tensor from individual values"""
        if device is None:
            device = DEVICE
        return torch.tensor(
            [[alignment, contrast, whitespace, hierarchy]],
            dtype=torch.float32,
            device=device
        )

    @staticmethod
    def to_dict(tensor: torch.Tensor) -> Dict[str, float]:
        """Convert V_Grammar tensor to dictionary (for single sample)"""
        if len(tensor.shape) == 2:
            tensor = tensor[0]  # Take first batch

        return {
            'Alignment': tensor[0].item(),
            'Contrast': tensor[1].item(),
            'Whitespace': tensor[2].item(),
            'Hierarchy': tensor[3].item()
        }


# Export all
__all__ = [
    'get_device',
    'DEVICE',
    'TensorShapes',
    'P_Image',
    'F_Tensor',
    'V_Meta',
    'V_Grammar'
]
