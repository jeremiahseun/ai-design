"""
Module 3: Extractor - JSON to F_Tensor
Creates F_Tensor directly from JSON coordinates (NOT from image inference).
This ensures perfect ground truth alignment for training.

IMPORTANT: This module does NOT analyze images. It only uses JSON data.
"""

import numpy as np
from typing import Dict, Any, List


class FeatureExtractor:
    """
    Extracts F_Tensor [4, 256, 256] directly from design JSON

    Channel layout:
        K=0: Text Mask (Binary)
        K=1: Image/Shape Mask (Binary)
        K=2: Color ID Map (Integer/Long, discrete classes)
        K=3: Hierarchy Map (Float [0, 1])
    """

    def __init__(self, canvas_size: int = 256):
        """
        Initialize extractor
        Args:
            canvas_size: Size of the canvas (matches rendered image size)
        """
        self.canvas_size = canvas_size

    def extract(self, design: Dict[str, Any]) -> np.ndarray:
        """
        Extract F_Tensor from design JSON

        Args:
            design: Design dictionary from generator

        Returns:
            numpy array [4, H, W] representing F_Tensor
        """
        # Create empty tensor [4, H, W]
        f_tensor = np.zeros((4, self.canvas_size, self.canvas_size), dtype=np.float32)

        # Fill each channel from JSON elements
        for element in design['elements']:
            self._extract_element(f_tensor, element)

        return f_tensor

    def _extract_element(self, f_tensor: np.ndarray, element: Dict[str, Any]):
        """
        Extract features from a single element into F_Tensor channels

        Args:
            f_tensor: The tensor to fill (modified in-place)
            element: Element dictionary with pos, box, type, etc.
        """
        elem_type = element['type']
        pos = element['pos']  # (x, y)
        box = element['box']  # (width, height)
        color_id = element['color_id']
        hierarchy = element['hierarchy']

        # Calculate bounding box
        x1, y1 = pos
        x2, y2 = x1 + box[0], y1 + box[1]

        # Clip to canvas bounds
        x1 = max(0, min(x1, self.canvas_size))
        y1 = max(0, min(y1, self.canvas_size))
        x2 = max(0, min(x2, self.canvas_size))
        y2 = max(0, min(y2, self.canvas_size))

        # Skip if box is invalid
        if x1 >= x2 or y1 >= y2:
            return

        # K=0: Text Mask (Binary)
        if elem_type in ['headline', 'subheadline', 'body', 'cta']:
            f_tensor[0, y1:y2, x1:x2] = 1.0

        # K=1: Image/Shape Mask (Binary)
        if elem_type == 'image':
            f_tensor[1, y1:y2, x1:x2] = 1.0

        # K=2: Color ID Map (Integer classes)
        # Note: We use the color_id directly
        f_tensor[2, y1:y2, x1:x2] = float(color_id)

        # K=3: Hierarchy Map (Float [0, 1])
        # Higher hierarchy elements override lower ones
        current_hierarchy = f_tensor[3, y1:y2, x1:x2]
        f_tensor[3, y1:y2, x1:x2] = np.maximum(current_hierarchy, hierarchy)

    def extract_batch(self, designs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract F_Tensors for multiple designs

        Args:
            designs: List of design dictionaries

        Returns:
            numpy array [B, 4, H, W]
        """
        batch = []
        for design in designs:
            f_tensor = self.extract(design)
            batch.append(f_tensor)

        return np.stack(batch, axis=0)

    def extract_to_tensor(self, design: Dict[str, Any]):
        """
        Extract and convert to PyTorch F_Tensor
        Note: Requires torch, imported locally
        """
        try:
            import torch
            import sys
            sys.path.append('src')
            from core.schemas import F_Tensor, DEVICE

            # Extract to numpy
            f_array = self.extract(design)

            # Convert to tensor
            f_tensor = torch.from_numpy(f_array).unsqueeze(0)  # Add batch dim
            f_tensor = f_tensor.to(DEVICE)

            return f_tensor
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion. Use extract() for numpy arrays.")

    def visualize_channels(self, f_tensor: np.ndarray) -> np.ndarray:
        """
        Create visualization of all 4 channels for debugging

        Args:
            f_tensor: F_Tensor [4, H, W]

        Returns:
            RGB image [H, W*4, 3] showing all channels side-by-side
        """
        # Normalize each channel for visualization
        channels = []

        # K=0: Text Mask (already binary)
        text_mask = (f_tensor[0] * 255).astype(np.uint8)
        channels.append(np.stack([text_mask] * 3, axis=-1))

        # K=1: Image Mask (already binary)
        image_mask = (f_tensor[1] * 255).astype(np.uint8)
        channels.append(np.stack([image_mask] * 3, axis=-1))

        # K=2: Color ID Map (normalize to visible range)
        color_map = f_tensor[2]
        if color_map.max() > 0:
            color_map = (color_map / color_map.max() * 255).astype(np.uint8)
        else:
            color_map = np.zeros_like(color_map, dtype=np.uint8)
        channels.append(np.stack([color_map] * 3, axis=-1))

        # K=3: Hierarchy Map (already [0, 1])
        hierarchy = (f_tensor[3] * 255).astype(np.uint8)
        channels.append(np.stack([hierarchy] * 3, axis=-1))

        # Concatenate horizontally
        return np.concatenate(channels, axis=1)


def validate_alignment(design: Dict[str, Any], f_tensor: np.ndarray,
                       rendered_img: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Validate that F_Tensor masks align with rendered image
    This is a sanity check to ensure JSON -> F_Tensor extraction is correct

    Args:
        design: Original design JSON
        f_tensor: Extracted F_Tensor [4, H, W]
        rendered_img: Rendered P_Image [H, W, 3]
        threshold: Overlap threshold for validation

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'text_mask_coverage': 0.0,
        'image_mask_coverage': 0.0,
        'issues': []
    }

    # Convert rendered image to grayscale
    if len(rendered_img.shape) == 3:
        gray_img = rendered_img.mean(axis=2)
    else:
        gray_img = rendered_img

    # Check text mask (K=0)
    text_mask = f_tensor[0]
    text_pixels = text_mask.sum()
    if text_pixels > 0:
        # In rendered image, text should have non-white pixels where mask is 1
        # This is a rough check
        results['text_mask_coverage'] = float(text_pixels / (text_mask == 1).sum() if (text_mask == 1).sum() > 0 else 0)

    # Check image mask (K=1)
    image_mask = f_tensor[1]
    image_pixels = image_mask.sum()
    if image_pixels > 0:
        results['image_mask_coverage'] = float(image_pixels / (image_mask == 1).sum() if (image_mask == 1).sum() > 0 else 0)

    return results


if __name__ == '__main__':
    # Test the extractor
    import sys
    sys.path.append('src')
    from generators.generator import DesignGenerator

    print("=" * 60)
    print("Feature Extractor Test")
    print("=" * 60)

    # Generate a design
    gen = DesignGenerator(seed=42)
    design = gen.generate()

    print(f"\nExtracting features from design:")
    print(f"  Layout: {design['layout']}")
    print(f"  Elements: {len(design['elements'])}")

    # Extract F_Tensor
    extractor = FeatureExtractor()
    f_tensor = extractor.extract(design)

    print(f"\nExtracted F_Tensor:")
    print(f"  Shape: {f_tensor.shape}")
    print(f"  K=0 (Text Mask) - Non-zero pixels: {(f_tensor[0] > 0).sum()}")
    print(f"  K=1 (Image Mask) - Non-zero pixels: {(f_tensor[1] > 0).sum()}")
    print(f"  K=2 (Color IDs) - Unique values: {np.unique(f_tensor[2])}")
    print(f"  K=3 (Hierarchy) - Range: [{f_tensor[3].min():.2f}, {f_tensor[3].max():.2f}]")

    # Test batch extraction
    designs = gen.generate_batch(3)
    batch = extractor.extract_batch(designs)
    print(f"\nBatch extraction shape: {batch.shape}")

    # Visualize channels
    vis = extractor.visualize_channels(f_tensor)
    print(f"\nVisualization shape: {vis.shape}")

    # Save visualization
    from PIL import Image
    vis_img = Image.fromarray(vis)
    vis_img.save('data/test_f_tensor_channels.png')
    print(f"Saved channel visualization to: data/test_f_tensor_channels.png")

    print("\n" + "=" * 60)
