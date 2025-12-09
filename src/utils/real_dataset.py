import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Tuple, List

class RealDesignDataset(Dataset):
    """
    Dataset for loading real design data from a central metadata file.

    Args:
        metadata_path (str): Path to the final_dataset/metadata.json file.
        data_root (str): The root directory of the project, used to resolve image paths.
        split (str): 'train' or 'val'.
        train_ratio (float): The proportion of the dataset to use for training.
    """

    def __init__(self, metadata_path: str, data_root: str, split: str = 'train', train_ratio: float = 0.9):
        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.split = split

        # Load the central metadata file
        with open(self.metadata_path, 'r') as f:
            all_metadata = json.load(f)

        # Split into train/val
        n_total = len(all_metadata)
        n_train = int(n_total * train_ratio)
        
        # Ensure consistent splits
        # No random shuffling here, to keep val set consistent
        if split == 'train':
            self.metadata_entries: List[Dict] = all_metadata[:n_train]
        elif split == 'val':
            self.metadata_entries: List[Dict] = all_metadata[n_train:]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        print(f"Loaded {len(self.metadata_entries)} real samples for {split} split from {self.metadata_path.name}")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Ensure all images are 256x256
            transforms.ToTensor(), # [0, 1]
        ])

    def __len__(self):
        return len(self.metadata_entries)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        metadata = self.metadata_entries[idx]
        
        # Resolve image path relative to the project root
        img_path = self.data_root / metadata['image_path']
        filename = img_path.stem

        # Load Image
        img = Image.open(img_path).convert('RGB')
        p_image = self.transform(img) # [3, 256, 256]

        # Load Metadata directly from the entry
        v_meta_tensor = torch.tensor([
            metadata['v_Goal'],
            metadata['v_Format'],
            metadata['v_Tone']
        ], dtype=torch.float32)

        return {
            'p_image': p_image,
            'v_meta': v_meta_tensor,
            'filename': filename
        }

def create_real_dataloaders(metadata_path: str,
                           data_root: str,
                           batch_size: int = 32,
                           train_ratio: float = 0.9,
                           num_workers: int = 0,
                           pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:

    train_dataset = RealDesignDataset(metadata_path, data_root, split='train', train_ratio=train_ratio)
    val_dataset = RealDesignDataset(metadata_path, data_root, split='val', train_ratio=train_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader