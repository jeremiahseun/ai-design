import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Tuple

class RealDesignDataset(Dataset):
    """
    Dataset for loading real design data (images + metadata)

    Directory structure:
        data_dir/
            images/         # PNG files
            metadata/       # JSON files with V_Meta
    """

    def __init__(self, data_dir: str, split: str = 'train', train_ratio: float = 0.9):
        self.data_dir = Path(data_dir)
        self.split = split

        # Get all image files
        image_dir = self.data_dir / 'images'
        self.image_files = sorted(list(image_dir.glob('*.png')))

        # Split into train/val
        n_total = len(self.image_files)
        n_train = int(n_total * train_ratio)

        if split == 'train':
            self.files = self.image_files[:n_train]
        elif split == 'val':
            self.files = self.image_files[n_train:]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        print(f"Loaded {len(self.files)} real samples for {split} split")

        self.transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        img_path = self.files[idx]
        filename = img_path.stem

        # Load Image
        img = Image.open(img_path).convert('RGB')
        p_image = self.transform(img) # [3, 256, 256]

        # Load Metadata
        meta_path = self.data_dir / 'metadata' / f'{filename}.json'
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        v_meta = metadata['v_meta']
        v_meta_tensor = torch.tensor([
            v_meta['v_Goal'],
            v_meta['v_Format'], # Note: Order might differ from synthetic, check decoder.py
            v_meta['v_Tone']
        ], dtype=torch.float32)

        # Note: In decoder.py, MetaEmbedding expects:
        # v_goal = v_meta[:, 0]
        # v_format = v_meta[:, 1]
        # v_tone = v_meta[:, 2:3]
        # So order should be [Goal, Format, Tone]

        return {
            'p_image': p_image,
            'v_meta': v_meta_tensor,
            'filename': filename
        }

def create_real_dataloaders(data_dir: str,
                           batch_size: int = 32,
                           train_ratio: float = 0.9,
                           num_workers: int = 0,
                           pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:

    train_dataset = RealDesignDataset(data_dir, split='train', train_ratio=train_ratio)
    val_dataset = RealDesignDataset(data_dir, split='val', train_ratio=train_ratio)

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
