"""
Dataset loader for synthetic design data
Loads pre-generated P_Images, F_Tensors, and metadata
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class SyntheticDesignDataset(Dataset):
    """
    Dataset for loading pre-generated synthetic design data

    Directory structure:
        data_dir/
            images/         # P_Image numpy files
            f_tensors/      # F_Tensor numpy files
            metadata/       # JSON files with V_Meta and V_Grammar
    """

    def __init__(self, data_dir: str, split: str = 'train', train_ratio: float = 0.9):
        """
        Args:
            data_dir: Root directory of the dataset
            split: 'train' or 'val'
            train_ratio: Ratio of training samples
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Get all sample indices
        image_dir = self.data_dir / 'images'
        self.all_indices = sorted([
            int(f.stem) for f in image_dir.glob('*.npy')
        ])

        # Split into train/val
        n_total = len(self.all_indices)
        n_train = int(n_total * train_ratio)

        if split == 'train':
            self.indices = self.all_indices[:n_train]
        elif split == 'val':
            self.indices = self.all_indices[n_train:]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        print(f"Loaded {len(self.indices)} samples for {split} split")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing all data for one sample
        """
        sample_idx = self.indices[idx]

        # Load P_Image
        p_image_path = self.data_dir / 'images' / f'{sample_idx:06d}.npy'
        p_image = np.load(p_image_path)  # [H, W, 3], range [0, 255]

        # Normalize to [0, 1] and convert to tensor [3, H, W]
        p_image = torch.from_numpy(p_image).float() / 255.0
        p_image = p_image.permute(2, 0, 1)  # [3, 256, 256]

        # Load F_Tensor
        f_tensor_path = self.data_dir / 'f_tensors' / f'{sample_idx:06d}.npy'
        f_tensor = np.load(f_tensor_path)  # [4, H, W]
        f_tensor = torch.from_numpy(f_tensor).float()

        # Load metadata
        meta_path = self.data_dir / 'metadata' / f'{sample_idx:06d}.json'
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Extract V_Grammar
        v_grammar = torch.tensor(metadata['v_grammar'], dtype=torch.float32)  # [4]

        # Extract V_Meta
        v_meta = metadata['v_meta']
        v_meta_tensor = torch.tensor([
            v_meta['v_Goal'],
            v_meta['v_Tone'],
            v_meta['v_Format']
        ], dtype=torch.float32)

        return {
            'p_image': p_image,           # [3, 256, 256]
            'f_tensor': f_tensor,         # [4, 256, 256]
            'v_grammar': v_grammar,       # [4]
            'v_meta': v_meta_tensor,      # [3]
            'v_meta_dict': v_meta,        # Original dict
            'index': sample_idx
        }


def create_dataloaders(data_dir: str,
                      batch_size: int = 32,
                      train_ratio: float = 0.9,
                      num_workers: int = 0,
                      pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size
        train_ratio: Ratio of training samples
        num_workers: Number of worker processes (0 for main process only)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SyntheticDesignDataset(data_dir, split='train', train_ratio=train_ratio)
    val_dataset = SyntheticDesignDataset(data_dir, split='val', train_ratio=train_ratio)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batch
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


class DatasetStats:
    """
    Utility to compute dataset statistics
    """

    @staticmethod
    def compute_stats(dataset: SyntheticDesignDataset) -> Dict:
        """
        Compute statistics over the dataset
        """
        stats = {
            'n_samples': len(dataset),
            'p_image_mean': [],
            'p_image_std': [],
            'f_tensor_coverage': {'text': [], 'image': []},
            'v_grammar': {'alignment': [], 'contrast': [], 'whitespace': [], 'hierarchy': []}
        }

        print(f"Computing statistics over {len(dataset)} samples...")

        for i in range(len(dataset)):
            sample = dataset[i]

            # P_Image stats
            stats['p_image_mean'].append(sample['p_image'].mean().item())
            stats['p_image_std'].append(sample['p_image'].std().item())

            # F_Tensor coverage
            text_coverage = (sample['f_tensor'][0] > 0).float().mean().item()
            image_coverage = (sample['f_tensor'][1] > 0).float().mean().item()
            stats['f_tensor_coverage']['text'].append(text_coverage)
            stats['f_tensor_coverage']['image'].append(image_coverage)

            # V_Grammar
            grammar = sample['v_grammar']
            stats['v_grammar']['alignment'].append(grammar[0].item())
            stats['v_grammar']['contrast'].append(grammar[1].item())
            stats['v_grammar']['whitespace'].append(grammar[2].item())
            stats['v_grammar']['hierarchy'].append(grammar[3].item())

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples")

        # Compute summary statistics
        summary = {
            'n_samples': stats['n_samples'],
            'p_image': {
                'mean': np.mean(stats['p_image_mean']),
                'std': np.mean(stats['p_image_std'])
            },
            'f_tensor_coverage': {
                'text_mean': np.mean(stats['f_tensor_coverage']['text']),
                'image_mean': np.mean(stats['f_tensor_coverage']['image'])
            },
            'v_grammar': {
                dim: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for dim, values in stats['v_grammar'].items()
            }
        }

        return summary


if __name__ == '__main__':
    # Test the dataset loader
    print("=" * 60)
    print("Dataset Loader Test")
    print("=" * 60)

    data_dir = 'data/synthetic_dataset'

    # Check if dataset exists
    if not Path(data_dir).exists():
        print(f"\nError: Dataset not found at {data_dir}")
        print("Please run: python generate_dataset.py --test")
        exit(1)

    # Create dataset
    dataset = SyntheticDesignDataset(data_dir, split='train', train_ratio=0.9)

    # Test loading a sample
    print(f"\nLoading sample 0...")
    sample = dataset[0]

    print(f"\nSample contents:")
    print(f"  p_image shape: {sample['p_image'].shape}")
    print(f"  p_image range: [{sample['p_image'].min():.3f}, {sample['p_image'].max():.3f}]")
    print(f"  f_tensor shape: {sample['f_tensor'].shape}")
    print(f"  v_grammar shape: {sample['v_grammar'].shape}")
    print(f"  v_grammar values: {sample['v_grammar']}")
    print(f"  v_meta shape: {sample['v_meta'].shape}")
    print(f"  v_meta_dict: {sample['v_meta_dict']}")

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=8,
        train_ratio=0.9
    )

    print(f"\nDataloader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # Test batch loading
    print(f"\nLoading first batch...")
    batch = next(iter(train_loader))

    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Compute statistics (on small subset for testing)
    print(f"\nComputing dataset statistics (first 10 samples)...")
    small_dataset = SyntheticDesignDataset(data_dir, split='train', train_ratio=0.9)
    small_dataset.indices = small_dataset.indices[:10]  # Only test on 10 samples

    stats = DatasetStats.compute_stats(small_dataset)

    print(f"\nDataset Statistics:")
    print(f"  P_Image mean: {stats['p_image']['mean']:.3f}")
    print(f"  P_Image std: {stats['p_image']['std']:.3f}")
    print(f"  Text coverage: {stats['f_tensor_coverage']['text_mean']:.3f}")
    print(f"  Image coverage: {stats['f_tensor_coverage']['image_mean']:.3f}")
    print(f"\n  V_Grammar:")
    for dim, values in stats['v_grammar'].items():
        print(f"    {dim}: μ={values['mean']:.3f}, σ={values['std']:.3f}")

    print("\n" + "=" * 60)
    print("Dataset loader test passed!")
    print("=" * 60)
