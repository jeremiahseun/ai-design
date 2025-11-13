"""
Visualization utilities for DTF training
Includes functions for plotting F_Tensors, training curves, and comparisons
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional
import json


def visualize_f_tensor_prediction(p_image: torch.Tensor,
                                  pred_f: torch.Tensor,
                                  target_f: torch.Tensor,
                                  save_path: Optional[str] = None,
                                  title: str = "F_Tensor Prediction"):
    """
    Visualize P_Image alongside predicted and target F_Tensors

    Args:
        p_image: [3, H, W] tensor, normalized [0, 1]
        pred_f: [4, H, W] predicted F_Tensor
        target_f: [4, H, W] target F_Tensor
        save_path: Optional path to save figure
        title: Figure title
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)

    # Convert tensors to numpy
    p_img_np = p_image.cpu().numpy().transpose(1, 2, 0)
    pred_f_np = pred_f.cpu().numpy()
    target_f_np = target_f.cpu().numpy()

    channel_names = ['Text Mask', 'Image Mask', 'Color ID', 'Hierarchy']

    # Row 1: P_Image
    ax = fig.add_subplot(gs[0, 1:4])
    ax.imshow(p_img_np)
    ax.set_title('Input: P_Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Row 2: Predicted F_Tensor channels
    for i in range(4):
        ax = fig.add_subplot(gs[1, i+0.5 if i >= 2 else i])
        channel = pred_f_np[i]

        if i == 2:  # Color ID - normalize for visualization
            if channel.max() > 0:
                channel = channel / channel.max()

        ax.imshow(channel, cmap='gray' if i != 2 else 'viridis')
        ax.set_title(f'Pred: {channel_names[i]}', fontsize=10)
        ax.axis('off')

    # Row 3: Target F_Tensor channels
    for i in range(4):
        ax = fig.add_subplot(gs[2, i+0.5 if i >= 2 else i])
        channel = target_f_np[i]

        if i == 2:  # Color ID
            if channel.max() > 0:
                channel = channel / channel.max()

        ax.imshow(channel, cmap='gray' if i != 2 else 'viridis')
        ax.set_title(f'Target: {channel_names[i]}', fontsize=10)
        ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_training_curves(log_file: str, save_path: Optional[str] = None):
    """
    Plot training curves from log file

    Args:
        log_file: Path to training log JSON
        save_path: Optional path to save figure
    """
    # Load log
    with open(log_file, 'r') as f:
        log = json.load(f)

    epochs = log['epochs']
    train_loss = log['train_loss']
    val_loss = log['val_loss']
    metrics = log.get('metrics', {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: IoU Metrics
    if 'text_iou' in metrics and 'image_iou' in metrics:
        ax = axes[0, 1]
        ax.plot(epochs, metrics['text_iou'], 'g-', label='Text Mask IoU', linewidth=2)
        ax.plot(epochs, metrics['image_iou'], 'b-', label='Image Mask IoU', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target (0.90)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title('Mask IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # Plot 3: Color Accuracy
    if 'color_accuracy' in metrics:
        ax = axes[1, 0]
        ax.plot(epochs, metrics['color_accuracy'], 'm-', linewidth=2)
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (0.80)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Color ID Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # Plot 4: Hierarchy MSE
    if 'hierarchy_mse' in metrics:
        ax = axes[1, 1]
        ax.plot(epochs, metrics['hierarchy_mse'], 'c-', linewidth=2)
        ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Target (0.05)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Hierarchy MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch_predictions(p_images: torch.Tensor,
                               pred_fs: torch.Tensor,
                               target_fs: torch.Tensor,
                               num_samples: int = 4,
                               save_path: Optional[str] = None):
    """
    Visualize multiple predictions in a grid

    Args:
        p_images: [B, 3, H, W]
        pred_fs: [B, 4, H, W]
        target_fs: [B, 4, H, W]
        num_samples: Number of samples to visualize
        save_path: Optional path to save
    """
    num_samples = min(num_samples, p_images.shape[0])

    fig = plt.figure(figsize=(20, 5 * num_samples))
    gs = GridSpec(num_samples, 9, figure=fig, hspace=0.3, wspace=0.2)

    channel_names = ['Text', 'Image', 'Color', 'Hier.']

    for i in range(num_samples):
        # P_Image
        ax = fig.add_subplot(gs[i, 0])
        img = p_images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        if i == 0:
            ax.set_title('P_Image', fontweight='bold')
        ax.axis('off')

        # Predicted channels
        for j in range(4):
            ax = fig.add_subplot(gs[i, j + 1])
            channel = pred_fs[i, j].cpu().numpy()
            if j == 2 and channel.max() > 0:
                channel = channel / channel.max()
            ax.imshow(channel, cmap='gray' if j != 2 else 'viridis')
            if i == 0:
                ax.set_title(f'Pred\n{channel_names[j]}', fontweight='bold', fontsize=9)
            ax.axis('off')

        # Target channels
        for j in range(4):
            ax = fig.add_subplot(gs[i, j + 5])
            channel = target_fs[i, j].cpu().numpy()
            if j == 2 and channel.max() > 0:
                channel = channel / channel.max()
            ax.imshow(channel, cmap='gray' if j != 2 else 'viridis')
            if i == 0:
                ax.set_title(f'Target\n{channel_names[j]}', fontweight='bold', fontsize=9)
            ax.axis('off')

    fig.suptitle('Batch Predictions', fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(images: List[np.ndarray],
                          titles: List[str],
                          save_path: Optional[str] = None,
                          suptitle: str = "Comparison"):
    """
    Create a grid of images for comparison

    Args:
        images: List of numpy arrays [H, W, C] or [H, W]
        titles: List of titles for each image
        save_path: Optional path to save
        suptitle: Overall title
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class TrainingLogger:
    """
    Logger for training metrics
    """

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f'{experiment_name}_log.json'

        self.log = {
            'experiment': experiment_name,
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'metrics': {},
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: Dict):
        """Log metrics for an epoch"""
        self.log['epochs'].append(epoch)
        self.log['train_loss'].append(train_loss)
        self.log['val_loss'].append(val_loss)

        # Update best
        if val_loss < self.log['best_val_loss']:
            self.log['best_val_loss'] = val_loss
            self.log['best_epoch'] = epoch

        # Log metrics
        for key, value in metrics.items():
            if key not in self.log['metrics']:
                self.log['metrics'][key] = []
            self.log['metrics'][key].append(value)

        # Save log
        self.save()

    def save(self):
        """Save log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)

    def plot_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        visualize_training_curves(str(self.log_file), save_path)


if __name__ == '__main__':
    print("=" * 60)
    print("Visualization Utilities Test")
    print("=" * 60)

    # Create dummy data
    p_image = torch.rand(3, 256, 256)
    pred_f = torch.rand(4, 256, 256)
    target_f = torch.rand(4, 256, 256)

    # Make masks binary
    pred_f[0:2] = (pred_f[0:2] > 0.5).float()
    target_f[0:2] = (target_f[0:2] > 0.5).float()

    print("\nTesting single sample visualization...")
    visualize_f_tensor_prediction(
        p_image, pred_f, target_f,
        save_path='data/test_viz_single.png',
        title='Test Prediction'
    )
    print("  Saved: data/test_viz_single.png")

    # Test batch visualization
    print("\nTesting batch visualization...")
    batch_p = torch.rand(4, 3, 256, 256)
    batch_pred = torch.rand(4, 4, 256, 256)
    batch_target = torch.rand(4, 4, 256, 256)

    visualize_batch_predictions(
        batch_p, batch_pred, batch_target,
        num_samples=2,
        save_path='data/test_viz_batch.png'
    )
    print("  Saved: data/test_viz_batch.png")

    print("\n" + "=" * 60)
    print("Visualization test complete!")
    print("=" * 60)
