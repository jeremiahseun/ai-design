"""
Test Script for Encoder (Module 5)
Tests: P_Image [B, 3, 256, 256] → F_Tensor [B, 4, 256, 256]

Validates that the trained encoder can extract structural features from design images.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.schemas import DEVICE
from models.encoder import UNetEncoder
from utils.dataset import SyntheticDesignDataset

def load_encoder(checkpoint_path, device):
    """Load trained encoder model"""
    print(f"Loading encoder from: {checkpoint_path}")

    # Create model (bilinear=False to match training config)
    model = UNetEncoder(n_channels=3, n_color_classes=18, bilinear=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ Encoder loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    val_loss = checkpoint.get('val_loss', checkpoint.get('loss', None))
    if val_loss is not None:
        print(f"   Val Loss: {val_loss:.6f}")
    else:
        print(f"   Val Loss: N/A")

    return model


def visualize_prediction(p_image, f_tensor_pred, f_tensor_gt=None, save_path=None):
    """
    Visualize encoder prediction

    Args:
        p_image: [3, 256, 256] RGB image
        f_tensor_pred: [4, 256, 256] Predicted F_Tensor
        f_tensor_gt: [4, 256, 256] Optional ground truth F_Tensor
        save_path: Path to save visualization
    """
    n_cols = 6 if f_tensor_gt is not None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

    # Original image
    img = p_image.cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(img)
    axes[0].set_title('P_Image (Input)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Predicted F_Tensor channels
    channel_names = ['Text Mask', 'Image Mask', 'Color ID', 'Hierarchy']
    for i, name in enumerate(channel_names):
        axes[i + 1].imshow(f_tensor_pred[i].cpu().numpy(), cmap='viridis')
        axes[i + 1].set_title(f'{name}\n(Predicted)', fontsize=10, fontweight='bold')
        axes[i + 1].axis('off')

    # Ground truth (if provided)
    if f_tensor_gt is not None:
        # Create composite showing GT vs Pred difference
        diff = torch.abs(f_tensor_gt - f_tensor_pred).mean(dim=0)
        axes[5].imshow(diff.cpu().numpy(), cmap='hot')
        axes[5].set_title('Error Map\n(|GT - Pred|)', fontsize=10, fontweight='bold')
        axes[5].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization: {save_path}")

    plt.close()


def test_encoder_on_dataset(model, data_dir, device, n_samples=5):
    """Test encoder on real dataset samples"""
    print("\n" + "="*60)
    print("Testing Encoder on Dataset Samples")
    print("="*60)

    # Create dataset
    dataset = SyntheticDesignDataset(data_dir, split='val')  # Use validation set
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        print("❌ No samples found in dataset")
        return

    # Create output directory
    vis_dir = Path('visualizations/encoder_test')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Test on random samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    total_metrics = {
        'text_iou': [],
        'image_iou': [],
        'color_accuracy': [],
        'hierarchy_mse': []
    }

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            print(f"\n[{idx+1}/{len(indices)}] Testing sample {sample_idx}...")

            # Get sample
            sample = dataset[sample_idx]
            p_image = sample['p_image'].unsqueeze(0).to(device)  # [1, 3, 256, 256]
            f_tensor_gt = sample['f_tensor'].unsqueeze(0).to(device)  # [1, 4, 256, 256]

            # Predict
            f_tensor_pred = model.predict(p_image)  # [1, 4, 256, 256]

            # Calculate metrics
            # Text mask IoU
            pred_text = (f_tensor_pred[:, 0:1] > 0.5).float()
            gt_text = f_tensor_gt[:, 0:1]
            intersection = (pred_text * gt_text).sum()
            union = pred_text.sum() + gt_text.sum() - intersection
            text_iou = (intersection / (union + 1e-6)).item()

            # Image mask IoU
            pred_image = (f_tensor_pred[:, 1:2] > 0.5).float()
            gt_image = f_tensor_gt[:, 1:2]
            intersection = (pred_image * gt_image).sum()
            union = pred_image.sum() + gt_image.sum() - intersection
            image_iou = (intersection / (union + 1e-6)).item()

            # Color accuracy
            pred_color = f_tensor_pred[:, 2]
            gt_color = f_tensor_gt[:, 2]
            color_acc = (pred_color == gt_color).float().mean().item()

            # Hierarchy MSE
            pred_hierarchy = f_tensor_pred[:, 3:4]
            gt_hierarchy = f_tensor_gt[:, 3:4]
            hierarchy_mse = ((pred_hierarchy - gt_hierarchy) ** 2).mean().item()

            print(f"   Text IoU: {text_iou:.4f}")
            print(f"   Image IoU: {image_iou:.4f}")
            print(f"   Color Accuracy: {color_acc:.4f}")
            print(f"   Hierarchy MSE: {hierarchy_mse:.4f}")

            total_metrics['text_iou'].append(text_iou)
            total_metrics['image_iou'].append(image_iou)
            total_metrics['color_accuracy'].append(color_acc)
            total_metrics['hierarchy_mse'].append(hierarchy_mse)

            # Visualize
            save_path = vis_dir / f'encoder_test_{sample_idx}.png'
            visualize_prediction(
                p_image[0],
                f_tensor_pred[0],
                f_tensor_gt[0],
                save_path
            )

    # Print summary
    print("\n" + "="*60)
    print("ENCODER TEST SUMMARY")
    print("="*60)
    print(f"Average Text IoU:       {np.mean(total_metrics['text_iou']):.4f}")
    print(f"Average Image IoU:      {np.mean(total_metrics['image_iou']):.4f}")
    print(f"Average Color Accuracy: {np.mean(total_metrics['color_accuracy']):.4f}")
    print(f"Average Hierarchy MSE:  {np.mean(total_metrics['hierarchy_mse']):.4f}")
    print("="*60)

    # Evaluation
    avg_iou = (np.mean(total_metrics['text_iou']) + np.mean(total_metrics['image_iou'])) / 2

    if avg_iou > 0.8:
        print("✅ PASS: Encoder performance is EXCELLENT (IoU > 0.8)")
    elif avg_iou > 0.6:
        print("⚠️  PASS: Encoder performance is GOOD (IoU > 0.6)")
    elif avg_iou > 0.4:
        print("⚠️  WARNING: Encoder performance is MODERATE (IoU > 0.4)")
    else:
        print("❌ FAIL: Encoder performance is POOR (IoU < 0.4)")

    return total_metrics


def test_encoder_on_random(model, device):
    """Test encoder on random synthetic input"""
    print("\n" + "="*60)
    print("Testing Encoder on Random Input")
    print("="*60)

    # Generate random input
    batch_size = 4
    p_image = torch.rand(batch_size, 3, 256, 256).to(device)

    print(f"Input shape: {p_image.shape}")

    with torch.no_grad():
        # Test forward pass
        outputs = model(p_image)
        print(f"\nOutput shapes:")
        for key, tensor in outputs.items():
            print(f"  {key}: {tensor.shape}")

        # Test predict mode
        f_tensor = model.predict(p_image)
        print(f"\nF_Tensor (predict mode): {f_tensor.shape}")

        # Check output ranges
        print(f"\nOutput value ranges:")
        print(f"  Text Mask: [{outputs['text_mask'].min():.3f}, {outputs['text_mask'].max():.3f}]")
        print(f"  Image Mask: [{outputs['image_mask'].min():.3f}, {outputs['image_mask'].max():.3f}]")
        print(f"  Color IDs: [{f_tensor[:, 2].min():.0f}, {f_tensor[:, 2].max():.0f}]")
        print(f"  Hierarchy: [{outputs['hierarchy'].min():.3f}, {outputs['hierarchy'].max():.3f}]")

    print("\n✅ Random input test passed!")


def main():
    print("="*60)
    print("ENCODER (Module 5) TEST")
    print("="*60)

    # Setup
    device = DEVICE
    print(f"\nDevice: {device}")

    checkpoint_path = 'checkpoints/encoder_best.pth'
    data_dir = 'data/synthetic_dataset'

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure encoder_best.pth is in the checkpoints/ directory")
        return

    # Load model
    model = load_encoder(checkpoint_path, device)

    # Test 1: Random input (sanity check)
    test_encoder_on_random(model, device)

    # Test 2: Real dataset samples (performance evaluation)
    if os.path.exists(data_dir):
        test_encoder_on_dataset(model, data_dir, device, n_samples=5)
    else:
        print(f"\n⚠️  Dataset not found at {data_dir}")
        print("Skipping dataset test.")

    print("\n" + "="*60)
    print("ENCODER TEST COMPLETE")
    print("="*60)
    print("\nCheck visualizations/encoder_test/ for output images")


if __name__ == '__main__':
    main()
