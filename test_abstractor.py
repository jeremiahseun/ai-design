"""
Test Script for Abstractor (Module 6)
Tests: F_Tensor [B, 4, 256, 256] → V_Meta + V_Grammar [B, 4]

Validates that the trained abstractor can predict design quality scores and metadata
from structural features.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from core.schemas import DEVICE
from models.abstractor import Abstractor
from utils.dataset import SyntheticDesignDataset

def load_abstractor(checkpoint_path, device):
    """Load trained abstractor model"""
    print(f"Loading abstractor from: {checkpoint_path}")

    # Create model (match training config)
    model = Abstractor(n_goal_classes=4, n_format_classes=3, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ Abstractor loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    val_loss = checkpoint.get('val_loss', checkpoint.get('loss', None))
    if val_loss is not None:
        print(f"   Val Loss: {val_loss:.6f}")
    else:
        print(f"   Val Loss: N/A")

    return model


def visualize_grammar_prediction(v_grammar_pred, v_grammar_gt, sample_idx, save_path=None):
    """
    Visualize grammar score predictions vs ground truth

    Args:
        v_grammar_pred: [4] Predicted grammar scores
        v_grammar_gt: [4] Ground truth grammar scores
        sample_idx: Sample index for title
        save_path: Path to save visualization
    """
    grammar_names = ['Alignment', 'Contrast', 'Whitespace', 'Hierarchy']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(grammar_names))
    width = 0.35

    # Plot bars
    pred_bars = ax.bar(x - width/2, v_grammar_pred, width, label='Predicted', alpha=0.8, color='steelblue')
    gt_bars = ax.bar(x + width/2, v_grammar_gt, width, label='Ground Truth', alpha=0.8, color='coral')

    # Add value labels on bars
    for bar in pred_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in gt_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Grammar Scores Prediction - Sample {sample_idx}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grammar_names, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization: {save_path}")

    plt.close()


def test_abstractor_on_dataset(model, data_dir, device, n_samples=5):
    """Test abstractor on real dataset samples"""
    print("\n" + "="*60)
    print("Testing Abstractor on Dataset Samples")
    print("="*60)

    # Create dataset
    dataset = SyntheticDesignDataset(data_dir, split='val')  # Use validation set
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        print("❌ No samples found in dataset")
        return

    # Create output directory
    vis_dir = Path('visualizations/abstractor_test')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Test on random samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    total_metrics = {
        'goal_accuracy': [],
        'format_accuracy': [],
        'tone_mae': [],
        'grammar_mae': []
    }

    grammar_names = ['Alignment', 'Contrast', 'Whitespace', 'Hierarchy']

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            print(f"\n[{idx+1}/{len(indices)}] Testing sample {sample_idx}...")

            # Get sample
            sample = dataset[sample_idx]
            f_tensor = sample['f_tensor'].unsqueeze(0).to(device)  # [1, 4, 256, 256]
            v_meta = sample['v_meta_dict']  # Dict

            # Ground truth
            v_goal_gt = torch.tensor([v_meta['v_Goal']]).to(device)
            v_format_gt = torch.tensor([v_meta['v_Format']]).to(device)
            v_tone_gt = torch.tensor([[v_meta['v_Tone']]]).to(device)
            v_grammar_gt = sample['v_grammar'].unsqueeze(0).to(device)  # [1, 4]

            # Predict
            predictions = model(f_tensor)

            # Extract predictions
            v_goal_pred = predictions['v_goal'].argmax(dim=1)  # [1]
            v_format_pred = predictions['v_format'].argmax(dim=1)  # [1]
            v_tone_pred = predictions['v_tone']  # [1, 1]
            v_grammar_pred = predictions['v_grammar']  # [1, 4]

            # Calculate metrics
            goal_correct = (v_goal_pred == v_goal_gt.long()).float().item()
            format_correct = (v_format_pred == v_format_gt.long()).float().item()
            tone_mae = torch.abs(v_tone_pred - v_tone_gt).item()
            grammar_mae = torch.abs(v_grammar_pred - v_grammar_gt).mean().item()

            print(f"   Goal: Pred={v_goal_pred.item()}, GT={v_goal_gt.item()}, Correct={goal_correct}")
            print(f"   Format: Pred={v_format_pred.item()}, GT={v_format_gt.item()}, Correct={format_correct}")
            print(f"   Tone MAE: {tone_mae:.4f}")
            print(f"   Grammar MAE: {grammar_mae:.4f}")

            # Print per-dimension grammar scores
            print(f"   Grammar Scores:")
            for i, name in enumerate(grammar_names):
                pred_val = v_grammar_pred[0, i].item()
                gt_val = v_grammar_gt[0, i].item()
                print(f"      {name:12s}: Pred={pred_val:.3f}, GT={gt_val:.3f}, Error={abs(pred_val-gt_val):.3f}")

            total_metrics['goal_accuracy'].append(goal_correct)
            total_metrics['format_accuracy'].append(format_correct)
            total_metrics['tone_mae'].append(tone_mae)
            total_metrics['grammar_mae'].append(grammar_mae)

            # Visualize grammar scores
            save_path = vis_dir / f'abstractor_test_{sample_idx}.png'
            visualize_grammar_prediction(
                v_grammar_pred[0].cpu().numpy(),
                v_grammar_gt[0].cpu().numpy(),
                sample_idx,
                save_path
            )

    # Print summary
    print("\n" + "="*60)
    print("ABSTRACTOR TEST SUMMARY")
    print("="*60)
    print(f"Average Goal Accuracy:   {np.mean(total_metrics['goal_accuracy']):.4f}")
    print(f"Average Format Accuracy: {np.mean(total_metrics['format_accuracy']):.4f}")
    print(f"Average Tone MAE:        {np.mean(total_metrics['tone_mae']):.4f}")
    print(f"Average Grammar MAE:     {np.mean(total_metrics['grammar_mae']):.4f}")
    print("="*60)

    # Evaluation
    avg_grammar_mae = np.mean(total_metrics['grammar_mae'])

    if avg_grammar_mae < 0.05:
        print("✅ PASS: Abstractor performance is EXCELLENT (Grammar MAE < 0.05)")
    elif avg_grammar_mae < 0.10:
        print("✅ PASS: Abstractor performance is GOOD (Grammar MAE < 0.10)")
    elif avg_grammar_mae < 0.15:
        print("⚠️  WARNING: Abstractor performance is MODERATE (Grammar MAE < 0.15)")
    else:
        print("❌ FAIL: Abstractor performance is POOR (Grammar MAE >= 0.15)")

    return total_metrics


def test_abstractor_on_random(model, device):
    """Test abstractor on random synthetic F_Tensor"""
    print("\n" + "="*60)
    print("Testing Abstractor on Random Input")
    print("="*60)

    # Generate random F_Tensor
    batch_size = 4
    f_tensor = torch.rand(batch_size, 4, 256, 256).to(device)

    # Make masks binary
    f_tensor[:, 0:2] = (f_tensor[:, 0:2] > 0.5).float()
    # Make color IDs discrete
    f_tensor[:, 2] = torch.randint(0, 18, (batch_size, 256, 256), device=device).float()

    print(f"Input shape: {f_tensor.shape}")

    with torch.no_grad():
        # Test forward pass
        outputs = model(f_tensor)

        print(f"\nOutput shapes:")
        for key, tensor in outputs.items():
            print(f"  {key}: {tensor.shape}")

        # Check output ranges
        print(f"\nOutput value ranges:")
        print(f"  v_goal (logits): [{outputs['v_goal'].min():.3f}, {outputs['v_goal'].max():.3f}]")
        print(f"  v_tone: [{outputs['v_tone'].min():.3f}, {outputs['v_tone'].max():.3f}]")
        print(f"  v_format (logits): [{outputs['v_format'].min():.3f}, {outputs['v_format'].max():.3f}]")
        print(f"  v_grammar: [{outputs['v_grammar'].min():.3f}, {outputs['v_grammar'].max():.3f}]")

        # Check that grammar scores are in [0, 1]
        assert outputs['v_grammar'].min() >= 0 and outputs['v_grammar'].max() <= 1, \
            "Grammar scores should be in [0, 1]"

    print("\n✅ Random input test passed!")


def main():
    print("="*60)
    print("ABSTRACTOR (Module 6) TEST")
    print("="*60)

    # Setup
    device = DEVICE
    print(f"\nDevice: {device}")

    checkpoint_path = 'checkpoints/abstractor_best.pth'
    data_dir = 'data/synthetic_dataset'

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure abstractor_best.pth is in the checkpoints/ directory")
        return

    # Load model
    model = load_abstractor(checkpoint_path, device)

    # Test 1: Random input (sanity check)
    test_abstractor_on_random(model, device)

    # Test 2: Real dataset samples (performance evaluation)
    if os.path.exists(data_dir):
        test_abstractor_on_dataset(model, data_dir, device, n_samples=5)
    else:
        print(f"\n⚠️  Dataset not found at {data_dir}")
        print("Skipping dataset test.")

    print("\n" + "="*60)
    print("ABSTRACTOR TEST COMPLETE")
    print("="*60)
    print("\nCheck visualizations/abstractor_test/ for output images")


if __name__ == '__main__':
    main()
