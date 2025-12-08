"""
Test Script for Decoder (Module 7)
Tests: V_Meta [B, 3] → P_Image [B, 3, 256, 256]

Validates that the trained decoder can generate design images from semantic metadata
using the DDPM sampling process.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.schemas import DEVICE
from models.decoder import ConditionalUNet
from models.diffusion_utils import DiffusionSchedule
from utils.real_dataset import RealDesignDataset


def load_decoder(checkpoint_path, device):
    """Load trained decoder model"""
    print(f"Loading decoder from: {checkpoint_path}")

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same config as training
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        n_goal_classes=10,
        n_format_classes=4,
        time_emb_dim=256,
        meta_emb_dim=256,
        attention_levels=(3,),  # Only at 32x32
        dropout=0.1,
        use_gradient_checkpointing=False  # Disable for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create diffusion schedule
    timesteps = checkpoint.get('timesteps', 1000)
    diffusion = DiffusionSchedule(
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type='linear',
        device=device
    )

    print(f"✅ Decoder loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    val_loss = checkpoint.get('val_loss', checkpoint.get('loss', None))
    if val_loss is not None:
        print(f"   Val Loss: {val_loss:.6f}")
    else:
        print(f"   Val Loss: N/A")
    print(f"   Timesteps: {timesteps}")

    return model, diffusion


@torch.no_grad()
def sample_images(model, diffusion, v_meta, device, num_inference_steps=50):
    """
    Sample images from the decoder using DDPM

    Args:
        model: Trained ConditionalUNet
        diffusion: DiffusionSchedule
        v_meta: [B, 3] Metadata (goal, format, tone)
        device: Device to run on
        num_inference_steps: Number of denoising steps (less = faster, more = better quality)

    Returns:
        Generated images [B, 3, 256, 256] in range [0, 1]
    """
    batch_size = v_meta.shape[0]

    # Use the robust DDIM sampling from DiffusionSchedule
    images = diffusion.ddim_sample(
        model=model,
        shape=(batch_size, 3, 256, 256),
        condition=v_meta,
        ddim_timesteps=num_inference_steps,
        eta=0.0, # Deterministic sampling
        clip_denoised=True,
        progress=True
    )

    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    return images


def visualize_generation(v_meta, generated_image, sample_idx, save_path=None):
    """
    Visualize generated image with metadata

    Args:
        v_meta: [3] Metadata (goal, format, tone)
        generated_image: [3, 256, 256] Generated image
        sample_idx: Sample index
        save_path: Path to save visualization
    """
    # Goal and format mappings (update these based on your dataset)
    goal_names = ['Inform', 'Persuade', 'Entertain', 'Inspire', 'Educate',
                  'Promote', 'Announce', 'Invite', 'Celebrate', 'Warn']
    format_names = ['Poster', 'Social', 'Flyer', 'Banner']

    goal_id = int(v_meta[0].item())
    format_id = int(v_meta[1].item())
    tone = v_meta[2].item()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Display generated image
    img = generated_image.cpu().numpy().transpose(1, 2, 0)
    ax.imshow(img)
    ax.axis('off')

    # Add metadata as title
    title = (f"Generated Design #{sample_idx}\n"
             f"Goal: {goal_names[goal_id]} | Format: {format_names[format_id]} | Tone: {tone:.2f}")
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")

    plt.close()


def test_decoder_on_dataset(model, diffusion, data_dir, device, n_samples=5, num_inference_steps=50):
    """Test decoder on real dataset metadata"""
    print("\n" + "="*60)
    print("Testing Decoder on Dataset Metadata")
    print("="*60)
    print(f"Inference steps: {num_inference_steps} (less steps = faster, more steps = better quality)")

    # Create dataset
    dataset = RealDesignDataset(data_dir, split='val')
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        print("❌ No samples found in dataset")
        return

    # Create output directory
    vis_dir = Path('visualizations/decoder_real_test')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Test on random samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    print(f"\nGenerating {len(indices)} designs...")
    print("This may take a few minutes...\n")

    for idx, sample_idx in enumerate(indices):
        print(f"[{idx+1}/{len(indices)}] Generating design from sample {sample_idx}...")

        # Get sample metadata
        sample = dataset[sample_idx]
        v_meta_tensor = sample['v_meta'].unsqueeze(0).to(device) # [1, 3]

        # Extract values for logging
        goal = int(v_meta_tensor[0, 0].item())
        fmt = int(v_meta_tensor[0, 1].item())
        tone = v_meta_tensor[0, 2].item()

        print(f"   Goal: {goal}, Format: {fmt}, Tone: {tone:.2f}")

        # Generate image
        generated_images = sample_images(model, diffusion, v_meta_tensor, device, num_inference_steps)

        # Visualize
        save_path = vis_dir / f'decoder_generated_{sample_idx}.png'
        visualize_generation(v_meta_tensor[0], generated_images[0], sample_idx, save_path)

        # Also save ground truth for comparison
        gt_image = sample['p_image']
        gt_save_path = vis_dir / f'decoder_ground_truth_{sample_idx}.png'
        plt.figure(figsize=(8, 8))
        plt.imshow(gt_image.numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.title(f"Ground Truth #{sample_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(gt_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved GT: {gt_save_path}")

    print("\n" + "="*60)
    print("DECODER TEST SUMMARY")
    print("="*60)
    print(f"✅ Generated {len(indices)} design images successfully")
    print(f"📁 Saved to: {vis_dir}")
    print("\nVisual inspection required:")
    print("  - Do generated images look like designs?")
    print("  - Are elements (text, shapes) visible?")
    print("  - Do colors and layouts make sense?")
    print("="*60)


def test_decoder_on_custom_metadata(model, diffusion, device, num_inference_steps=50):
    """Test decoder on custom metadata (not from dataset)"""
    print("\n" + "="*60)
    print("Testing Decoder on Custom Metadata")
    print("="*60)

    # Create output directory
    vis_dir = Path('visualizations/decoder_real_test')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases
    test_cases = [
        {"goal": 1, "format": 0, "tone": 0.3, "desc": "Persuade_Poster_Calm"},
        {"goal": 1, "format": 2, "tone": 0.8, "desc": "Persuade_Flyer_Energetic"},
        {"goal": 1, "format": 0, "tone": 0.5, "desc": "Persuade_Poster_Neutral"},
        {"goal": 1, "format": 2, "tone": 0.9, "desc": "Persuade_Flyer_Bold"},
    ]

    print(f"Generating {len(test_cases)} custom designs...")
    print("This may take a few minutes...\n")

    for idx, case in enumerate(test_cases):
        print(f"[{idx+1}/{len(test_cases)}] Generating: {case['desc']}...")
        print(f"   Goal: {case['goal']}, Format: {case['format']}, Tone: {case['tone']:.2f}")

        # Create v_meta tensor
        v_meta = torch.tensor([[
            case['goal'],
            case['format'],
            case['tone']
        ]], dtype=torch.float32).to(device)

        # Generate image
        generated_images = sample_images(model, diffusion, v_meta, device, num_inference_steps)

        # Save
        save_path = vis_dir / f'decoder_custom_{case["desc"]}.png'
        visualize_generation(v_meta[0], generated_images[0], case['desc'], save_path)

    print("\n✅ Custom metadata test complete!")


def test_decoder_random_input(model, diffusion, device):
    """Quick sanity check with random input"""
    print("\n" + "="*60)
    print("Testing Decoder with Random Input (Sanity Check)")
    print("="*60)

    # Random metadata
    batch_size = 2
    v_meta = torch.tensor([
        [0, 0, 0.5],  # goal=0, format=0, tone=0.5
        [1, 1, 0.7]   # goal=1, format=1, tone=0.7
    ], dtype=torch.float32).to(device)

    print(f"Input v_meta shape: {v_meta.shape}")
    print(f"Generating {batch_size} images with 10 inference steps (fast)...")

    # Quick generation with few steps
    generated_images = sample_images(model, diffusion, v_meta, device, num_inference_steps=10)

    print(f"\nOutput shape: {generated_images.shape}")
    print(f"Output range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
    print(f"Output mean/std: {generated_images.mean():.3f} / {generated_images.std():.3f}")

    # Check output is in valid range
    assert generated_images.min() >= 0 and generated_images.max() <= 1, \
        "Output should be in [0, 1] range"

    print("\n✅ Random input test passed!")


def main():
    print("="*60)
    print("DECODER (Module 7) TEST")
    print("="*60)

    # Setup
    device = DEVICE
    # device = torch.device('cpu')
    print(f"\nDevice: {device}")

    checkpoint_path = 'checkpoints/decoder_best.pth'
    data_dir = 'data/real_designs'

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure decoder_best.pth is in the checkpoints/ directory")
        return

    # Load model
    model, diffusion = load_decoder(checkpoint_path, device)

    # Test 1: Random input (sanity check)
    test_decoder_random_input(model, diffusion, device)

    # Test 2: Custom metadata
    print("\nℹ️  Running custom metadata test...")
    response = input("Generate 4 custom designs? This takes ~2 minutes. [Y/n]: ")
    if response.lower() != 'n':
        test_decoder_on_custom_metadata(model, diffusion, device, num_inference_steps=100)

    # Test 3: Dataset metadata
    if os.path.exists(data_dir):
        print("\nℹ️  Running dataset metadata test...")
        response = input("Generate 5 designs from dataset? This takes ~3 minutes. [Y/n]: ")
        if response.lower() != 'n':
            test_decoder_on_dataset(model, diffusion, data_dir, device, n_samples=5, num_inference_steps=100)
    else:
        print(f"\n⚠️  Dataset not found at {data_dir}")
        print("Skipping dataset test.")

    print("\n" + "="*60)
    print("DECODER TEST COMPLETE")
    print("="*60)
    print("\nCheck visualizations/decoder_real_test/ for generated images")
    print("\nIMPORTANT: Manually inspect the generated images:")
    print("  ✓ Do they look like designs?")
    print("  ✓ Are colors, shapes, and layouts reasonable?")
    print("  ✓ Compare with ground truth images")


if __name__ == '__main__':
    main()
