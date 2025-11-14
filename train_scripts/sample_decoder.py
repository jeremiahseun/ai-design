"""
Sampling script for Conditional DDPM Decoder
Generates design images from semantic metadata

Usage:
    # Generate with specific metadata
    python train_scripts/sample_decoder.py --checkpoint checkpoints/decoder_best.pth \
        --goal 0 --format 0 --tone 0.5 --num_samples 4

    # Generate grid with different conditions
    python train_scripts/sample_decoder.py --checkpoint checkpoints/decoder_best.pth \
        --grid --num_samples 16

    # Use DDIM for faster sampling
    python train_scripts/sample_decoder.py --checkpoint checkpoints/decoder_best.pth \
        --ddim --ddim_steps 50 --num_samples 4
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from core.schemas import DEVICE
from models.decoder import ConditionalUNet
from models.diffusion_utils import DiffusionSchedule


def load_model(checkpoint_path, device):
    """
    Load trained decoder model
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint or use defaults
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        n_goal_classes=10,
        n_format_classes=4
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    timesteps = checkpoint.get('timesteps', 1000)

    print(f"  Model loaded (timesteps: {timesteps})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, timesteps


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image

    Args:
        tensor: [C, H, W] in range [-1, 1]

    Returns:
        PIL Image
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Convert to numpy
    np_image = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    np_image = (np_image * 255).astype(np.uint8)

    return Image.fromarray(np_image)


def save_image_grid(images, save_path, nrow=4):
    """
    Save a grid of images

    Args:
        images: List of PIL Images
        save_path: Path to save grid
        nrow: Number of images per row
    """
    n_images = len(images)
    ncol = (n_images + nrow - 1) // nrow

    img_width, img_height = images[0].size
    grid_width = img_width * nrow
    grid_height = img_height * ncol

    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid.paste(img, (col * img_width, row * img_height))

    grid.save(save_path)
    print(f"Saved grid to: {save_path}")


@torch.no_grad()
def generate_samples(model, diffusion, v_meta_list, device, use_ddim=False, ddim_steps=50):
    """
    Generate samples from the model

    Args:
        model: Trained decoder model
        diffusion: Diffusion schedule
        v_meta_list: List of metadata tensors [B, 3]
        device: Device
        use_ddim: Use DDIM sampling (faster)
        ddim_steps: Number of DDIM steps

    Returns:
        Generated images as list of tensors
    """
    batch_size = len(v_meta_list)
    v_meta = torch.stack(v_meta_list).to(device)  # [B, 3]

    shape = (batch_size, 3, 256, 256)

    print(f"\nGenerating {batch_size} samples...")
    print(f"  Method: {'DDIM' if use_ddim else 'DDPM'}")
    if use_ddim:
        print(f"  DDIM steps: {ddim_steps}")

    if use_ddim:
        # DDIM sampling (faster)
        samples = diffusion.ddim_sample(
            model=model,
            shape=shape,
            condition=v_meta,
            ddim_timesteps=ddim_steps,
            eta=0.0,
            clip_denoised=True,
            progress=True
        )
    else:
        # DDPM sampling (slower but potentially higher quality)
        samples = diffusion.p_sample_loop(
            model=model,
            shape=shape,
            condition=v_meta,
            clip_denoised=True,
            progress=True
        )

    return [samples[i] for i in range(batch_size)]


def main():
    parser = argparse.ArgumentParser(description='Sample from Conditional DDPM Decoder')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # Sampling
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to generate')
    parser.add_argument('--ddim', action='store_true',
                       help='Use DDIM sampling (faster)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM steps')

    # Conditioning
    parser.add_argument('--goal', type=int, default=None,
                       help='Goal ID (if None, sample random)')
    parser.add_argument('--format', type=int, default=None,
                       help='Format ID (if None, sample random)')
    parser.add_argument('--tone', type=float, default=None,
                       help='Tone value [0, 1] (if None, sample random)')

    # Grid mode
    parser.add_argument('--grid', action='store_true',
                       help='Generate grid with varied conditions')

    # Output
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save samples')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (default: auto-generated)')

    # Diffusion
    parser.add_argument('--beta_start', type=float, default=0.0001,
                       help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.02,
                       help='Ending beta value')
    parser.add_argument('--schedule_type', type=str, default='linear',
                       choices=['linear', 'cosine'], help='Noise schedule type')

    args = parser.parse_args()

    print("=" * 80)
    print("Conditional DDPM Decoder - Sampling")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Enable MPS fallback
    if DEVICE.type == 'mps':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, timesteps = load_model(args.checkpoint, DEVICE)

    # Create diffusion schedule
    diffusion = DiffusionSchedule(
        timesteps=timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule_type,
        device=DEVICE
    )

    # Create conditioning metadata
    v_meta_list = []

    if args.grid:
        # Generate grid with varied conditions
        print("\nGenerating grid with varied conditions...")
        n_goal = min(args.num_samples // 4, 4)
        n_format = min(args.num_samples // n_goal, 4)

        for goal in range(n_goal):
            for fmt in range(n_format):
                tone = torch.rand(1).item()  # Random tone
                v_meta = torch.tensor([goal, fmt, tone])
                v_meta_list.append(v_meta)

                print(f"  Sample {len(v_meta_list)}: goal={goal}, format={fmt}, tone={tone:.2f}")

    else:
        # Generate with specified or random conditions
        print("\nGenerating samples with conditions:")
        for i in range(args.num_samples):
            if args.goal is not None:
                goal = args.goal
            else:
                goal = torch.randint(0, 10, (1,)).item()

            if args.format is not None:
                fmt = args.format
            else:
                fmt = torch.randint(0, 4, (1,)).item()

            if args.tone is not None:
                tone = args.tone
            else:
                tone = torch.rand(1).item()

            v_meta = torch.tensor([goal, fmt, tone])
            v_meta_list.append(v_meta)

            print(f"  Sample {i+1}: goal={goal}, format={fmt}, tone={tone:.2f}")

    # Generate samples
    sample_tensors = generate_samples(
        model, diffusion, v_meta_list, DEVICE,
        use_ddim=args.ddim, ddim_steps=args.ddim_steps
    )

    # Convert to images
    print("\nConverting to images...")
    images = [tensor_to_image(tensor) for tensor in sample_tensors]

    # Save images
    if args.grid or args.num_samples > 1:
        # Save as grid
        if args.output_name:
            output_path = output_dir / args.output_name
        else:
            output_path = output_dir / f'grid_{args.num_samples}samples.png'

        save_image_grid(images, output_path, nrow=4)

    else:
        # Save individual image
        if args.output_name:
            output_path = output_dir / args.output_name
        else:
            output_path = output_dir / f'sample_goal{args.goal}_format{args.format}.png'

        images[0].save(output_path)
        print(f"Saved sample to: {output_path}")

    print("\n" + "=" * 80)
    print("Sampling Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
