"""
Training script for Conditional DDPM Decoder (Module 7)
Trains the model to generate design images from semantic metadata.
With resetting optimizer LR to initial value.

Usage:
    python train_scripts/train_decoder.py --epochs 30 --batch_size 16
    python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_15.pth
    python train_scripts/train_decoder.py --validate --checkpoint checkpoints/decoder_best.pth
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add src to path
if os.path.exists('/kaggle/input'):
    # Running on Kaggle, add the specific path to the project code
    sys.path.insert(0, '/kaggle/input/ai-design-code')
else:
    # Running locally, add the parent directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.schemas import DEVICE
from src.models.decoder import ConditionalUNet
from src.models.diffusion_utils import DiffusionSchedule
from src.utils.dataset import create_dataloaders
from src.utils.real_dataset import create_real_dataloaders

def train_epoch(model, dataloader, diffusion, optimizer, device, epoch, gradient_accumulation_steps=1, scaler=None):
    """
    Train for one epoch with gradient accumulation and mixed precision support
    """
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        p_images = batch['p_image'].to(device)  # [B, 3, 256, 256]
        v_meta = batch['v_meta'].to(device)     # [B, 3]

        batch_size = p_images.shape[0]

        # Normalize images to [-1, 1]
        p_images = p_images * 2.0 - 1.0

        # Sample random timesteps for each image in batch
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(p_images)

        # Add noise to images (forward diffusion)
        x_t = diffusion.q_sample(p_images, t, noise)

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                predicted_noise = model(x_t, t, v_meta)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                loss = loss / gradient_accumulation_steps
        else:
            predicted_noise = model(x_t, t, v_meta)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights after accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Accumulate (multiply back by gradient_accumulation_steps for logging)
        total_loss += loss.item() * gradient_accumulation_steps

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    # Average loss
    avg_loss = total_loss / len(dataloader)

    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, diffusion, device, epoch):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        p_images = batch['p_image'].to(device)
        v_meta = batch['v_meta'].to(device)

        batch_size = p_images.shape[0]

        # Normalize images to [-1, 1]
        p_images = p_images * 2.0 - 1.0

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(p_images)

        # Add noise
        x_t = diffusion.q_sample(p_images, t, noise)

        # Predict noise
        predicted_noise = model(x_t, t, v_meta)

        # Calculate loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Accumulate
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    # Average loss
    avg_loss = total_loss / len(dataloader)

    return avg_loss


def save_checkpoint(model, optimizer, scheduler, diffusion, epoch, val_loss, checkpoint_dir, train_loss=None, is_best=False, save_interval=1):
    """
    Save model checkpoint
    Strategy: Keep best model + checkpoints at specified intervals
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'timesteps': diffusion.timesteps
    }

    # Save checkpoint only at specified interval
    if epoch % save_interval == 0:
        # Delete previous epoch checkpoint ONLY if interval is 1 (to save space)
        if save_interval == 1 and epoch > 0:
            prev_checkpoint = checkpoint_dir / f'decoder_epoch_{epoch-1:03d}.pth'
            if prev_checkpoint.exists():
                prev_checkpoint.unlink()
                # print(f"  🗑️  Deleted previous checkpoint: epoch {epoch-1}")

        checkpoint_path = checkpoint_dir / f'decoder_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved epoch checkpoint: epoch {epoch}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'decoder_best.pth'
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint
    Returns: (epoch, val_loss)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', checkpoint.get('loss', float('inf')))  # Backward compatibility

    print(f"  Resumed from epoch {epoch}")
    print(f"  Previous val_loss: {val_loss:.6f}")

    return epoch, val_loss


def train(args):
    """
    Main training function
    """
    print("=" * 80)
    print("Conditional DDPM Decoder Training (Module 7)")
    print("=" * 80)

    # Set device
    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  CUDA available: {torch.cuda.is_available()}")
    elif device.type == 'mps':
        print(f"  MPS available: {torch.backends.mps.is_available()}")

    # Enable MPS fallback if needed
    if device.type == 'mps':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print(f"\nLoading dataset from: {args.data_dir}")

    if args.use_real_data:
        print(f"  Using REAL DATA dataset loader from: {args.metadata_path}")
        # Get absolute path for data_root
        data_root_path = Path(args.data_root).resolve()
        train_loader, val_loader = create_real_dataloaders(
            metadata_path=args.metadata_path,
            data_root=str(data_root_path),
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.num_workers
        )
    else:
        print(f"  Using SYNTHETIC DATA dataset loader from: {args.data_dir}")
        train_loader, val_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.num_workers
        )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create diffusion schedule
    print(f"\nInitializing diffusion schedule...")
    diffusion = DiffusionSchedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule_type,
        device=device
    )
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Beta range: [{args.beta_start}, {args.beta_end}]")
    print(f"  Schedule: {args.schedule_type}")

    # Create model
    print(f"\nInitializing model...")
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        n_goal_classes=args.n_goal_classes,
        n_format_classes=args.n_format_classes,
        time_emb_dim=args.time_emb_dim,
        meta_emb_dim=args.meta_emb_dim,
        attention_levels=(3,),  # Only apply attention at 32x32 (smallest resolution)
        dropout=args.dropout,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    if args.use_gradient_checkpointing:
        print(f"  Gradient checkpointing: Enabled")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7 # End LR slightly above zero
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        print(f"  Mixed precision: Enabled")

    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print("  Resuming training...")
        scheduler_to_load = scheduler if not args.reset_scheduler else None
        optimizer_to_load = optimizer if not args.reset_optimizer else None
        if args.reset_scheduler:
            print("  Scheduler state will be reset.")
        if args.reset_optimizer:
            print("  Optimizer state will be reset.")

        completed_epoch, resumed_val_loss = load_checkpoint(model, args.resume, optimizer_to_load, scheduler_to_load, device=device)
        best_val_loss = resumed_val_loss
        start_epoch = completed_epoch + 1

        # If we reset the scheduler, we must manually sync its internal epoch counter
        # to the epoch we are resuming from, otherwise it will start from epoch 0.
        if args.reset_scheduler:
            print(f"  Manually setting new scheduler's epoch to {completed_epoch}")
            scheduler.last_epoch = completed_epoch

            # Also, forcefully reset the optimizer's learning rate to the initial value,
            # as the value from the checkpoint (which is 0) might be preserved.
            print(f"  Forcefully resetting optimizer LR to initial value: {args.lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    training_history = {
        'train_loss': [],
        'val_loss': []
    }

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, diffusion, optimizer, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scaler=scaler
        )

        # Validate
        val_loss = validate_epoch(
            model, val_loader, diffusion, device, epoch
        )

        # Update scheduler
        scheduler.step()

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint (every epoch to maintain resume capability)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Always save (keeps last epoch + best, auto-deletes old epochs)
        save_checkpoint(
            model, optimizer, scheduler, diffusion, epoch,
            val_loss, checkpoint_dir, train_loss=train_loss, is_best=is_best
        )

        print("-" * 80)

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    # Save training history
    history_path = log_dir / 'decoder_training_log.json'
    with open(history_path, 'w') as f:
        json.dump({
            'best_val_loss': best_val_loss,
            'training_history': training_history
        }, f, indent=2)

    print(f"Training log saved: {history_path}")
    print(f"Best checkpoint: {checkpoint_dir / 'decoder_best.pth'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train Conditional DDPM Decoder')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/synthetic_dataset',
                       help='Path to synthetic dataset directory')
    parser.add_argument('--metadata_path', type=str, default='data/real_designs/final_dataset/metadata.json',
                       help='Path to real data metadata.json file')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory for resolving image paths in real dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Train/val split ratio')

    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels')
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Channel multipliers for each level')
    parser.add_argument('--n_goal_classes', type=int, default=10,
                       help='Number of goal classes')
    parser.add_argument('--n_format_classes', type=int, default=4,
                       help='Number of format classes')
    parser.add_argument('--time_emb_dim', type=int, default=256,
                       help='Time embedding dimension')
    parser.add_argument('--meta_emb_dim', type=int, default=256,
                       help='Meta embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Diffusion
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                       help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.02,
                       help='Ending beta value')
    parser.add_argument('--schedule_type', type=str, default='linear',
                       choices=['linear', 'cosine'], help='Noise schedule type')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (reduced for memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (effective batch size = batch_size * this)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                        help='Device to use for training (cuda, mps, cpu)')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training (AMP)')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True,
                       help='Use gradient checkpointing to save memory')
    parser.add_argument('--use_real_data', action='store_true', help='Use real data loader')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--reset-scheduler', action='store_true',
                        help='Do not load scheduler state when resuming from a checkpoint.')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Do not load optimizer state when resuming from a checkpoint.')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')

    # Visualization
    parser.add_argument('--vis_dir', type=str, default='visualizations/decoder',
                       help='Directory to save visualizations')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
