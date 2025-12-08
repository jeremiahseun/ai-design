"""
Training script for U-Net Encoder (Module 5)
Trains the model to extract F_Tensor from P_Image

Usage:
    python train_scripts/train_encoder.py --epochs 5 --batch_size 32
    python train_scripts/train_encoder.py --resume checkpoints/encoder_epoch_3.pth
    python train_scripts/train_encoder.py --validate --checkpoint checkpoints/encoder_best.pth
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.schemas import DEVICE
from models.encoder import UNetEncoder, CompositeLoss, calculate_metrics
from utils.dataset import create_dataloaders
from utils.visualization import visualize_batch_predictions, TrainingLogger


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_metrics = {'dice': 0, 'ce_color': 0, 'mse_hierarchy': 0,
                    'text_iou': 0, 'image_iou': 0, 'color_accuracy': 0, 'hierarchy_mse': 0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        p_images = batch['p_image'].to(device)
        f_tensors = batch['f_tensor'].to(device)

        # Forward pass
        predictions = model(p_images)

        # Calculate loss
        loss, loss_dict = criterion(predictions, f_tensors)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Calculate metrics
        metrics = calculate_metrics(predictions, f_tensors)

        # Accumulate
        total_loss += loss.item()
        for key in total_metrics:
            if key in loss_dict:
                total_metrics[key] += loss_dict[key]
            elif key in metrics:
                total_metrics[key] += metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'text_iou': f'{metrics["text_iou"]:.3f}',
            'img_iou': f'{metrics["image_iou"]:.3f}'
        })

    # Average metrics
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, epoch, save_vis=False, vis_dir=None):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0
    total_metrics = {'dice': 0, 'ce_color': 0, 'mse_hierarchy': 0,
                    'text_iou': 0, 'image_iou': 0, 'color_accuracy': 0, 'hierarchy_mse': 0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')

    # Save first batch for visualization
    first_batch_saved = False

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        p_images = batch['p_image'].to(device)
        f_tensors = batch['f_tensor'].to(device)

        # Forward pass
        predictions = model(p_images)

        # Calculate loss
        loss, loss_dict = criterion(predictions, f_tensors)

        # Calculate metrics
        metrics = calculate_metrics(predictions, f_tensors)

        # Accumulate
        total_loss += loss.item()
        for key in total_metrics:
            if key in loss_dict:
                total_metrics[key] += loss_dict[key]
            elif key in metrics:
                total_metrics[key] += metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'text_iou': f'{metrics["text_iou"]:.3f}',
            'img_iou': f'{metrics["image_iou"]:.3f}'
        })

        # Save visualization of first batch
        if save_vis and not first_batch_saved and vis_dir is not None:
            # Get predicted F_Tensor
            pred_f = model.predict(p_images[:4])  # First 4 samples

            vis_path = vis_dir / f'epoch_{epoch:03d}_val.png'
            visualize_batch_predictions(
                p_images[:4],
                pred_f,
                f_tensors[:4],
                num_samples=4,
                save_path=str(vis_path)
            )
            first_batch_saved = True

    # Average metrics
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, checkpoint_dir, is_best=False):
    """
    Save model checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'encoder_epoch_{epoch:03d}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'encoder_best.pth'
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved best model (val_loss: {loss:.4f})")

    return checkpoint_path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"  Resumed from epoch {epoch}")

    return epoch


def train(args):
    """
    Main training function
    """
    print("=" * 80)
    print("DTF U-Net Encoder Training")
    print("=" * 80)

    # Set device
    device = DEVICE
    print(f"\nDevice: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

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
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print(f"\nInitializing model...")
    model = UNetEncoder(
        n_channels=3,
        n_color_classes=args.n_color_classes,
        bilinear=args.bilinear
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CompositeLoss(
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Create logger
    logger = TrainingLogger(args.log_dir, 'encoder_training')

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch = load_checkpoint(model, args.resume, optimizer, scheduler)
        start_epoch += 1  # Start from next epoch

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch,
            save_vis=(epoch % args.vis_interval == 0),
            vis_dir=vis_dir
        )

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

        # Print summary
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Text IoU: {val_metrics['text_iou']:.3f} | Image IoU: {val_metrics['image_iou']:.3f}")
        print(f"  Color Acc: {val_metrics['color_accuracy']:.3f} | Hierarchy MSE: {val_metrics['hierarchy_mse']:.4f}")

        # Log to logger
        logger.log_epoch(epoch, train_loss, val_loss, val_metrics)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_metrics, checkpoint_dir, is_best
            )

        print("-" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print(f"Training log: {logger.log_file}")

    # Plot training curves
    logger.plot_curves(save_path=str(log_dir / 'training_curves.png'))
    print(f"Training curves: {log_dir / 'training_curves.png'}")


def validate_only(args):
    """
    Run validation only
    """
    print("=" * 80)
    print("DTF U-Net Encoder Validation")
    print("=" * 80)

    device = DEVICE
    print(f"\nDevice: {device}")

    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    _, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio
    )

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = UNetEncoder(n_channels=3, n_color_classes=args.n_color_classes).to(device)
    load_checkpoint(model, args.checkpoint)

    # Validate
    criterion = CompositeLoss()
    val_loss, val_metrics = validate_epoch(
        model, val_loader, criterion, device, 0,
        save_vis=True,
        vis_dir=Path(args.vis_dir)
    )

    # Print results
    print("\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Text IoU: {val_metrics['text_iou']:.3f}")
    print(f"  Image IoU: {val_metrics['image_iou']:.3f}")
    print(f"  Color Accuracy: {val_metrics['color_accuracy']:.3f}")
    print(f"  Hierarchy MSE: {val_metrics['hierarchy_mse']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net Encoder')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/synthetic_dataset',
                       help='Path to dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Train/val split ratio')

    # Model
    parser.add_argument('--n_color_classes', type=int, default=18,
                       help='Number of color classes')
    parser.add_argument('--bilinear', action='store_true',
                       help='Use bilinear upsampling instead of transposed conv')

    # Training
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')

    # Loss weights
    parser.add_argument('--dice_weight', type=float, default=1.0,
                       help='Weight for dice loss')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                       help='Weight for cross-entropy loss')
    parser.add_argument('--mse_weight', type=float, default=1.0,
                       help='Weight for MSE loss')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save checkpoint every N epochs')

    # Visualization
    parser.add_argument('--vis_dir', type=str, default='visualizations/encoder',
                       help='Directory to save visualizations')
    parser.add_argument('--vis_interval', type=int, default=1,
                       help='Generate visualizations every N epochs')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')

    # Mode
    parser.add_argument('--validate', action='store_true',
                       help='Run validation only')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to use for validation')

    args = parser.parse_args()

    # Validate mode
    if args.validate:
        if args.checkpoint is None:
            print("Error: --checkpoint required for validation mode")
            return
        validate_only(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
