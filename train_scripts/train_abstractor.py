"""
Training script for Abstractor (Module 6)
Trains the model to predict V_Grammar and V_Meta from F_Tensor

Usage:
    python train_scripts/train_abstractor.py --epochs 15 --batch_size 64
    python train_scripts/train_abstractor.py --resume checkpoints/abstractor_epoch_10.pth
    python train_scripts/train_abstractor.py --validate --checkpoint checkpoints/abstractor_best.pth
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.schemas import DEVICE
from models.abstractor import Abstractor, AbstractorLoss, calculate_metrics
from utils.dataset import create_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_metrics = {
        'goal_accuracy': 0,
        'format_accuracy': 0,
        'tone_mae': 0,
        'grammar_mae_total': 0
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        f_tensors = batch['f_tensor'].to(device)

        # Prepare targets
        targets = {
            'v_goal': batch['v_meta'][:, 0].to(device),      # [B]
            'v_tone': batch['v_meta'][:, 1:2].to(device),    # [B, 1]
            'v_format': batch['v_meta'][:, 2].to(device),    # [B]
            'v_grammar': batch['v_grammar'].to(device)       # [B, 4]
        }

        # Forward pass
        predictions = model(f_tensors)

        # Calculate loss
        loss, loss_dict = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)

        # Accumulate
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'goal_acc': f'{metrics["goal_accuracy"]:.3f}',
            'format_acc': f'{metrics["format_accuracy"]:.3f}',
            'tone_mae': f'{metrics["tone_mae"]:.4f}',
            'grammar_mae': f'{metrics["grammar_mae_total"]:.4f}'
        })

    # Average metrics
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0
    total_metrics = {
        'goal_accuracy': 0,
        'format_accuracy': 0,
        'tone_mae': 0,
        'grammar_mae_alignment': 0,
        'grammar_mae_contrast': 0,
        'grammar_mae_whitespace': 0,
        'grammar_mae_hierarchy': 0,
        'grammar_mae_total': 0
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            f_tensors = batch['f_tensor'].to(device)

            # Prepare targets
            targets = {
                'v_goal': batch['v_meta'][:, 0].to(device),
                'v_tone': batch['v_meta'][:, 1:2].to(device),
                'v_format': batch['v_meta'][:, 2].to(device),
                'v_grammar': batch['v_grammar'].to(device)
            }

            # Forward pass
            predictions = model(f_tensors)

            # Calculate loss
            loss, loss_dict = criterion(predictions, targets)

            # Calculate metrics
            metrics = calculate_metrics(predictions, targets)

            # Accumulate
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'goal_acc': f'{metrics["goal_accuracy"]:.3f}',
                'format_acc': f'{metrics["format_accuracy"]:.3f}',
                'grammar_mae': f'{metrics["grammar_mae_total"]:.4f}'
            })

    # Average metrics
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint.get('metrics', {})

    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss:.4f}")
    if metrics:
        print(f"  Goal Acc: {metrics.get('goal_accuracy', 0):.3f}")
        print(f"  Format Acc: {metrics.get('format_accuracy', 0):.3f}")
        print(f"  Grammar MAE: {metrics.get('grammar_mae_total', 0):.4f}")

    return epoch


def main():
    parser = argparse.ArgumentParser(description='Train Abstractor')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_dataset', help='Dataset directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for validation')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained ResNet')

    args = parser.parse_args()

    print("=" * 80)
    print("Abstractor Training (Module 6)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Pretrained ResNet: {args.pretrained}")
    print("=" * 80)

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_ratio=0.9,
        num_workers=0,
        pin_memory=True
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = Abstractor(
        n_goal_classes=4,
        n_format_classes=3,
        pretrained=args.pretrained
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create loss and optimizer
    criterion = AbstractorLoss(
        weight_goal=1.0,
        weight_format=1.0,
        weight_tone=1.0,
        weight_grammar=1.0
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, DEVICE) + 1

    # Validation only mode
    if args.validate:
        if args.checkpoint:
            load_checkpoint(model, None, args.checkpoint, DEVICE)
        print("\n" + "=" * 80)
        print("Running Validation")
        print("=" * 80)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE, 0)
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Goal Accuracy: {val_metrics['goal_accuracy']:.4f}")
        print(f"  Format Accuracy: {val_metrics['format_accuracy']:.4f}")
        print(f"  Tone MAE: {val_metrics['tone_mae']:.4f}")
        print(f"  Grammar MAE (Alignment): {val_metrics['grammar_mae_alignment']:.4f}")
        print(f"  Grammar MAE (Contrast): {val_metrics['grammar_mae_contrast']:.4f}")
        print(f"  Grammar MAE (Whitespace): {val_metrics['grammar_mae_whitespace']:.4f}")
        print(f"  Grammar MAE (Hierarchy): {val_metrics['grammar_mae_hierarchy']:.4f}")
        print(f"  Grammar MAE (Total): {val_metrics['grammar_mae_total']:.4f}")
        print("=" * 80)
        return

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    best_val_loss = float('inf')
    best_epoch = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, DEVICE, epoch
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_metrics'].append(train_metrics)
        training_history['val_metrics'].append(val_metrics)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Goal Acc:   {train_metrics['goal_accuracy']:.4f} | {val_metrics['goal_accuracy']:.4f}")
        print(f"  Format Acc: {train_metrics['format_accuracy']:.4f} | {val_metrics['format_accuracy']:.4f}")
        print(f"  Grammar MAE: {train_metrics['grammar_mae_total']:.4f} | {val_metrics['grammar_mae_total']:.4f}")

        # Save checkpoint
        checkpoint_path = f'checkpoints/abstractor_epoch_{epoch:03d}.pth'
        save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = 'checkpoints/abstractor_best.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, best_path)
            print(f"  *** New best model! Val loss: {val_loss:.4f} ***")

        print("-" * 80)

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"\nFinal metrics (best epoch):")
    best_metrics = training_history['val_metrics'][best_epoch]
    print(f"  Goal Accuracy: {best_metrics['goal_accuracy']:.4f}")
    print(f"  Format Accuracy: {best_metrics['format_accuracy']:.4f}")
    print(f"  Tone MAE: {best_metrics['tone_mae']:.4f}")
    print(f"  Grammar MAE (Total): {best_metrics['grammar_mae_total']:.4f}")
    print(f"  Grammar MAE (Alignment): {best_metrics['grammar_mae_alignment']:.4f}")
    print(f"  Grammar MAE (Contrast): {best_metrics['grammar_mae_contrast']:.4f}")
    print(f"  Grammar MAE (Whitespace): {best_metrics['grammar_mae_whitespace']:.4f}")
    print(f"  Grammar MAE (Hierarchy): {best_metrics['grammar_mae_hierarchy']:.4f}")

    # Check success criteria
    print(f"\nSuccess Criteria Check:")
    goal_pass = best_metrics['goal_accuracy'] > 0.85
    format_pass = best_metrics['format_accuracy'] > 0.85
    grammar_pass = best_metrics['grammar_mae_total'] < 0.10

    print(f"  Goal Accuracy > 0.85: {'✓' if goal_pass else '✗'} ({best_metrics['goal_accuracy']:.4f})")
    print(f"  Format Accuracy > 0.85: {'✓' if format_pass else '✗'} ({best_metrics['format_accuracy']:.4f})")
    print(f"  Grammar MAE < 0.10: {'✓' if grammar_pass else '✗'} ({best_metrics['grammar_mae_total']:.4f})")

    if goal_pass and format_pass and grammar_pass:
        print("\n🎉 All success criteria met!")
    else:
        print("\n⚠️  Some criteria not met. Consider training longer or adjusting hyperparameters.")

    # Save training history
    history_path = 'logs/abstractor_training_log.json'
    with open(history_path, 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'training_history': training_history
        }, f, indent=2)

    print(f"\nTraining log saved: {history_path}")
    print(f"Best checkpoint: checkpoints/abstractor_best.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()
