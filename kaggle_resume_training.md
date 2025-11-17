# Kaggle Resume Training - Decoder from Epoch 16

## Critical Fixes for Disk Space Issues

The previous session failed because:
1. Notebook output grew too large (452k+ lines)
2. Papermill autosave filled the disk
3. Training stopped at epoch 17

**Solutions implemented:**
- Disable verbose logging to notebook
- Clear cell outputs periodically
- Monitor disk space
- Increase checkpoint save frequency

---

## Setup: Fresh Kaggle Notebook

### Cell 1: Install Dependencies & Monitor Disk

```python
# Install required packages
!pip install -q tqdm pillow numpy torch torchvision

# Check GPU
import torch
print(f"✅ GPU Available: {torch.cuda.is_available()}")
print(f"✅ GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Initial disk space check
import subprocess
result = subprocess.run(['df', '-h', '/kaggle/working'], capture_output=True, text=True)
print("\n💾 Initial Disk Space:")
print(result.stdout)
```

---

### Cell 2: Extract Dataset

```python
import os
import tarfile

# Extract dataset
dataset_path = '/kaggle/input/ai-design-synthetic-dataset/synthetic_dataset.tar.gz'
extract_path = '/kaggle/working/data'

print("📦 Extracting dataset...")
os.makedirs(extract_path, exist_ok=True)

with tarfile.open(dataset_path, 'r:gz') as tar:
    tar.extractall(path='/kaggle/working/')

print(f"✅ Dataset extracted to: {extract_path}")

# Verify dataset
import os
dataset_dir = '/kaggle/working/data/synthetic_dataset'
if os.path.exists(dataset_dir):
    file_count = len([f for f in os.listdir(dataset_dir) if f.endswith('.png')])
    print(f"✅ Found {file_count} images")
else:
    print("❌ Dataset directory not found!")
```

---

### Cell 3: Extract Source Code

```python
import tarfile

# Extract source code
code_path = '/kaggle/input/ai-design-code/ai-design-code.tar.gz'

print("📦 Extracting source code...")
with tarfile.open(code_path, 'r:gz') as tar:
    tar.extractall(path='/kaggle/working/')

print("✅ Source code extracted")

# Verify structure
import os
print("\n📁 Directory structure:")
for root, dirs, files in os.walk('/kaggle/working'):
    level = root.replace('/kaggle/working', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:  # Only show 2 levels deep
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show max 5 files per dir
            print(f"{subindent}{file}")
```

---

### Cell 4: Load Checkpoint from Dataset

```python
import shutil
import os

# Copy checkpoint from input to working directory
checkpoint_input = '/kaggle/input/decoder-checkpoint-epoch16/decoder_best.pth'
checkpoint_dir = '/kaggle/working/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, 'decoder_epoch_016.pth')
shutil.copy(checkpoint_input, checkpoint_path)

# Verify checkpoint
checkpoint_size = os.path.getsize(checkpoint_path) / (1024**2)
print(f"✅ Checkpoint loaded: {checkpoint_size:.2f} MB")
print(f"📍 Path: {checkpoint_path}")

# Load checkpoint to verify it's valid
import torch
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"\n📊 Checkpoint info:")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
```

---

### Cell 5: Create Disk Space Monitor

```python
import os
import subprocess

def check_disk_space():
    """Monitor disk space and warn if low"""
    result = subprocess.run(['df', '/kaggle/working'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        parts = lines[1].split()
        if len(parts) >= 5:
            used_percent = int(parts[4].rstrip('%'))
            available = parts[3]
            return used_percent, available
    return None, None

def disk_space_warning(epoch):
    """Check disk space and print warning if needed"""
    used_percent, available = check_disk_space()
    if used_percent and used_percent > 80:
        print(f"⚠️  WARNING: Disk {used_percent}% full (Available: {available})")
        if used_percent > 90:
            print("🚨 CRITICAL: Disk almost full! Consider stopping to save checkpoints.")
    return used_percent

# Test it
used, avail = check_disk_space()
print(f"💾 Current disk usage: {used}% (Available: {avail})")
```

---

### Cell 6: Create Logs Directory

```python
import os

# Create directories
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/logs', exist_ok=True)

print("✅ Directories created:")
print("  - /kaggle/working/checkpoints")
print("  - /kaggle/working/logs")
```

---

### Cell 7: Resume Training with Disk Space Monitoring

```python
import os
import sys

# Change to working directory
os.chdir('/kaggle/working')
sys.path.insert(0, '/kaggle/working')

# Run training with resume
# We're resuming from epoch 16, so we'll train to epoch 50 (34 more epochs)
# With monitoring and frequent saves

print("🚀 Starting training from epoch 16...")
print("Target: 50 epochs total (34 more epochs)")
print("Estimated time: ~10 hours")
print()

!python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_016.pth \
    --epochs 50 \
    --batch_size 32 \
    --data_dir data/synthetic_dataset \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --save_interval 5 \
    --device cuda
```

---

### Cell 8: Post-Training - Check Results

```python
import os

print("📊 Training Complete!")
print()

# List checkpoints
checkpoints_dir = '/kaggle/working/checkpoints'
if os.path.exists(checkpoints_dir):
    checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')])
    print(f"✅ Saved {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints[-5:]:  # Show last 5
        size = os.path.getsize(os.path.join(checkpoints_dir, ckpt)) / (1024**2)
        print(f"  {ckpt}: {size:.2f} MB")

# Check final disk space
import subprocess
result = subprocess.run(['df', '-h', '/kaggle/working'], capture_output=True, text=True)
print("\n💾 Final Disk Space:")
print(result.stdout)
```

---

### Cell 9: Prepare Checkpoints for Download

```python
import tarfile
import os

print("🗜️  Compressing checkpoints for download...")

# Compress only the important checkpoints
checkpoints_to_save = [
    'decoder_best.pth',
    'decoder_epoch_050.pth',  # Final epoch
    'decoder_epoch_045.pth',
    'decoder_epoch_040.pth',
]

# Create a directory with only these checkpoints
import shutil
download_dir = '/kaggle/working/checkpoints_to_download'
os.makedirs(download_dir, exist_ok=True)

for ckpt in checkpoints_to_save:
    src = os.path.join('/kaggle/working/checkpoints', ckpt)
    if os.path.exists(src):
        shutil.copy(src, download_dir)
        print(f"✅ Copied: {ckpt}")

# Compress
with tarfile.open('/kaggle/working/decoder_checkpoints_epoch16-50.tar.gz', 'w:gz') as tar:
    tar.add(download_dir, arcname='checkpoints')

compressed_size = os.path.getsize('/kaggle/working/decoder_checkpoints_epoch16-50.tar.gz') / (1024**2)
print(f"\n✅ Compressed to: {compressed_size:.2f} MB")
print("📥 Ready to download from Output tab: decoder_checkpoints_epoch16-50.tar.gz")
```

---

## Important Notes

### Disk Space Management
- The notebook will now have minimal output
- Disk space is monitored every epoch
- If disk reaches 90%, consider stopping and downloading checkpoints

### Training Progress
- Starting from epoch 16 (val_loss: 0.0009)
- Target: epoch 50 (34 more epochs)
- Estimated time: ~10 hours on T4 GPU
- Checkpoints saved every 5 epochs

### If Session Times Out
- Download the checkpoint archive from Output tab
- Upload as new Kaggle dataset
- Start fresh notebook
- Resume from latest checkpoint

### After Training
- Download `decoder_checkpoints_epoch16-50.tar.gz` from Output tab
- Contains: best model + epochs 40, 45, 50
- Total size: ~2-3 GB compressed

---

## Quick Checklist

Before starting:
- [ ] Upload `decoder_best.pth` as Kaggle dataset
- [ ] Create fresh Kaggle notebook
- [ ] Enable GPU (T4 or P100)
- [ ] Add datasets: synthetic data, code, checkpoint
- [ ] Copy-paste cells 1-9
- [ ] Run training (Cell 7)
- [ ] Monitor progress every 2-3 hours
- [ ] Download checkpoints when done

Good luck! 🚀
