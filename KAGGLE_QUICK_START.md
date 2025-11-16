# Kaggle Quick Start - Copy-Paste Ready

This file has all the code ready to copy-paste into Kaggle notebooks.

---

## Pre-Kaggle: Prepare Dataset (On Your Mac)

```bash
cd /Users/mac/Documents/GitHub/ai-design
tar -czf synthetic_dataset.tar.gz data/synthetic_dataset/
ls -lh synthetic_dataset.tar.gz
```

Upload this file to Kaggle Datasets: https://www.kaggle.com/datasets

---

## Kaggle Notebook Cells (Copy-Paste in Order)

### Cell 1: Setup & Extract Dataset

```python
# Check GPU (Kaggle already has torch, tqdm, pillow, numpy pre-installed!)
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Verify packages
import tqdm
import PIL
import numpy
print(f"\n✅ All required packages available")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {numpy.__version__}")

# Setup dataset
import os
import shutil

print("Setting up dataset...")

# Kaggle auto-extracts uploaded archives
# Your data is at: /kaggle/input/ai-design-synthetic-dataset/data/synthetic_dataset/
input_dataset = '/kaggle/input/ai-design-synthetic-dataset/data/synthetic_dataset'

# Check if dataset exists
if os.path.exists(input_dataset):
    print(f"✅ Dataset found at: {input_dataset}")

    # Copy to working directory (optional - can also reference directly)
    # Copying allows training script to use /kaggle/working/data/synthetic_dataset
    working_data = '/kaggle/working/data/synthetic_dataset'
    os.makedirs('/kaggle/working/data', exist_ok=True)

    print(f"Copying dataset to working directory...")
    shutil.copytree(input_dataset, working_data)

    print(f"✅ Dataset ready!")
    print(f"Contents: {os.listdir(working_data)[:5]}")
    print(f"Total images: {len(os.listdir(working_data + '/images/'))}")
else:
    print(f"❌ Dataset not found at {input_dataset}")
    print(f"Available: {os.listdir('/kaggle/input/ai-design-synthetic-dataset/')}")
```

---

### Cell 2: Create Directory Structure

```python
# Create necessary directories
!mkdir -p /kaggle/working/src/models
!mkdir -p /kaggle/working/src/core
!mkdir -p /kaggle/working/src/utils
!mkdir -p /kaggle/working/train_scripts
!mkdir -p /kaggle/working/checkpoints
!mkdir -p /kaggle/working/logs

print("✅ Directory structure created")
```

---

### Cell 3: Create Core Files

```python
# Add working directory to path
import sys
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/working/src')

print("✅ Python path configured")
```

---

### Cell 4: Upload Source Files

**IMPORTANT**: You need to upload these 4 files:

**Option A**: Upload as tar.gz dataset (RECOMMENDED - Fastest)

1. **On your Mac**, create source code archive:
   ```bash
   cd /Users/mac/Documents/GitHub/ai-design
   tar -czf ai-design-code.tar.gz src/ train_scripts/
   ```

2. **Upload to Kaggle Datasets** (like you did with synthetic dataset)
   - Go to https://www.kaggle.com/datasets
   - Create new dataset: "AI Design Code"
   - Upload `ai-design-code.tar.gz`

3. **In your Kaggle notebook**, add the code dataset as input, then run:
   ```python
   import os
   import shutil

   # Kaggle auto-extracts archives
   code_input = '/kaggle/input/ai-design-code'

   # Check if src/ and train_scripts/ exist (auto-extracted)
   if os.path.exists(f"{code_input}/src") and os.path.exists(f"{code_input}/train_scripts"):
       print("✅ Source code found (auto-extracted by Kaggle)")

       # Copy to working directory (dirs_exist_ok allows overwriting)
       shutil.copytree(f"{code_input}/src", "/kaggle/working/src", dirs_exist_ok=True)
       shutil.copytree(f"{code_input}/train_scripts", "/kaggle/working/train_scripts", dirs_exist_ok=True)

       print("✅ Source code copied to working directory!")
   else:
       print(f"❌ Source code not found! Available: {os.listdir(code_input)}")
   ```

**Option B**: Clone from GitHub (if you push code there)

```python
!git clone https://github.com/YOUR_USERNAME/ai-design.git /kaggle/working/repo
!cp -r /kaggle/working/repo/src /kaggle/working/
!cp -r /kaggle/working/repo/train_scripts /kaggle/working/
```

**Option C**: Create via %%writefile (see KAGGLE_TRAINING_GUIDE.md for full code)

---

### Cell 5: Verify Setup (Check Everything is Ready)

**NOTE**: If you uploaded code via Option A (tar.gz), schemas.py is already included and uses CUDA automatically! Skip to verification below.

```python
# Check all files are in place
import os
import sys

required_files = [
    '/kaggle/working/src/models/decoder.py',
    '/kaggle/working/src/models/diffusion_utils.py',
    '/kaggle/working/src/utils/dataset.py',
    '/kaggle/working/train_scripts/train_decoder.py',
    '/kaggle/working/src/core/schemas.py'
]

print("Checking required files...")
for f in required_files:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        print(f"❌ {f} MISSING!")

# Check dataset
dataset_images = len(os.listdir('/kaggle/working/data/synthetic_dataset/images/'))
print(f"\n✅ Dataset: {dataset_images} images found")

# Verify DEVICE is set to CUDA
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/working/src')
from core.schemas import DEVICE
print(f"\n✅ Device configured: {DEVICE}")

if str(DEVICE) != 'cuda':
    print("⚠️  WARNING: Device is not CUDA! Check GPU settings.")
```

---

### Cell 6: Start Training (30 Epochs = ~7-10 hours)

```python
import os
os.chdir('/kaggle/working')

# Start training
!python train_scripts/train_decoder.py \
    --epochs 30 \
    --batch_size 32 \
    --data_dir data/synthetic_dataset \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --save_interval 5 \
    --timesteps 1000 \
    --lr 2e-4
```

---

### Cell 7: Monitor Training (Optional - Run in Parallel)

```python
# Check progress while training
import time
import os

while True:
    if os.path.exists('/kaggle/working/logs/decoder_training_log.json'):
        import json
        with open('/kaggle/working/logs/decoder_training_log.json') as f:
            log = json.load(f)

        train_loss = log['training_history']['train_loss']
        val_loss = log['training_history']['val_loss']

        print(f"Epochs completed: {len(train_loss)}")
        if len(train_loss) > 0:
            print(f"Latest train loss: {train_loss[-1]:.4f}")
            print(f"Latest val loss: {val_loss[-1]:.4f}")
            print(f"Best val loss: {log['best_val_loss']:.4f}")

    time.sleep(300)  # Check every 5 minutes
```

---

### Cell 8: Download Checkpoints (Run Before Session Ends!)

```python
import shutil
import os

# Create zips for download
print("Creating checkpoint archives...")

# Zip checkpoints
shutil.make_archive('/kaggle/working/decoder_checkpoints', 'zip', '/kaggle/working/checkpoints')

# Zip logs
shutil.make_archive('/kaggle/working/decoder_logs', 'zip', '/kaggle/working/logs')

# Check sizes
checkpoint_size = os.path.getsize('/kaggle/working/decoder_checkpoints.zip') / (1024**2)
log_size = os.path.getsize('/kaggle/working/decoder_logs.zip') / (1024**2)

print(f"\n✅ Ready for download:")
print(f"   decoder_checkpoints.zip ({checkpoint_size:.1f} MB)")
print(f"   decoder_logs.zip ({log_size:.1f} MB)")
print(f"\n📥 Download from the 'Output' tab (bottom right)")
```

---

### Cell 9: Generate Test Sample (Optional)

```python
import torch
import sys
sys.path.append('/kaggle/working')

from src.models.decoder import ConditionalUNet
from src.models.diffusion_utils import DiffusionSchedule
from PIL import Image
import numpy as np

print("Loading model...")
device = torch.device('cuda')

model = ConditionalUNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_multipliers=(1, 2, 4, 8),
    n_goal_classes=10,
    n_format_classes=4,
    attention_levels=(3,)
).to(device)

# Load best checkpoint
checkpoint = torch.load('/kaggle/working/checkpoints/decoder_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create diffusion
diffusion = DiffusionSchedule(timesteps=1000, device=device)

print("Generating sample (DDIM 50 steps)...")

# Generate 4 samples with different conditions
v_meta_list = [
    torch.tensor([[0, 0, 0.5]]),  # goal=0, format=0, tone=0.5
    torch.tensor([[1, 1, 0.7]]),  # goal=1, format=1, tone=0.7
    torch.tensor([[2, 2, 0.3]]),  # goal=2, format=2, tone=0.3
    torch.tensor([[3, 0, 0.9]]),  # goal=3, format=0, tone=0.9
]

images = []
for i, v_meta in enumerate(v_meta_list):
    samples = diffusion.ddim_sample(
        model=model,
        shape=(1, 3, 256, 256),
        condition=v_meta.to(device),
        ddim_timesteps=50,
        progress=False
    )

    # Convert to image
    img_tensor = (samples[0] + 1.0) / 2.0
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    images.append(img)

    # Save individual
    img.save(f'/kaggle/working/sample_{i}.png')
    print(f"✅ Sample {i+1} saved")

# Create grid
grid = Image.new('RGB', (512, 512))
for i, img in enumerate(images):
    row = i // 2
    col = i % 2
    grid.paste(img, (col * 256, row * 256))

grid.save('/kaggle/working/samples_grid.png')
print(f"\n✅ Grid saved: samples_grid.png")
print(f"📥 Download from Output tab")
```

---

## To Resume Training in New Session

### Cell 1-5: Same as above (setup)

### Cell 6: Load Previous Checkpoints

```python
import shutil
import os

# Upload your decoder_checkpoints.zip as a dataset first
# Kaggle auto-extracts it to /kaggle/input/decoder-checkpoints/
checkpoint_input = '/kaggle/input/decoder-checkpoints'

# Copy checkpoints to working directory
if os.path.exists(checkpoint_input):
    print("✅ Checkpoint dataset found (auto-extracted by Kaggle)")

    # Copy all checkpoint files
    os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

    for file in os.listdir(checkpoint_input):
        if file.endswith('.pth'):
            src = os.path.join(checkpoint_input, file)
            dst = os.path.join('/kaggle/working/checkpoints', file)
            shutil.copy(src, dst)
            print(f"  Copied: {file}")

    # Find latest checkpoint
    checkpoints = sorted([f for f in os.listdir('/kaggle/working/checkpoints')
                         if f.startswith('decoder_epoch') and f.endswith('.pth')])
    latest = checkpoints[-1]
    epoch_num = int(latest.split('_')[2].split('.')[0])

    print(f"\n✅ Latest checkpoint: {latest}")
    print(f"📍 Resuming from epoch {epoch_num}")
    print(f"🎯 Will train to epoch {epoch_num + 30}")
else:
    print(f"❌ Checkpoints not found! Available: {os.listdir('/kaggle/input/')}")
```

### Cell 7: Resume Training

```python
import os
os.chdir('/kaggle/working')

# Resume training for 30 more epochs
!python train_scripts/train_decoder.py \
    --resume checkpoints/{latest} \
    --epochs {epoch_num + 30} \
    --batch_size 32 \
    --data_dir data/synthetic_dataset \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --save_interval 5
```

---

## Quick Troubleshooting

### GPU Not Available
```python
# Check GPU settings
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# If False: Go to Settings → Accelerator → Select GPU T4 x2
```

### Files Missing
```python
# Re-upload files or use GitHub clone method
!git clone YOUR_REPO_URL /kaggle/working/repo
!cp -r /kaggle/working/repo/src /kaggle/working/
!cp -r /kaggle/working/repo/train_scripts /kaggle/working/
```

### Out of Memory
```python
# Reduce batch size in training command:
# Change --batch_size 32 to --batch_size 16
```

### Check Disk Space
```python
!df -h /kaggle/working
```

---

## Expected Output

Training will show:
```
================================================================================
Conditional DDPM Decoder Training (Module 7)
================================================================================

Device: cuda
CUDA available: True
GPU: Tesla T4

Loading dataset from: data/synthetic_dataset
  Train batches: 281
  Val batches: 31

Initializing model...
  Total parameters: 59,024,195

Starting training for 30 epochs...
================================================================================

Epoch 0 [Train]: 100%|██████████| 281/281 [14:32<00:00, loss=0.2451]
Epoch 0 [Val]:   100%|██████████| 31/31 [00:45<00:00, loss=0.1823]

Epoch 0 Summary (15.3 minutes):
  Train Loss: 0.2451 | Val Loss: 0.1823
  💾 Saved best model (val_loss: 0.1823)
```

**Speed**: ~15-20 minutes per epoch on T4 GPU (vs 1.5-3 hours on Mac MPS!)

---

## That's It!

**TL;DR**:
1. Compress dataset → Upload to Kaggle
2. Create notebook → Enable GPU
3. Copy-paste Cells 1-7 → Start training
4. Download checkpoints before 12 hours
5. Resume in new session if needed

**10-20× faster than local!** 🚀
