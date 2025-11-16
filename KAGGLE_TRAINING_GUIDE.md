# Kaggle Training Guide - Decoder Module 7

**Why Kaggle?**
- ✅ Free GPU (T4 or P100) - much faster than MPS
- ✅ No memory constraints (16GB GPU memory)
- ✅ No local machine slowdown
- ✅ 30 hours/week free GPU quota
- ⚠️ 12-hour session limit (but we can resume!)

---

## Part 1: Prepare Your Data Locally

### Step 1: Create a Dataset Archive

On your Mac, create a compressed archive of your dataset:

```bash
cd /Users/mac/Documents/GitHub/ai-design

# Create a tar.gz of your dataset (will be ~4-6 GB compressed)
tar -czf synthetic_dataset.tar.gz data/synthetic_dataset/

# Check the size
ls -lh synthetic_dataset.tar.gz
```

**Expected size**: ~4-6 GB (from 12 GB uncompressed)

---

## Part 2: Upload Dataset to Kaggle

### Step 2: Create Kaggle Account

1. Go to https://www.kaggle.com
2. Sign up or log in
3. Verify your phone number (required for GPU access)

### Step 3: Create a Kaggle Dataset

1. Go to https://www.kaggle.com/datasets
2. Click "**New Dataset**" button
3. Upload the `synthetic_dataset.tar.gz` file
   - Title: "AI Design Synthetic Dataset"
   - Slug: `ai-design-synthetic-dataset`
   - Make it **Private** (keep your data private)
4. Click "**Create**"
5. Wait for upload to complete (~10-30 minutes depending on internet speed)

**Your dataset URL will be**: `https://www.kaggle.com/datasets/YOUR_USERNAME/ai-design-synthetic-dataset`

---

## Part 3: Create Training Notebook

### Step 4: Create New Notebook

1. Go to https://www.kaggle.com/code
2. Click "**New Notebook**"
3. Title: "AI Design Decoder Training"
4. Turn on GPU:
   - Click "⚙️ Settings" (right sidebar)
   - Scroll to "Accelerator"
   - Select "**GPU T4 x2**" or "**GPU P100**"
   - Click "Save"

### Step 5: Add Dataset to Notebook

1. In the notebook, click "**+ Add Data**" (right sidebar)
2. Search for "ai-design-synthetic-dataset"
3. Click "Add" on your dataset
4. It will appear under "Input" as `/kaggle/input/ai-design-synthetic-dataset/`

---

## Part 4: Setup Code in Kaggle Notebook

### Step 6: Copy-Paste This Setup Code

**Cell 1: Install Dependencies & Extract Dataset**

```python
# Install required packages
!pip install -q tqdm pillow numpy torch torchvision

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Extract dataset
import os
import tarfile

dataset_path = '/kaggle/input/ai-design-synthetic-dataset/synthetic_dataset.tar.gz'
extract_path = '/kaggle/working/data'

print("Extracting dataset...")
os.makedirs(extract_path, exist_ok=True)

with tarfile.open(dataset_path, 'r:gz') as tar:
    tar.extractall(path='/kaggle/working/')

print(f"Dataset extracted to: {extract_path}")
print(f"Contents: {os.listdir('/kaggle/working/data/synthetic_dataset/')}")
```

**Cell 2: Upload Source Code Files**

You need to upload these files from your local machine:
1. `src/models/decoder.py`
2. `src/models/diffusion_utils.py`
3. `src/core/schemas.py`
4. `src/utils/dataset.py`
5. `train_scripts/train_decoder.py`

**Option A: Upload as Kaggle Dataset (Recommended)**

Create another dataset with your source code:
```bash
# On your Mac
cd /Users/mac/Documents/GitHub/ai-design
tar -czf ai-design-code.tar.gz src/ train_scripts/
```

Then upload `ai-design-code.tar.gz` as a new Kaggle dataset and add it to your notebook.

**Option B: Copy-Paste Code Directly**

In Kaggle notebook, create cells with your code files (see Part 5 below).

---

## Part 5: Source Code Setup (Copy-Paste Method)

If you chose Option B, create these cells:

**Cell 3: Create Directory Structure**

```python
!mkdir -p /kaggle/working/src/models
!mkdir -p /kaggle/working/src/core
!mkdir -p /kaggle/working/src/utils
!mkdir -p /kaggle/working/train_scripts
!mkdir -p /kaggle/working/checkpoints
!mkdir -p /kaggle/working/logs
```

**Cell 4: Create schemas.py**

```python
%%writefile /kaggle/working/src/core/schemas.py
import torch

# Use CUDA for Kaggle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Cell 5: Upload decoder.py**

```python
%%writefile /kaggle/working/src/models/decoder.py
# Copy the entire contents of your src/models/decoder.py here
# (I'll provide this in a separate file - see KAGGLE_CODE_FILES.md)
```

**Cell 6: Upload diffusion_utils.py**

```python
%%writefile /kaggle/working/src/models/diffusion_utils.py
# Copy the entire contents of your src/models/diffusion_utils.py here
```

**Cell 7: Upload dataset.py**

```python
%%writefile /kaggle/working/src/utils/dataset.py
# Copy the entire contents of your src/utils/dataset.py here
```

**Cell 8: Upload train_decoder.py**

```python
%%writefile /kaggle/working/train_scripts/train_decoder.py
# Copy the entire contents of your train_scripts/train_decoder.py here
```

---

## Part 6: Training on Kaggle

### Step 7: Start Training (12-Hour Session)

**Cell 9: Run Training**

```python
# Change to working directory
import os
os.chdir('/kaggle/working')

# Run training for as many epochs as possible in 12 hours
# Estimate: ~30-40 epochs possible in 12 hours on T4 GPU
!python train_scripts/train_decoder.py \
    --epochs 30 \
    --batch_size 32 \
    --data_dir data/synthetic_dataset \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --save_interval 5
```

**Expected Progress**:
- Each epoch: ~10-20 minutes on GPU (vs 1.5-3 hours on MPS!)
- 12 hours = ~30-40 epochs

### Step 8: Monitor Progress

Kaggle will show you the training output in real-time. You'll see:
```
Epoch 0 [Train]: 100%|██████████| 281/281 [02:15<00:00, loss=0.2451, avg_loss=0.2451]
Epoch 0 [Val]:   100%|██████████| 31/31 [00:15<00:00, loss=0.1823, avg_loss=0.1823]

Epoch 0 Summary (2.5 minutes):
  Train Loss: 0.2451 | Val Loss: 0.1823
  💾 Saved best model (val_loss: 0.1823)
```

---

## Part 7: Save and Resume Training

### Step 9: Download Checkpoints Before Session Ends

**Important**: Kaggle sessions are deleted after 12 hours! You must download checkpoints.

**Cell 10: Prepare Checkpoints for Download**

```python
# Create a zip of all checkpoints
import shutil
shutil.make_archive('/kaggle/working/decoder_checkpoints', 'zip', '/kaggle/working/checkpoints')

# Create a zip of logs
shutil.make_archive('/kaggle/working/decoder_logs', 'zip', '/kaggle/working/logs')

print("✅ Checkpoints ready for download:")
print("  - /kaggle/working/decoder_checkpoints.zip")
print("  - /kaggle/working/decoder_logs.zip")
```

**Download Method**:
1. In Kaggle notebook, click "**Output**" tab (bottom)
2. Download `decoder_checkpoints.zip`
3. Download `decoder_logs.zip`

### Step 10: Resume Training in New Session

When the 12-hour session ends or you want to continue:

1. **Start a new Kaggle notebook** (same setup as before)
2. **Upload your checkpoint zip** as a new dataset
3. **Extract checkpoints**:

```python
import zipfile
with zipfile.ZipFile('/kaggle/input/decoder-checkpoints/decoder_checkpoints.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/checkpoints')

# Find the latest checkpoint
import os
checkpoints = sorted([f for f in os.listdir('/kaggle/working/checkpoints') if f.startswith('decoder_epoch')])
latest = checkpoints[-1]
epoch_num = int(latest.split('_')[2].split('.')[0])

print(f"Latest checkpoint: {latest}")
print(f"Resuming from epoch {epoch_num}")
```

4. **Resume training**:

```python
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

## Part 8: Training Schedule

### Recommended Multi-Session Plan

**Session 1**: Epochs 0-30 (12 hours)
```python
--epochs 30
```

**Session 2**: Epochs 30-60 (resume, 12 hours)
```python
--resume checkpoints/decoder_epoch_029.pth --epochs 60
```

**Session 3**: Epochs 60-90 (resume, 12 hours)
```python
--resume checkpoints/decoder_epoch_059.pth --epochs 90
```

**Total**: ~90 epochs in 3 sessions (~36 hours of training time)

---

## Part 9: Final Model Download

### Step 11: Download Trained Model

After final training session:

```python
# Download the best model
import shutil
shutil.copy('/kaggle/working/checkpoints/decoder_best.pth', '/kaggle/working/decoder_best_final.pth')

print("✅ Final model ready for download:")
print("  - /kaggle/working/decoder_best_final.pth")
print("  Size:", os.path.getsize('/kaggle/working/decoder_best_final.pth') / (1024**2), "MB")
```

Download this file from the Output tab.

---

## Part 10: Generate Samples on Kaggle

### Step 12: Test Generation (Optional)

Before downloading, you can test generation on Kaggle:

```python
# Create sampling script
%%writefile /kaggle/working/test_generation.py
import torch
import sys
sys.path.append('/kaggle/working')

from src.models.decoder import ConditionalUNet
from src.models.diffusion_utils import DiffusionSchedule

# Load model
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

checkpoint = torch.load('checkpoints/decoder_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create diffusion
diffusion = DiffusionSchedule(timesteps=1000, device=device)

# Generate sample
v_meta = torch.tensor([[0, 0, 0.5]]).to(device)  # goal=0, format=0, tone=0.5
samples = diffusion.ddim_sample(
    model=model,
    shape=(1, 3, 256, 256),
    condition=v_meta,
    ddim_timesteps=50,
    progress=True
)

# Save sample
from PIL import Image
import numpy as np

img_tensor = (samples[0] + 1.0) / 2.0
img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(img_np).save('/kaggle/working/generated_sample.png')

print("✅ Generated sample saved!")
```

Run it:
```python
!python /kaggle/working/test_generation.py
```

Download the generated image from Output tab.

---

## Part 11: Kaggle Tips & Tricks

### GPU Quota Management

- **30 hours/week** free GPU time
- **12 hours max per session**
- Sessions auto-stop after 12 hours
- Monitor your quota: https://www.kaggle.com/settings

### Best Practices

1. **Save checkpoints frequently** (`--save_interval 5`)
2. **Download checkpoints every 6 hours** (mid-session)
3. **Use DDIM sampling** for faster generation tests (50 steps vs 1000)
4. **Monitor GPU memory**: Add this cell to check usage
   ```python
   import torch
   print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
   print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
   ```

### Troubleshooting

**If session disconnects**:
- Checkpoints are saved every 5 epochs
- Download what you have, start new session, resume

**If GPU quota runs out**:
- Wait for weekly reset (Monday 00:00 UTC)
- Or switch to CPU (much slower but works)

**If upload fails**:
- Split dataset into smaller chunks
- Upload code via GitHub instead (clone in Kaggle)

---

## Part 12: Quick Start Checklist

- [ ] Compress dataset: `tar -czf synthetic_dataset.tar.gz data/synthetic_dataset/`
- [ ] Create Kaggle account + verify phone
- [ ] Upload dataset to Kaggle Datasets
- [ ] Create new notebook with GPU enabled
- [ ] Add dataset to notebook
- [ ] Copy-paste setup code (Cells 1-8)
- [ ] Start training (Cell 9)
- [ ] Monitor progress (check every 2-3 hours)
- [ ] Download checkpoints before 12-hour limit
- [ ] Resume in new session if needed
- [ ] Download final model

---

## Part 13: Expected Results

### Training Speed Comparison

| Device | Time per Epoch | Time for 30 Epochs |
|--------|---------------|-------------------|
| **Mac MPS** (batch=8) | ~3-6 hours | ~90-180 hours (unfeasible) |
| **Kaggle T4** (batch=32) | ~15-20 min | ~7.5-10 hours ✅ |
| **Kaggle P100** (batch=32) | ~10-15 min | ~5-7.5 hours ✅ |

**Bottom line**: Kaggle is **10-20× faster** than your Mac!

### Cost

- **Free**: 30 hours/week GPU
- **Paid**: $0.10-0.30/hour for more GPU time

For this project, **free tier is more than enough**!

---

## Need Help?

If you run into issues, common solutions:

1. **Dataset not found**: Check path is `/kaggle/working/data/synthetic_dataset`
2. **GPU not available**: Restart notebook, enable GPU in settings
3. **Out of memory**: Reduce batch size to 16 or 8
4. **Session timeout**: Download checkpoints and resume

---

Good luck with training! 🚀

**Summary**: Upload dataset → Create notebook → Enable GPU → Run training → Download checkpoints → 10-20× faster than local!
