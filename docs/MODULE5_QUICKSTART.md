# Module 5 Quick Start: U-Net Encoder Training

**Status**: ✅ Ready to Train

All code for Module 5 (U-Net Encoder) is implemented and ready. This guide will help you train your first model.

---

## What Was Implemented

### 1. U-Net Encoder Architecture (`src/models/encoder.py`)
- Standard U-Net with skip connections
- 4 output heads for F_Tensor channels
- CompositeLoss (Dice + CrossEntropy + MSE)
- Metrics calculation (IoU, accuracy, MSE)
- **~31M parameters**

### 2. Dataset Loader (`src/utils/dataset.py`)
- Loads pre-generated synthetic dataset
- Automatic train/val split (90/10 default)
- PyTorch DataLoader integration
- Dataset statistics utilities

### 3. Visualization Tools (`src/utils/visualization.py`)
- F_Tensor prediction comparisons
- Training curve plotting
- Batch prediction grids
- Training logger

### 4. Training Script (`train_scripts/train_encoder.py`)
- Complete training loop with validation
- Checkpoint management
- Automatic visualization
- Resume training support
- Validation-only mode

---

## Training Workflow

### Step 1: Generate Dataset

If you haven't generated the full dataset yet:

```bash
# Generate 10,000 samples (~30-60 minutes)
python generate_dataset.py --num_samples 10000
```

Or for quick testing:

```bash
# Generate 1,000 samples (~3-5 minutes)
python generate_dataset.py --num_samples 1000
```

**Expected output:**
```
data/synthetic_dataset/
├── images/         # 10,000 .npy files (P_Images)
├── f_tensors/      # 10,000 .npy files (F_Tensors)
├── metadata/       # 10,000 .json files
└── visualizations/ # Sample visualizations
```

---

### Step 2: Test Dataset Loading

Verify the dataset loader works:

```bash
python -m src.utils.dataset
```

**Expected output:**
```
Loaded 9000 samples for train split
Loaded 1000 samples for val split
✓ Dataset loader test passed!
```

---

### Step 3: Test Encoder Architecture

Verify the model works:

```bash
python -m src.models.encoder
```

**Expected output:**
```
Model Parameters:
  Total: 31,037,666
  Trainable: 31,037,666

✓ Encoder architecture test passed!
```

---

### Step 4: Train the Encoder

#### Quick Training (5 epochs, ~10-15 minutes on M3 Pro)

```bash
python train_scripts/train_encoder.py \
  --epochs 5 \
  --batch_size 32 \
  --data_dir data/synthetic_dataset
```

#### If Memory Issues (Reduce batch size)

```bash
python train_scripts/train_encoder.py \
  --epochs 5 \
  --batch_size 16 \
  --data_dir data/synthetic_dataset
```

#### Full Training (10 epochs)

```bash
python train_scripts/train_encoder.py \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --data_dir data/synthetic_dataset
```

---

### Step 5: Monitor Training

During training, you'll see:

```
Epoch 0 [Train]: 100%|████████| 281/281 [02:15<00:00]
  loss: 0.3421, text_iou: 0.892, img_iou: 0.856

Epoch 0 [Val]:   100%|████████| 32/32 [00:12<00:00]
  loss: 0.2987, text_iou: 0.915, img_iou: 0.881

Epoch 0 Summary (147.3s):
  Train Loss: 0.3421 | Val Loss: 0.2987
  Text IoU: 0.915 | Image IoU: 0.881
  Color Acc: 0.823 | Hierarchy MSE: 0.0342
  💾 Saved best model (val_loss: 0.2987)
```

**Success Criteria:**
- ✅ Text IoU > 0.90
- ✅ Image IoU > 0.85
- ✅ Color Accuracy > 0.80
- ✅ Hierarchy MSE < 0.05

---

### Step 6: Inspect Results

After training, check these directories:

#### 1. Checkpoints
```
checkpoints/
├── encoder_best.pth       # Best model
├── encoder_epoch_000.pth  # Per-epoch checkpoints
├── encoder_epoch_001.pth
...
```

#### 2. Visualizations
```
visualizations/encoder/
├── epoch_000_val.png      # Validation predictions
├── epoch_001_val.png
...
```

**Open these images** to visually verify the model is learning!

#### 3. Training Logs
```
logs/
├── encoder_training_log.json
└── training_curves.png    # Loss/metrics over time
```

---

## Advanced Usage

### Resume Training

```bash
python train_scripts/train_encoder.py \
  --resume checkpoints/encoder_epoch_003.pth \
  --epochs 10
```

### Validation Only

```bash
python train_scripts/train_encoder.py \
  --validate \
  --checkpoint checkpoints/encoder_best.pth
```

### Custom Hyperparameters

```bash
python train_scripts/train_encoder.py \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-4 \
  --dice_weight 2.0 \
  --ce_weight 0.3 \
  --mse_weight 1.5
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### Problem: "Dataset not found"

**Solution:**
```bash
# Generate dataset first
python generate_dataset.py --num_samples 1000
```

---

### Problem: "RuntimeError: MPS backend out of memory"

**Solution:** Reduce batch size
```bash
python train_scripts/train_encoder.py --batch_size 16
# or even
python train_scripts/train_encoder.py --batch_size 8
```

---

### Problem: Loss not decreasing

**Possible causes:**
1. Learning rate too high → try `--lr 5e-5`
2. Dataset too small → generate more samples
3. Loss weights unbalanced → adjust `--dice_weight`, `--ce_weight`, `--mse_weight`

---

### Problem: Training very slow

**On Apple Silicon:**
- Should process ~2 batches/second on M3 Pro
- 5 epochs on 10k samples ≈ 10-15 minutes

**If slower:**
- Reduce `--num_workers` to 0
- Check Activity Monitor for CPU usage
- Ensure MPS is being used (check training output)

---

## Expected Training Times

**On Apple Silicon M3 Pro:**

| Dataset Size | Batch Size | Epochs | Time     |
|--------------|------------|--------|----------|
| 1,000        | 32         | 5      | ~2 min   |
| 5,000        | 32         | 5      | ~8 min   |
| 10,000       | 32         | 5      | ~15 min  |
| 10,000       | 32         | 10     | ~30 min  |

---

## What to Do After Training

Once your model reaches the success criteria:

### 1. Visual Inspection
Open `visualizations/encoder/epoch_00X_val.png` and check:
- ✅ Text masks align with text in P_Image
- ✅ Image masks align with image placeholders
- ✅ Hierarchy map shows stronger values on important elements

### 2. Check Metrics
Look at `logs/training_curves.png`:
- ✅ Loss is decreasing steadily
- ✅ IoU curves plateau above 0.85
- ✅ No overfitting (train/val gap small)

### 3. Proceed to Module 6
If everything looks good:
```bash
# Next: Train the Abstractor (Module 6)
# This will predict V_Grammar from F_Tensor
```

---

## Quick Command Reference

```bash
# Generate dataset
python generate_dataset.py --num_samples 10000

# Train encoder (default)
python train_scripts/train_encoder.py --epochs 5 --batch_size 32

# Train with low memory
python train_scripts/train_encoder.py --epochs 5 --batch_size 16

# Resume training
python train_scripts/train_encoder.py --resume checkpoints/encoder_epoch_003.pth --epochs 10

# Validate only
python train_scripts/train_encoder.py --validate --checkpoint checkpoints/encoder_best.pth

# Test components
python -m src.models.encoder
python -m src.utils.dataset
python -m src.utils.visualization
```

---

## Files Created for Module 5

```
src/models/encoder.py           # U-Net architecture + loss
src/utils/dataset.py            # Dataset loader
src/utils/visualization.py      # Visualization tools
train_scripts/train_encoder.py  # Training script
```

---

## Next Steps

After successfully training the Encoder:

1. ✅ **Module 5 Complete** - You can extract F_Tensor from P_Image
2. 📋 **Module 6 (Next)** - Train Abstractor (F_Tensor → V_Grammar)
3. 📋 **Module 7** - Train Decoder (V_Meta → P_Image)
4. 📋 **Module 8** - Fine-tuning loop
5. 📋 **Module 9** - Innovation loop

---

**Ready to train?** Start with:

```bash
python train_scripts/train_encoder.py --epochs 5 --batch_size 32
```

Good luck! 🚀
