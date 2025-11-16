# Decoder Training Guide - Module 7

**Created**: 2024-11-14
**Strategy**: Incremental training in multiples of 2 epochs
**Internet Required**: ❌ NO - All training is local

---

## Training Decision Summary

You decided to train the decoder incrementally:
- **Start with**: 2 epochs first (test run)
- **Continue in**: Multiples of 2 epochs (2, 4, 6, 8, etc.)
- **Flexibility**: Can run 2+ epochs based on time availability
- **Check progress**: Generate samples after each session to see improvements

**Why this works**:
- No internet needed - all data is local
- Can stop/resume anytime - checkpoints save automatically
- Low commitment per session (~1.5-3 hours for 2 epochs)
- Easy to see progress incrementally

---

## Internet Requirement

### ❌ NO INTERNET NEEDED

**All training is completely local**:
- ✅ Dataset is on your machine (`data/synthetic_dataset/`)
- ✅ Models are defined locally (`src/models/`)
- ✅ PyTorch/libraries already installed
- ✅ Checkpoints save to local disk (`checkpoints/`)

**Only time you need internet**:
- Installing new packages (one-time, if needed)
- Downloading pretrained weights (already done for Encoder/Abstractor)

**For this decoder training**: You can disconnect from internet completely!

---

## Training Commands - Multiples of 2

### Session 1: First 2 Epochs (START HERE)

```bash
python train_scripts/train_decoder.py --epochs 2 --batch_size 16
```

**Expected time**: ~1.5-3 hours
**Output**:
- Checkpoints: `checkpoints/decoder_epoch_000.pth`, `checkpoints/decoder_epoch_001.pth`
- Log: `logs/decoder_training_log.json`

---

### Session 2: Next 2 Epochs (Total: 4)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_001.pth \
    --epochs 4 \
    --batch_size 16
```

**Expected time**: ~1.5-3 hours
**New checkpoints**: `decoder_epoch_002.pth`, `decoder_epoch_003.pth`

---

### Session 3: Next 2 Epochs (Total: 6)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_003.pth \
    --epochs 6 \
    --batch_size 16
```

**Expected time**: ~1.5-3 hours
**New checkpoints**: `decoder_epoch_004.pth`, `decoder_epoch_005.pth`

---

### Session 4: Next 2 Epochs (Total: 8)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_005.pth \
    --epochs 8 \
    --batch_size 16
```

**Expected time**: ~1.5-3 hours
**New checkpoints**: `decoder_epoch_006.pth`, `decoder_epoch_007.pth`

---

### Session 5: Next 2 Epochs (Total: 10)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_007.pth \
    --epochs 10 \
    --batch_size 16
```

**Expected time**: ~1.5-3 hours
**New checkpoints**: `decoder_epoch_008.pth`, `decoder_epoch_009.pth`

---

### Session 6: Next 2 Epochs (Total: 12)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_009.pth \
    --epochs 12 \
    --batch_size 16
```

---

### Session 7: Next 2 Epochs (Total: 14)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_011.pth \
    --epochs 14 \
    --batch_size 16
```

---

### Session 8: Next 2 Epochs (Total: 16)

```bash
python train_scripts/train_decoder.py \
    --resume checkpoints/decoder_epoch_013.pth \
    --epochs 16 \
    --batch_size 16
```

---

### Continue Pattern...

**For epochs 18, 20, 22, 24, 26, 28, 30**:

```bash
# Epoch 18 (resume from epoch 15)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_015.pth --epochs 18 --batch_size 16

# Epoch 20 (resume from epoch 17)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_017.pth --epochs 20 --batch_size 16

# Epoch 22 (resume from epoch 19)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_019.pth --epochs 22 --batch_size 16

# Epoch 24 (resume from epoch 21)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_021.pth --epochs 24 --batch_size 16

# Epoch 26 (resume from epoch 23)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_023.pth --epochs 26 --batch_size 16

# Epoch 28 (resume from epoch 25)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_025.pth --epochs 28 --batch_size 16

# Epoch 30 (resume from epoch 27)
python train_scripts/train_decoder.py --resume checkpoints/decoder_epoch_027.pth --epochs 30 --batch_size 16
```

---

## How to Continue Training - Quick Reference

### Rule: Resume from (target_epoch - 2)

If you want to train to epoch **N**, resume from epoch **(N - 2)**

**Formula**:
```
Target Epoch = Current Epoch + Number of New Epochs
Resume From = Current Epoch - 1 (the last completed checkpoint)
```

### Examples:

| Current Status | Want to Train | Resume From | Command |
|----------------|---------------|-------------|---------|
| 0 epochs done | 2 more epochs | N/A (fresh start) | `--epochs 2` |
| 2 epochs done | 2 more epochs (→4) | epoch 1 | `--resume decoder_epoch_001.pth --epochs 4` |
| 4 epochs done | 2 more epochs (→6) | epoch 3 | `--resume decoder_epoch_003.pth --epochs 6` |
| 6 epochs done | 2 more epochs (→8) | epoch 5 | `--resume decoder_epoch_005.pth --epochs 8` |
| 10 epochs done | 2 more epochs (→12) | epoch 9 | `--resume decoder_epoch_009.pth --epochs 12` |

### What if I want to train MORE than 2 epochs?

**Example: Train 4 epochs instead of 2**

| Current Status | Want to Train | New Target | Resume From | Command |
|----------------|---------------|------------|-------------|---------|
| 0 epochs done | 4 epochs | 4 | N/A | `--epochs 4` |
| 4 epochs done | 4 more (→8) | 8 | epoch 3 | `--resume decoder_epoch_003.pth --epochs 8` |
| 8 epochs done | 4 more (→12) | 12 | epoch 7 | `--resume decoder_epoch_007.pth --epochs 12` |

**Example: Train 5 epochs**

| Current Status | Want to Train | New Target | Resume From | Command |
|----------------|---------------|------------|-------------|---------|
| 0 epochs done | 5 epochs | 5 | N/A | `--epochs 5` |
| 5 epochs done | 5 more (→10) | 10 | epoch 4 | `--resume decoder_epoch_004.pth --epochs 10` |
| 10 epochs done | 5 more (→15) | 15 | epoch 9 | `--resume decoder_epoch_009.pth --epochs 15` |

---

## How to Identify Which Checkpoint to Use

### Method 1: Check Checkpoint Directory

```bash
ls -lh checkpoints/decoder_*.pth | tail -5
```

This shows your most recent checkpoints. The **highest numbered one** is your latest.

Example output:
```
decoder_epoch_000.pth
decoder_epoch_001.pth
decoder_epoch_002.pth
decoder_epoch_003.pth  <-- Latest (use this to resume)
```

### Method 2: Check Training Log

```bash
cat logs/decoder_training_log.json
```

Look at the last entry in `"train_loss"` array - the length tells you how many epochs completed.

### Method 3: Quick Formula

**Last completed epoch** = (highest checkpoint number)

Example:
- You see `decoder_epoch_009.pth` → Last completed = Epoch 9
- To continue 2 more epochs → Resume from `decoder_epoch_009.pth --epochs 11`
  - Wait, that's wrong! It should be `--epochs 11` to train 2 more (9 + 2 = 11)

**CORRECT FORMULA**:
```
To train N more epochs from current position:
--resume decoder_epoch_XXX.pth --epochs (XXX + 1 + N)
```

Example:
- Current: `decoder_epoch_009.pth` (9 epochs done)
- Want: 2 more epochs
- Command: `--resume decoder_epoch_009.pth --epochs 11`
  - Because: 9 (done) + 2 (more) = 11 total

---

## Checking Progress Between Sessions

### After Each Session, Check:

#### 1. **Loss Values** (Should decrease over time)

```bash
cat logs/decoder_training_log.json
```

Look for:
- `train_loss`: Should generally decrease
- `val_loss`: Should generally decrease

**Good signs**:
- Loss dropping from 0.5 → 0.3 → 0.2 → 0.15
- Validation loss following training loss

**Warning signs**:
- Loss increasing or staying flat after 10+ epochs
- Large gap between train and val loss (overfitting)

---

#### 2. **Generate Samples** (Visual quality check)

After each 2-epoch session, generate samples:

```bash
# Generate 4 samples with different conditions
python train_scripts/sample_decoder.py \
    --checkpoint checkpoints/decoder_best.pth \
    --num_samples 4 \
    --ddim --ddim_steps 50 \
    --output_name session_N.png
```

**What to look for**:
- **Epochs 0-4**: Noise/blur → Basic shapes emerging
- **Epochs 4-8**: Recognizable elements (text regions, image areas)
- **Epochs 8-12**: More coherent layouts
- **Epochs 12-20**: Cleaner designs, better colors
- **Epochs 20-30**: Refinement, better details

Save outputs as `session_1.png`, `session_2.png`, etc. to compare!

---

#### 3. **Check Checkpoint Sizes**

```bash
ls -lh checkpoints/decoder_*.pth
```

All checkpoints should be **similar size** (~240-260 MB). If one is much smaller, it may be corrupted.

---

## Expected Timeline

| Session | Epochs | Time per Session | Cumulative Time |
|---------|--------|-----------------|-----------------|
| 1 | 0→2 | ~1.5-3 hours | ~2 hours |
| 2 | 2→4 | ~1.5-3 hours | ~4 hours |
| 3 | 4→6 | ~1.5-3 hours | ~6 hours |
| 4 | 6→8 | ~1.5-3 hours | ~8 hours |
| 5 | 8→10 | ~1.5-3 hours | ~10 hours |
| 6 | 10→12 | ~1.5-3 hours | ~12 hours |
| 7 | 12→14 | ~1.5-3 hours | ~14 hours |
| 8 | 14→16 | ~1.5-3 hours | ~16 hours |
| ... | ... | ... | ... |
| 15 | 28→30 | ~1.5-3 hours | ~30 hours |

**To reach 30 epochs**: ~15 sessions of 2 epochs each

---

## Troubleshooting

### Fixed Issues ✅

**Issue 1: Attention Memory Error**
- Attention at high resolutions causes MPS out-of-memory errors
- **Fix**: Only apply attention at 32×32 (smallest resolution)
- Result: Memory usage reduced to ~256 MB for attention

**Issue 2: Skip Connection Dimension Mismatch**
- U-Net skip connections had dimension mismatches
- **Root cause**: `channels` list (construction time) and `hs` list (runtime) were out of sync due to attention blocks
- **Fix**:
  - Don't append to `channels` list when adding AttentionBlock during construction
  - Don't append to `hs` list for AttentionBlock during forward pass
  - Both lists now track only ResidualBlock and Downsample outputs
- Result: Skip connections now align correctly

**Issue 3: Attention Applied at Wrong Resolution in Decoder**
- Even with `attention_levels=(3,)` set, attention was still being applied at 256×256 during upsampling!
- **Root cause**: Decoder uses `reversed(channel_multipliers)`, so index mapping was wrong
  - Encoder level 3 = 32×32 ✅
  - Decoder i=3 = 256×256 ❌ (because channel_multipliers are reversed)
- **Fix**: Mirror the encoder levels: `encoder_level = len(channel_multipliers) - 1 - i`
  - Now if attention at encoder level 3 → applies at decoder level 0 (which is also 32×32)
- Result: Attention only applied at 32×32 in BOTH encoder and decoder

**Current Settings**:
- `attention_levels=(3,)` - Only applies self-attention at 32×32 resolution
- Memory: ~256 MB for attention (manageable)
- Batch size 16 should work on most MPS devices
- If still OOM, reduce batch size to 8

### If Training STILL Fails with OOM

**Option 1: Reduce Batch Size to 8**

```bash
python train_scripts/train_decoder.py \
    --epochs 2 \
    --batch_size 8  # <-- Changed from 16 to 8
```

This halves memory usage but doubles training time per epoch.

**Option 2: Reduce Batch Size to 4** (if 8 still fails)

```bash
python train_scripts/train_decoder.py \
    --epochs 2 \
    --batch_size 4  # <-- Very safe, but 4x slower
```

**Option 3: Remove Attention Entirely** (if memory is really tight)

Edit `train_scripts/train_decoder.py` line 251:
```python
attention_levels=(),  # <-- Empty tuple = no attention
```

The model will still work fine without self-attention, just slightly less expressive.

### If Training is Too Slow

You can reduce number of timesteps (affects quality slightly):

```bash
python train_scripts/train_decoder.py \
    --epochs 2 \
    --batch_size 16 \
    --timesteps 500  # <-- Faster (default is 1000)
```

### If You Want to Start Over

```bash
# Delete checkpoints
rm checkpoints/decoder_*.pth

# Delete logs
rm logs/decoder_training_log.json

# Start fresh
python train_scripts/train_decoder.py --epochs 2 --batch_size 16
```

---

## Quick Reference Card

### I just finished epoch X, how do I continue?

1. Check your latest checkpoint:
   ```bash
   ls checkpoints/decoder_*.pth | tail -1
   ```

2. Note the number (e.g., `decoder_epoch_009.pth` = epoch 9)

3. To train N more epochs:
   ```bash
   python train_scripts/train_decoder.py \
       --resume checkpoints/decoder_epoch_009.pth \
       --epochs 11 \  # <-- (9 + 2 more epochs)
       --batch_size 16
   ```

### General formula:
```
New --epochs value = Last checkpoint number + 1 + How many more you want

Example:
- Last checkpoint: decoder_epoch_015.pth
- Want to train: 2 more epochs
- Command: --resume decoder_epoch_015.pth --epochs 17
           (because 15 + 1 + 1 = 17... wait that's wrong)

CORRECT:
- Command: --resume decoder_epoch_015.pth --epochs 17
  - This will train epochs 16 and 17 (2 more epochs)
```

Actually, let me clarify:

**Correct formula**:
- Last checkpoint number = Last COMPLETED epoch
- To train to total epoch T: `--epochs T`
- So if you finished epoch 15 and want 2 more:
  - `--resume decoder_epoch_015.pth --epochs 17`
  - This trains epochs 16 and 17

**Simplified**:
```
Last completed + How many more = New --epochs value

Examples:
- Completed 9, want 2 more: --epochs 11 (trains epochs 10-11)
- Completed 15, want 5 more: --epochs 20 (trains epochs 16-20)
```

---

## Final Notes

- ✅ No internet required during training
- ✅ Can pause anytime - checkpoints save automatically
- ✅ Generate samples after each session to see progress
- ✅ Training gets better with more epochs (up to ~30-50)
- ✅ You can mix session lengths (2, 4, 5 epochs - whatever works)

**When to stop?**
- After 30 epochs minimum
- When generated samples look good to you
- When validation loss plateaus

Good luck with training! 🚀
