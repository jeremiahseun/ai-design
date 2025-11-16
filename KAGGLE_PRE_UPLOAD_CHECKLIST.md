# Kaggle Pre-Upload Checklist

Run these commands **before** creating your tar.gz archives to avoid issues.

---

## 1. Clean macOS Hidden Files

macOS creates hidden `._` metadata files that will break the dataset loader.

### Check for hidden files:
```bash
cd /Users/mac/Documents/GitHub/ai-design

# Check dataset
find data/synthetic_dataset -name "._*" | head -10

# Check code
find src train_scripts -name "._*" | head -10
```

### Remove hidden files (if any found):
```bash
# Remove from dataset
find data/synthetic_dataset -name "._*" -delete

# Remove from code
find src train_scripts -name "._*" -delete

# Verify they're gone
find data/synthetic_dataset -name "._*"
find src train_scripts -name "._*"
```

---

## 2. Verify Dataset Integrity

```bash
# Check dataset structure
ls -lh data/synthetic_dataset/

# Should see:
#   images/           (directory with 10,000 .npy files)
#   metadata.json     (file)
#   synthetic_metadata.json (file)

# Check image count
ls data/synthetic_dataset/images/*.npy | wc -l
# Should output: 10000

# Check first few filenames (should be numeric)
ls data/synthetic_dataset/images/ | head -5
# Should see: 000000.npy, 000001.npy, 000002.npy, etc.
```

---

## 3. Verify Code Files Updated

Make sure the dataset.py fix is in place:

```bash
# Check if the fix is present
grep -A 2 "if not f.name.startswith" src/utils/dataset.py

# Should see:
#   if not f.name.startswith('._')  # Skip macOS metadata files
```

If not found, the fix wasn't applied. Re-apply it.

---

## 4. Create Archives CORRECTLY

Use these commands to avoid macOS metadata:

### Dataset Archive:
```bash
cd /Users/mac/Documents/GitHub/ai-design

# Use COPYFILE_DISABLE to prevent ._* files
COPYFILE_DISABLE=1 tar -czf synthetic_dataset.tar.gz data/synthetic_dataset/

# Verify archive contents (should NOT see ._* files)
tar -tzf synthetic_dataset.tar.gz | grep "\._"

# If above shows any ._* files, recreate without them:
find data/synthetic_dataset -name "._*" -delete
COPYFILE_DISABLE=1 tar -czf synthetic_dataset.tar.gz data/synthetic_dataset/
```

### Code Archive:
```bash
cd /Users/mac/Documents/GitHub/ai-design

# Clean hidden files first
find src train_scripts -name "._*" -delete

# Create archive
COPYFILE_DISABLE=1 tar -czf ai-design-code.tar.gz src/ train_scripts/

# Verify (should show no ._* files)
tar -tzf ai-design-code.tar.gz | grep "\._"
```

---

---


## 5. Final Verification

```bash
# Check archive sizes
ls -lh *.tar.gz

# Expected sizes:
#   synthetic_dataset.tar.gz: ~4-6 GB
#   ai-design-code.tar.gz: ~50-100 KB

# List contents of both archives
echo "=== Dataset Archive Contents ==="
tar -tzf synthetic_dataset.tar.gz | head -20

echo "=== Code Archive Contents ==="
tar -tzf ai-design-code.tar.gz
```

Expected code archive structure:
```
src/
src/core/
src/core/__init__.py
src/core/schemas.py
src/models/
src/models/__init__.py
src/models/decoder.py
src/models/diffusion_utils.py
src/utils/
src/utils/__init__.py
src/utils/dataset.py
train_scripts/
train_scripts/train_decoder.py
train_scripts/sample_decoder.py
```

---

## 6. Kaggle Upload Checklist

After creating archives:

- [ ] No `._*` files in dataset archive
- [ ] No `._*` files in code archive
- [ ] Dataset has 10,000 images
- [ ] dataset.py includes the `if not f.name.startswith('._')` fix
- [ ] Archive sizes look correct
- [ ] Ready to upload to Kaggle!

---

## Common Issues and Fixes

### Issue: "ValueError: invalid literal for int()"
**Cause**: macOS hidden files (._*) in dataset
**Fix**: Delete hidden files and recreate archive with `COPYFILE_DISABLE=1`

### Issue: "Dataset not found" on Kaggle
**Cause**: Wrong path or Kaggle didn't auto-extract
**Fix**: Check path is `/kaggle/input/ai-design-synthetic-dataset/data/synthetic_dataset`

### Issue: "Module not found" errors
**Cause**: Code not copied correctly
**Fix**: Verify Cell 4 ran with `dirs_exist_ok=True`

---

## Quick Copy-Paste Commands

Run all these in order:

```bash
cd /Users/mac/Documents/GitHub/ai-design

# 1. Clean hidden files
find data/synthetic_dataset src train_scripts -name "._*" -delete

# 2. Verify dataset
ls data/synthetic_dataset/images/*.npy | wc -l

# 3. Create dataset archive
COPYFILE_DISABLE=1 tar -czf synthetic_dataset.tar.gz data/synthetic_dataset/

# 4. Create code archive
COPYFILE_DISABLE=1 tar -czf ai-design-code.tar.gz src/ train_scripts/

# 5. Verify no hidden files
tar -tzf synthetic_dataset.tar.gz | grep "\._"
tar -tzf ai-design-code.tar.gz | grep "\._"

# 6. Check sizes
ls -lh *.tar.gz

echo "✅ Archives ready for Kaggle upload!"
```

---

## After Upload to Kaggle

In your Kaggle notebook, Cell 1 should show:

```
✅ All required packages available
  PyTorch: 2.x.x
  NumPy: 1.x.x

Setting up dataset...
✅ Dataset found at: /kaggle/input/ai-design-synthetic-dataset/data/synthetic_dataset
Copying dataset to working directory...
✅ Dataset ready!
Contents: ['images', 'metadata.json', 'synthetic_metadata.json']
Total images: 10000
```

If you see fewer than 10000 images or any errors, stop and check the archive!
