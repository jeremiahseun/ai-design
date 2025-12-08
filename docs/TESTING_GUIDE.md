# Safety Testing Guide - Before Module 9

This guide explains how to run the safety tests that validate your trained models before implementing the Innovation Loop (Module 9).

## Overview

You've completed training three models:
- ✅ **Encoder** (Module 5): Extracts structural features from images
- ✅ **Abstractor** (Module 6): Predicts design quality scores
- ✅ **Decoder** (Module 7): Generates design images from metadata

Before building Module 9, we need to verify each model works correctly.

---

## Test Scripts

### 1. test_encoder.py
**Purpose**: Validates that the encoder correctly extracts structural features

**What it tests**:
- Can it identify text regions?
- Can it identify image/shape regions?
- Can it extract color information?
- Can it understand element hierarchy?

**How to run**:
```bash
python test_encoder.py
```

**Expected output**:
```
ENCODER TEST SUMMARY
=====================
Average Text IoU:       0.8523    ← Should be > 0.6
Average Image IoU:      0.9012    ← Should be > 0.6
Average Color Accuracy: 0.7834
Average Hierarchy MSE:  0.0234

✅ PASS: Encoder performance is EXCELLENT
```

**Time**: ~30 seconds

**Outputs**: `visualizations/encoder_test/` - Side-by-side comparisons

---

### 2. test_abstractor.py
**Purpose**: Validates that the abstractor predicts grammar scores accurately

**What it tests**:
- Can it predict alignment scores?
- Can it predict contrast scores?
- Can it predict whitespace scores?
- Can it predict hierarchy scores?
- Can it classify goal and format?

**How to run**:
```bash
python test_abstractor.py
```

**Expected output**:
```
ABSTRACTOR TEST SUMMARY
=======================
Average Goal Accuracy:   0.8000
Average Format Accuracy: 0.9000
Average Tone MAE:        0.0523
Average Grammar MAE:     0.0523    ← Should be < 0.10

✅ PASS: Abstractor performance is GOOD
```

**Time**: ~20 seconds

**Outputs**: `visualizations/abstractor_test/` - Bar charts of predicted vs ground truth scores

---

### 3. test_decoder.py
**Purpose**: Validates that the decoder generates realistic design images

**What it tests**:
- Can it generate images from metadata?
- Do generated images look like designs?
- Are colors and layouts reasonable?

**How to run**:
```bash
python test_decoder.py
```

**Interactive prompts**:
- Generate 4 custom designs? [Y/n]
- Generate 5 designs from dataset? [Y/n]

**Expected output**:
- Generated design images
- Ground truth comparisons
- **Manual inspection required!**

**Time**: ~3-5 minutes (depending on number of samples)

**Outputs**: `visualizations/decoder_test/` - Generated and ground truth images

**What to look for**:
- ✓ Images should have visible elements (text areas, shapes)
- ✓ Colors should make sense
- ✓ Layouts should look organized
- ✗ Don't expect perfect designs - synthetic data has limitations

---

### 4. test_pipeline.py
**Purpose**: Validates that ALL THREE models work together

**What it tests**:
- Complete loop: `V_Meta → Decoder → P_Image → Encoder → F_Tensor → Abstractor → V_Grammar`
- Does generated image pass through encoder?
- Do extracted features produce reasonable grammar scores?
- Is metadata consistent throughout the pipeline?

**How to run**:
```bash
python test_pipeline.py
```

**Expected output**:
```
PIPELINE TEST SUMMARY
=====================
Average Grammar Score: 0.623     ← Should be > 0.5
Metadata Consistency:  67%       ← Should be > 60%

✅ PASS: Pipeline is working correctly!
   All models integrate successfully.
   ✓ Ready for Module 9: Innovation Loop
```

**Time**: ~3-5 minutes

**Outputs**: `visualizations/pipeline_test/` - Complete pipeline visualizations showing input → all stages → output

---

## Testing Workflow

### Step 1: Quick Sanity Check (5 minutes)
Run all tests to see if basic functionality works:

```bash
# Run each test script
python test_encoder.py
python test_abstractor.py
python test_decoder.py      # Skip generation prompts (press 'n')
python test_pipeline.py     # Skip (press 'n')
```

**What you're checking**:
- ✓ All checkpoints load without errors
- ✓ Models run without crashing
- ✓ Output shapes are correct

### Step 2: Full Evaluation (15-20 minutes)
If sanity checks pass, run full tests with generation:

```bash
python test_encoder.py      # Full run
python test_abstractor.py   # Full run
python test_decoder.py      # Press 'Y' for both prompts
python test_pipeline.py     # Press 'Y'
```

**What you're checking**:
- Encoder IoU > 0.6
- Abstractor Grammar MAE < 0.10
- Decoder generates reasonable images
- Pipeline integration works

### Step 3: Visual Inspection
Review the visualizations:

```bash
open visualizations/encoder_test/
open visualizations/abstractor_test/
open visualizations/decoder_test/
open visualizations/pipeline_test/
```

**Look for**:
- Encoder correctly identifies regions
- Grammar scores make sense (good designs = higher scores)
- Generated images look like designs (not noise)
- Pipeline maintains consistency

---

## Pass Criteria

### Minimum Requirements for Module 9:

| Test | Metric | Minimum | Good | Excellent |
|------|--------|---------|------|-----------|
| **Encoder** | Text/Image IoU | > 0.4 | > 0.6 | > 0.8 |
| **Abstractor** | Grammar MAE | < 0.15 | < 0.10 | < 0.05 |
| **Decoder** | Visual Quality | Visible elements | Clear designs | Professional look |
| **Pipeline** | Avg Grammar | > 0.3 | > 0.5 | > 0.7 |
| **Pipeline** | Consistency | > 50% | > 60% | > 80% |

### Decision Matrix:

**✅ All Pass** → Proceed to Module 9!

**⚠️ Some Warning** → Module 9 will work but may not optimize well. Consider:
- Retraining weak models
- Adjusting hyperparameters in Module 9
- Lower expectations for final quality

**❌ Any Fail** → Do NOT proceed to Module 9. Instead:
1. Identify which model failed
2. Check training logs for that model
3. Consider retraining with adjusted hyperparameters
4. Debug model architecture or data issues

---

## Common Issues & Solutions

### Issue: Encoder IoU < 0.4
**Symptom**: Encoder can't identify text or image regions accurately

**Solutions**:
- Check if encoder was trained long enough
- Verify synthetic data quality (are masks correct?)
- Consider training with more data
- Check if masks in dataset are actually correct

### Issue: Abstractor Grammar MAE > 0.15
**Symptom**: Grammar predictions are way off

**Solutions**:
- Check if abstractor was trained on correct F_Tensors
- Verify grammar score calculations in training data
- Train for more epochs
- Check for overfitting (high train accuracy, low val)

### Issue: Decoder generates noise/garbage
**Symptom**: Generated images don't look like anything

**Solutions**:
- Verify decoder training didn't fail early
- Check if diffusion schedule is correct
- Increase num_inference_steps in test (try 100 instead of 50)
- Model may need more training epochs

### Issue: Pipeline grammar scores all low
**Symptom**: Pipeline works but all grammar scores < 0.3

**Solutions**:
- Generated images may be low quality (decoder issue)
- Encoder may not extract features properly
- Abstractor may predict poorly
- Run individual tests to isolate the problem

---

## Next Steps

Once all tests pass:

### ✅ All Green?
→ You're ready for Module 9: Innovation Loop!
→ This is where the AI will actually "design" by optimizing towards better grammar scores

### ⚠️ Some Yellow?
→ Module 9 will work but results may be mediocre
→ Decide if you want to proceed or improve models first

### ❌ Any Red?
→ Fix the failing models before Module 9
→ Module 9 requires all three models working together

---

## Questions to Ask After Testing:

1. **Do the encoder's predictions look reasonable?**
   - Are text masks actually covering text?
   - Are colors correctly identified?

2. **Do the abstractor's scores make sense?**
   - Do messy designs get lower scores?
   - Do clean designs get higher scores?

3. **Do the decoder's images look like designs?**
   - Can you see elements?
   - Do colors work?
   - Is there any layout structure?

4. **Does the pipeline maintain consistency?**
   - If you input "poster", does it predict "poster"?
   - Are grammar scores in a reasonable range?

If you answer "yes" to most of these, you're ready for Module 9! 🚀

---

## Summary

**Test scripts created:**
- ✅ `test_encoder.py` - Tests structural feature extraction
- ✅ `test_abstractor.py` - Tests grammar score prediction
- ✅ `test_decoder.py` - Tests image generation
- ✅ `test_pipeline.py` - Tests complete integration

**Time investment:**
- Quick check: 5 minutes
- Full evaluation: 15-20 minutes
- Well worth it to avoid wasting time on broken Module 9!

**Remember**: Module 9 is the "magic" where AI optimizes designs. But magic only works if the foundation (these three models) is solid!
