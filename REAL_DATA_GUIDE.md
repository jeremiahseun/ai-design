# Using Real Designer Data - Implementation Guide

## Why Real Data?

The synthetic data approach is a POC, but has limitations:
- Simple geometric designs only
- Doesn't capture real design principles
- Decoder performance suffers

**To create professional-quality designs, you need to learn from professionals!**

---

## Option 1: Retrain Decoder Only (Recommended)

### Overview
- Keep: Encoder ✅ + Abstractor ✅ (they work on real images too!)
- Retrain: Decoder only
- Data needed: Real design PNGs + metadata labels

### Step 1: Collect Real Designs

Gather 1000-5000 professional designs:
- Posters, social media posts, flyers, banners
- Diverse styles, colors, layouts
- High resolution (ideally 512×512 or larger)

**Sources:**
- Your own design portfolio
- Public design databases (Behance, Dribbble - check licenses!)
- Stock design websites (with proper licensing)
- Commission designers (ethical + legal)

### Step 2: Label Metadata

For each design, create a JSON file:

```json
{
  "filename": "design_001.png",
  "v_meta": {
    "v_Goal": 1,     // 0-9: Inform, Persuade, Entertain, etc.
    "v_Format": 0,   // 0-3: Poster, Social, Flyer, Banner
    "v_Tone": 0.8    // 0-1: Calm to Energetic
  }
}
```

**Labeling tips:**
- Goal: What's the design trying to achieve?
- Format: What medium is it for?
- Tone: How energetic/bold does it feel? (0=calm, 1=bold)

**Time estimate:** 30 seconds per design = 8 hours for 1000 designs

### Step 3: Prepare Dataset

Create dataset structure:
```
data/real_designs/
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── metadata/
    ├── 000001.json
    ├── 000002.json
    └── ...
```

Resize all images to 256×256:
```python
from PIL import Image
from pathlib import Path

input_dir = Path("raw_designs")
output_dir = Path("data/real_designs/images")
output_dir.mkdir(parents=True, exist_ok=True)

for i, img_path in enumerate(input_dir.glob("*.png")):
    img = Image.open(img_path)
    img = img.resize((256, 256), Image.LANCZOS)
    img.save(output_dir / f"{i:06d}.png")
```

### Step 4: Update Decoder Training

Modify `train_scripts/train_decoder.py`:

```python
# Change dataset loader to use real data
from utils.real_dataset import RealDesignDataset  # You'd create this

train_loader, val_loader = create_real_dataloaders(
    data_dir='data/real_designs',
    batch_size=32,
    train_ratio=0.9
)
```

### Step 5: Train on Kaggle

- Upload `data/real_designs/` as Kaggle dataset
- Train for 100-150 epochs (longer than synthetic!)
- Expected time: 15-20 hours on T4 GPU

**Expected results:**
- Decoder generates actual designs (not noise!)
- Learns real color palettes, layouts, typography
- Innovation loop will produce real improvements

---

## Option 2: Full Real Data Pipeline (Advanced)

### Overview
Retrain all three models on real data with auto-generated labels.

### Auto-Labeling Pipeline

#### Step 1: Generate F_Tensors from Real Images

Use computer vision to extract structural features:

```python
import cv2
import pytesseract
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def auto_label_design(image_path):
    """Auto-generate F_Tensor from real design image"""
    img = cv2.imread(image_path)

    # Channel 0: Text Mask (OCR-based)
    # Use Tesseract to detect text regions
    text_data = pytesseract.image_to_boxes(img)
    text_mask = create_mask_from_boxes(text_data, img.shape)

    # Channel 1: Image/Shape Mask (Segmentation-based)
    # Use Detectron2 for object detection
    predictor = DefaultPredictor(cfg)  # Pre-trained model
    outputs = predictor(img)
    image_mask = create_mask_from_segments(outputs)

    # Channel 2: Color ID Map (K-means clustering)
    colors = extract_dominant_colors(img, n_colors=18)
    color_map = assign_pixels_to_colors(img, colors)

    # Channel 3: Hierarchy Map (Saliency-based)
    # Use visual saliency to estimate importance
    saliency = compute_saliency_map(img)
    hierarchy = normalize_to_01(saliency)

    f_tensor = np.stack([text_mask, image_mask, color_map, hierarchy])
    return f_tensor
```

**Tools needed:**
- `pytesseract` - OCR for text detection
- `detectron2` - Object detection/segmentation
- `opencv` - Image processing
- Saliency models (OpenCV or deep learning)

**Quality:**
- ⚠️ Not perfect (70-80% accurate)
- Good enough for training
- Can manually correct critical samples

#### Step 2: Calculate V_Grammar from F_Tensors

Use the existing grammar engine:

```python
from generators.grammar import GrammarEngine

grammar_engine = GrammarEngine()

# Auto-calculate scores from F_Tensor
v_grammar = grammar_engine.calculate_all_scores(f_tensor, p_image)
# Returns: [alignment, contrast, whitespace, hierarchy]
```

**This works automatically!** No manual labeling needed.

#### Step 3: Label V_Meta (Manual or Semi-Auto)

**Manual approach:**
- Label goal, format, tone for each design (30 sec/design)

**Semi-auto approach:**
- Use CLIP or similar model to classify:
```python
import clip

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("design.png")).unsqueeze(0)

# Classify goal
goals = ["informational", "promotional", "entertaining", "inspiring"]
text = clip.tokenize(goals)
logits_per_image, _ = model(image, text)
probs = logits_per_image.softmax(dim=-1)
predicted_goal = probs.argmax()
```

---

## Trade-Offs

### Synthetic Data
- ✅ Easy to generate
- ✅ Perfect labels
- ✅ Infinite scale
- ❌ Simple designs
- ❌ Poor decoder quality

### Real Data
- ✅ Professional quality
- ✅ Real design principles
- ✅ Better decoder results
- ❌ Need to collect data
- ❌ Labeling effort
- ❌ Copyright considerations

---

## Recommended Path Forward

### Phase 1: Quick Win (1-2 weeks)
1. Collect 1000 real designs
2. Label metadata only (8 hours)
3. Retrain decoder on real data
4. Test innovation loop with real decoder

**Expected outcome:** Decoder generates actual designs!

### Phase 2: Full Pipeline (1 month)
1. Auto-label F_Tensors with CV tools
2. Calculate V_Grammar automatically
3. Retrain all three models
4. Full DTF system on real data

**Expected outcome:** Professional-quality AI design system!

### Phase 3: Production (Ongoing)
1. Collect more diverse data (10K+ designs)
2. Fine-tune models continuously
3. Add style transfer capabilities
4. Deploy as API/service

---

## Legal & Ethical Considerations

### Copyright
- ⚠️ Can't use copyrighted designs without permission
- ✅ Use your own designs
- ✅ Use public domain designs
- ✅ Use properly licensed stock designs
- ✅ Commission original designs

### Attribution
- If using public designs, provide attribution
- Respect Creative Commons licenses
- Don't train on work without permission

### Ethical Use
- Don't replace human designers - augment them!
- Use for inspiration, not plagiarism
- Maintain human oversight on outputs

---

## Conclusion

**Your instinct is correct!** To make professional designs, you need professional training data.

**Start here:**
1. Collect 500-1000 real designs (your own or licensed)
2. Label metadata (8 hours work)
3. Retrain decoder (15 hours on Kaggle)
4. Test innovation loop

Once decoder works, you'll see the magic! The innovation loop will:
- Generate real designs
- Optimize real layouts
- Improve real quality

This is how you go from POC → Production! 🚀
