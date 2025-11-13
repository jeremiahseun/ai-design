# Design Tensor Framework (DTF): Master Implementation Specification

**Target System:** Apple Silicon (M3 Pro) | **Framework:** PyTorch (MPS) | **Language:** Python

## 1. Project Overview

We are building a Proof-of-Concept (POC) for a "Design Tensor Framework." The core hypothesis is that graphic design can be decomposed into Structural ($\mathcal{F}$) and Semantic ($\mathbf{V}$) components. We will use a **Synthetic Data Strategy** to bypass data annotation bottlenecks.

**Global Device Context:**
All code must check for and prioritize `torch.device("mps")`. Fallback to CPU only if necessary.

## 2. Directory Structure

The project must follow this structure:

```text
dtf_project/
├── data/                   # Storage for synthetic dataset
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py      # Data contracts (Module 0)
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── generator.py    # JSON Source of Truth (Module 1)
│   │   ├── renderer.py     # JSON -> P_Image (Module 2)
│   │   ├── extractor.py    # JSON -> F_Tensor (Module 3)
│   │   ├── grammar.py      # F_Tensor -> V_Grammar (Module 4)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py      # U-Net (P -> F)
│   │   ├── abstractor.py   # CNN-Transformer (F -> V)
│   │   ├── decoder.py      # Conditional DDPM (V -> P)
│   ├── utils/
│       ├── visualization.py # Helper to view tensors
├── train_scripts/          # Training loops for Modules 5, 6, 7, 8
├── integration/            # Innovation loop (Module 9)
└── requirements.txt
````

## 3\. Data Contracts (`src/core/schemas.py`)

**Strict Tensor Definitions:**

1. **`P_Image`**: $[B, 3, 256, 256]$. FloatTensor, normalized $[0, 1]$.
2. **`F_Tensor`**: $[B, 4, 256, 256]$.
      * $K=0$: Text Mask (Binary)
      * $K=1$: Image/Shape Mask (Binary)
      * $K=2$: Color ID Map (Integer/Long, discrete classes)
      * $K=3$: Hierarchy Map (Float $[0, 1]$)
3. **`V_Meta`**: Dictionary containing:
      * `v_Goal` (int/one-hot), `v_Tone` (float), `v_Content` (embedding/str), `v_Format` (int).
4. **`V_Grammar`**: $[B, 4]$. FloatTensor $[0, 1]$.
      * Order: Alignment, Contrast, Whitespace, Hierarchy.

## 4\. Phase 1: The Synthetic Pipeline (Modules 1-4)

### Module 1: Generator (`src/generators/generator.py`)

* **Goal:** Create procedural "Source of Truth" JSONs.
* **Logic:** Randomly select layout (left/center), palette, and content strings.
* **Output:** A Python Dict containing `elements` (list of dicts with `pos`, `box`, `color_id`, `hierarchy`) and `meta` info.

### Module 2: Renderer (`src/generators/renderer.py`)

* **Tool:** Pillow (PIL).
* **Logic:** Draw the JSON elements onto a canvas. Text is text, images are placeholders (rectangles).
* **Output:** `P_Image` tensor.

### Module 3: Extractor (`src/generators/extractor.py`)

* **Constraint:** Do NOT infer from the image. Draw directly from JSON coordinates to tensors.
* **Logic:**
  * Create 4 empty channels $(H, W)$.
  * Fill channels based on element properties in JSON.
* **Output:** `F_Tensor`.

### Module 4: Grammar Engine (`src/generators/grammar.py`)

* **Dependencies:** OpenCV (`cv2`), NumPy.
* **Functions:**
    1. `calculate_alignment`: Histogram variance of x-coordinates (left/center/right).
    2. `calculate_contrast`: Sample pixel pairs (Text vs Background) using mask dilation; compute WCAG luminance ratio.
    3. `calculate_whitespace`: Distance transform on inverted mask; Gini coefficient of values.
    4. `calculate_hierarchy`: Cosine similarity between `F_Tensor[3]` (Target) and calculated visual weight (Size \* Contrast).

-----

## 5\. Phase 2: Model Architectures (Modules 5-7)

### Module 5: The Encoder (U-Net)

* **Input:** `P_Image` $[B, 3, 256, 256]$
* **Output:** `F_Tensor` $[B, 4, 256, 256]$
* **Loss:** `CompositeLoss` = `DiceLoss` (for Masks) + `CrossEntropy` (for ColorID) + `MSE` (for Hierarchy).

### Module 6: The Abstractor (ResNet + Heads)

* **Input:** `F_Tensor` $[B, 4, 256, 256]$
* **Architecture:** ResNet backbone -\> Flatten -\> Split into two MLP heads.
* **Head 1:** Predicts `V_Meta` (Classification/Regression).
* **Head 2:** Predicts `V_Grammar` (Regression).
* **Loss:** CE (Meta) + MSE (Grammar).

### Module 7: The Decoder (Conditional DDPM)

* **Architecture:** U-Net based Diffusion Model.
* **Conditioning:** `V_Meta` must be embedded and injected (via Cross-Attention or Timestep addition).
* **Input:** Noise $\epsilon$ + Condition.
* **Output:** Reconstructed `P_Image`.

-----

## 6\. Phase 3: Integration & Innovation

### Module 8: Fine-Tuning

* **Logic:** Freeze Encoder and Abstractor.
* **Loop:**
    1. `gen_img` = Decoder(`noise`, `meta`)
    2. `pred_grammar` = Abstractor(Encoder(`gen_img`))
    3. `loss` = `Diffusion_Loss` - $\lambda \times \text{sum}(\text{pred\_grammar})$
* **Goal:** Update Decoder weights to maximize grammar scores.

### Module 9: Innovation Loop (`src/integration/innovate.py`)

* **Logic:** Inference-time optimization.
* **Loop:**
    1. Fix models (Eval mode).
    2. Generate image from latent $z$.
    3. Calculate Grammar Score.
    4. Backprop to $z$ (Gradient Ascent).
    5. Update $z$.
* **Output:** A GIF showing the design "snapping" into alignment.
