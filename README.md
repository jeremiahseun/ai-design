# Design Tensor Framework (DTF) - POC Implementation

A Proof-of-Concept for decomposing graphic design into **Structural** (F_Tensor) and **Semantic** (V_Meta/V_Grammar) components using synthetic data and deep learning.

**Target System:** Apple Silicon (M3 Pro) | **Framework:** PyTorch (MPS) | **Language:** Python

---

## Project Overview

The Design Tensor Framework tests the hypothesis that graphic design can be decomposed and learned through:

1. **Structural Features (F_Tensor)**: Spatial layout, element types, colors, hierarchy
2. **Semantic Features (V_Meta)**: Design goals, tone, content, format
3. **Grammar Scores (V_Grammar)**: Alignment, Contrast, Whitespace, Hierarchy

By using **synthetic data generation**, we bypass expensive data annotation and create perfect ground truth labels.

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run verification script
python verify_pipeline.py
```

This will:
- Test all Phase 1 modules
- Check MPS device availability
- Generate test visualizations in `data/`

### 3. Generate Dataset

```bash
# Quick test (100 samples)
python generate_dataset.py --test

# Full dataset (10,000 samples, ~30-60 minutes)
python generate_dataset.py --num_samples 10000
```

---

## Project Structure

```
ai-design/
├── data/                          # Generated datasets
│   └── synthetic_dataset/
│       ├── images/                # P_Image (rendered designs)
│       ├── f_tensors/             # F_Tensor (structural features)
│       ├── metadata/              # V_Meta + V_Grammar scores
│       └── visualizations/        # Side-by-side visualizations
│
├── src/
│   ├── core/
│   │   └── schemas.py            # Module 0: Data contracts
│   ├── generators/
│   │   ├── generator.py          # Module 1: JSON generator
│   │   ├── renderer.py           # Module 2: JSON → P_Image
│   │   ├── extractor.py          # Module 3: JSON → F_Tensor
│   │   └── grammar.py            # Module 4: F_Tensor → V_Grammar
│   ├── models/                   # Phase 2: Neural networks
│   │   ├── encoder.py            # Module 5: U-Net (P → F)
│   │   ├── abstractor.py         # Module 6: ResNet (F → V)
│   │   └── decoder.py            # Module 7: DDPM (V → P)
│   └── utils/
│       └── visualization.py      # Helper utilities
│
├── train_scripts/                # Training loops
├── integration/                  # Module 9: Innovation loop
├── generate_dataset.py           # Dataset generation script
└── verify_pipeline.py            # Pipeline verification
```

---

## Architecture

### Phase 1: Synthetic Data Pipeline ✅ COMPLETE

1. **Generator** (Module 1): Creates procedural design briefs as JSON
2. **Renderer** (Module 2): Renders JSON → RGB image (P_Image)
3. **Extractor** (Module 3): Extracts JSON → Structural features (F_Tensor)
4. **Grammar Engine** (Module 4): Calculates design quality scores (V_Grammar)

### Phase 2: Model Training 🚧 PENDING

5. **Encoder** (Module 5): U-Net that learns P_Image → F_Tensor
6. **Abstractor** (Module 6): ResNet that learns F_Tensor → V_Grammar + V_Meta
7. **Decoder** (Module 7): Conditional DDPM that generates P_Image from V_Meta

### Phase 3: Integration 🚧 PENDING

8. **Fine-Tuning** (Module 8): RL-style loop to maximize grammar scores
9. **Innovation Loop** (Module 9): Gradient ascent on latent space for design optimization

---

## Data Contracts

### P_Image (Rendered Design)
- **Shape**: `[B, 3, 256, 256]`
- **Type**: `FloatTensor`
- **Range**: `[0, 1]` normalized

### F_Tensor (Structural Features)
- **Shape**: `[B, 4, 256, 256]`
- **Channels**:
  - `K=0`: Text Mask (Binary)
  - `K=1`: Image/Shape Mask (Binary)
  - `K=2`: Color ID Map (Integer classes)
  - `K=3`: Hierarchy Map (Float `[0, 1]`)

### V_Meta (Semantic Metadata)
- `v_Goal`: Design goal/purpose (int, one-hot)
- `v_Tone`: Emotional tone (float `[0, 1]`)
- `v_Content`: Content description (string/embedding)
- `v_Format`: Format type (int: poster, social, flyer, etc.)

### V_Grammar (Design Quality Scores)
- **Shape**: `[B, 4]`
- **Dimensions**: `[Alignment, Contrast, Whitespace, Hierarchy]`
- **Range**: `[0, 1]` (higher is better)

---

## Grammar Scoring System

### 1. Alignment
- Measures clustering of elements along grid lines
- Uses histogram variance of x-coordinates
- Lower variance = better alignment

### 2. Contrast
- Measures text readability using WCAG luminance ratios
- Samples text vs background pixels
- Good contrast: ratio ≥ 4.5 (WCAG AA)

### 3. Whitespace
- Measures distribution of empty space
- Uses distance transform + Gini coefficient
- Uniform distribution = better whitespace usage

### 4. Hierarchy
- Measures visual weight alignment with intended hierarchy
- Compares designed hierarchy (F_Tensor[3]) with calculated visual weight
- Uses cosine similarity

---

## Development Workflow

### Current Status: Phase 1 Complete ✅

All synthetic data generation components are implemented and tested.

### Next Steps: Phase 2 Implementation

1. **Implement U-Net Encoder** (Module 5)
   ```bash
   # Will be implemented next
   python train_scripts/train_encoder.py
   ```

2. **Implement Abstractor** (Module 6)
   ```bash
   python train_scripts/train_abstractor.py
   ```

3. **Implement Conditional DDPM** (Module 7)
   ```bash
   python train_scripts/train_decoder.py
   ```

---

## Testing & Validation

### Run Individual Module Tests

```bash
# Test schemas
python test_schemas.py

# Test generator
python -m src.generators.generator

# Test renderer
python -m src.generators.renderer

# Test extractor
python -m src.generators.extractor

# Test grammar engine
python -m src.generators.grammar
```

### Inspect Visualizations

After running `verify_pipeline.py` or `generate_dataset.py`, check:

- `data/verify_combined.png` - Side-by-side visualization
- `data/synthetic_dataset/visualizations/` - Dataset samples

---

## Troubleshooting

### MPS (Apple Silicon) Issues

**Error**: `Placeholder storage has not been allocated on MPS device`
```python
# Solution: Explicitly move tensors to device
tensor = tensor.to(device)
```

**Error**: `Op not implemented on MPS`
```python
# Solution: Enable fallback to CPU
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Memory Issues

If you encounter OOM errors during dataset generation:
```bash
# Generate in smaller batches
python generate_dataset.py --num_samples 1000
```

---

## Technical Details

### Why Synthetic Data?

1. **Perfect Ground Truth**: No annotation errors
2. **Infinite Scale**: Generate millions of samples
3. **Controlled Variation**: Test specific design principles
4. **No Copyright Issues**: All data is procedurally generated

### Design Philosophy

- **JSON as Source of Truth**: Everything derives from the design JSON
- **No Inference in Extraction**: F_Tensor is drawn directly from coordinates, not inferred
- **Modular Pipeline**: Each component can be tested independently
- **MPS-First**: Optimized for Apple Silicon, with fallbacks

---

## Citation & References

This is a proof-of-concept implementation. If you use this code, please cite:

```
Design Tensor Framework (DTF)
A synthetic data approach to learning design principles
2024
```

---

## License

MIT License - See LICENSE file for details

---

## Contributing

This is a research POC. Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

---

## Contact & Support

For questions or issues, please open a GitHub issue.

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🚧
