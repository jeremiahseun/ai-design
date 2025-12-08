"""
Phase 1 Pipeline Verification Script
Tests the complete data generation pipeline before full dataset generation.
This should be run after installing dependencies to ensure everything works.
"""

import sys
import os
import numpy as np

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 60)
print("DTF PHASE 1 PIPELINE VERIFICATION")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from core.schemas import *
    from generators.generator import DesignGenerator, COLOR_MAP
    from generators.renderer import DesignRenderer
    from generators.extractor import FeatureExtractor
    from generators.grammar import GrammarEngine
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check device
print("\n2. Checking compute device...")
print(f"   Device: {DEVICE}")
if DEVICE.type == "mps":
    print("   ✓ MPS (Apple Silicon) detected")
elif DEVICE.type == "cuda":
    print("   ✓ CUDA detected")
else:
    print("   ⚠ Using CPU (training will be slow)")

# Test 3: Generate design JSON
print("\n3. Testing JSON Generator (Module 1)...")
gen = DesignGenerator(seed=42)
design = gen.generate()
print(f"   Layout: {design['layout']}")
print(f"   Format: {design['format']}")
print(f"   Elements: {len(design['elements'])}")
print("   ✓ Design generated successfully")

# Test 4: Render to image
print("\n4. Testing Renderer (Module 2)...")
renderer = DesignRenderer()
img = renderer.render(design)
print(f"   Image shape: {img.shape}")
print(f"   Pixel range: [{img.min()}, {img.max()}]")
assert img.shape == (256, 256, 3), "Image shape mismatch"
assert img.min() >= 0 and img.max() <= 255, "Pixel values out of range"
print("   ✓ Rendering successful")

# Test 5: Extract F_Tensor
print("\n5. Testing Feature Extractor (Module 3)...")
extractor = FeatureExtractor()
f_tensor = extractor.extract(design)
print(f"   F_Tensor shape: {f_tensor.shape}")
print(f"   K=0 (Text Mask) pixels: {(f_tensor[0] > 0).sum()}")
print(f"   K=1 (Image Mask) pixels: {(f_tensor[1] > 0).sum()}")
print(f"   K=2 (Color IDs) unique: {len(np.unique(f_tensor[2]))}")
print(f"   K=3 (Hierarchy) range: [{f_tensor[3].min():.2f}, {f_tensor[3].max():.2f}]")
assert f_tensor.shape == (4, 256, 256), "F_Tensor shape mismatch"
print("   ✓ Feature extraction successful")

# Test 6: Calculate grammar scores
print("\n6. Testing Grammar Engine (Module 4)...")
grammar = GrammarEngine()
scores = grammar.calculate_all(f_tensor, img)
print("   Grammar Scores:")
for dimension, score in scores.items():
    bar = "█" * int(score * 20)
    print(f"     {dimension:12s}: {score:.3f} {bar}")
assert all(0 <= v <= 1 for v in scores.values()), "Scores out of [0, 1] range"
print("   ✓ Grammar calculation successful")

# Test 7: Test PyTorch tensor conversion
print("\n7. Testing PyTorch tensor conversion...")
try:
    import torch
    p_img_tensor = P_Image.from_numpy(img)
    print(f"   P_Image tensor shape: {p_img_tensor.shape}")
    print(f"   P_Image device: {p_img_tensor.device}")
    assert P_Image.validate(p_img_tensor), "P_Image validation failed"

    f_tensor_torch = torch.from_numpy(f_tensor).unsqueeze(0).to(DEVICE)
    print(f"   F_Tensor shape: {f_tensor_torch.shape}")
    print(f"   F_Tensor device: {f_tensor_torch.device}")
    assert F_Tensor.validate(f_tensor_torch), "F_Tensor validation failed"

    v_grammar = V_Grammar.from_values(
        scores['Alignment'],
        scores['Contrast'],
        scores['Whitespace'],
        scores['Hierarchy']
    )
    print(f"   V_Grammar shape: {v_grammar.shape}")
    print(f"   V_Grammar device: {v_grammar.device}")
    assert V_Grammar.validate(v_grammar), "V_Grammar validation failed"

    print("   ✓ PyTorch tensors created successfully")
except Exception as e:
    print(f"   ✗ PyTorch conversion failed: {e}")
    sys.exit(1)

# Test 8: Create visualization
print("\n8. Creating test visualizations...")
try:
    from PIL import Image

    # Save rendered image
    os.makedirs('data', exist_ok=True)
    renderer.save_image(img, 'data/verify_rendered.png')
    print("   Saved: data/verify_rendered.png")

    # Save F_Tensor visualization
    f_vis = extractor.visualize_channels(f_tensor)
    vis_img = Image.fromarray(f_vis)
    vis_img.save('data/verify_f_tensor.png')
    print("   Saved: data/verify_f_tensor.png")

    # Save side-by-side
    combined = np.concatenate([img, f_vis], axis=1)
    combined_img = Image.fromarray(combined.astype(np.uint8))
    combined_img.save('data/verify_combined.png')
    print("   Saved: data/verify_combined.png")

    print("   ✓ Visualizations created successfully")
except Exception as e:
    print(f"   ✗ Visualization failed: {e}")

# Test 9: Batch processing
print("\n9. Testing batch processing...")
designs = gen.generate_batch(5)
print(f"   Generated {len(designs)} designs")

batch_imgs = renderer.render_batch(designs)
print(f"   Rendered batch shape: {batch_imgs.shape}")

batch_f = extractor.extract_batch(designs)
print(f"   F_Tensor batch shape: {batch_f.shape}")

batch_scores = grammar.calculate_batch(batch_f, batch_imgs)
print(f"   Grammar batch shape: {batch_scores.shape}")
print(f"   Mean scores: {batch_scores.mean(axis=0)}")

print("   ✓ Batch processing successful")

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✓ All Phase 1 components are working correctly!")
print("\nNext Steps:")
print("  1. Inspect visualizations in data/verify_*.png")
print("  2. Run: python generate_dataset.py --test")
print("     (Generates 100 samples for quick testing)")
print("  3. Run: python generate_dataset.py --num_samples 10000")
print("     (Generate full dataset - takes ~30-60 minutes)")
print("\nAfter dataset generation, proceed to Phase 2 (Model Training)")
print("=" * 60)
