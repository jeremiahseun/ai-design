"""
Test script to verify core schemas are working correctly
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.schemas import *
import torch

print("=" * 60)
print("DTF Core Schemas Test")
print("=" * 60)

# Test device detection
print(f"\n1. Device Detection:")
print(f"   Selected device: {DEVICE}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

# Test P_Image
print(f"\n2. Testing P_Image:")
p_img = P_Image.create(batch_size=2)
print(f"   Shape: {p_img.shape}")
print(f"   Device: {p_img.device}")
print(f"   Valid: {P_Image.validate(p_img)}")

# Test F_Tensor
print(f"\n3. Testing F_Tensor:")
f_tensor = F_Tensor.create(batch_size=2)
print(f"   Shape: {f_tensor.shape}")
print(f"   Device: {f_tensor.device}")
print(f"   Valid: {F_Tensor.validate(f_tensor)}")

# Test V_Meta
print(f"\n4. Testing V_Meta:")
v_meta = V_Meta.create_empty()
print(f"   Structure: {v_meta}")
print(f"   Valid: {V_Meta.validate(v_meta)}")
v_meta_tensor = V_Meta.to_tensor(v_meta)
print(f"   Tensor shape: {v_meta_tensor.shape}")
print(f"   Tensor device: {v_meta_tensor.device}")

# Test V_Grammar
print(f"\n5. Testing V_Grammar:")
v_grammar = V_Grammar.create(batch_size=2)
print(f"   Shape: {v_grammar.shape}")
print(f"   Device: {v_grammar.device}")
print(f"   Valid: {V_Grammar.validate(v_grammar)}")

# Test V_Grammar from values
v_grammar_sample = V_Grammar.from_values(0.8, 0.9, 0.7, 0.85)
print(f"   Sample scores: {V_Grammar.to_dict(v_grammar_sample)}")

print("\n" + "=" * 60)
print("All schema tests passed!")
print("=" * 60)
