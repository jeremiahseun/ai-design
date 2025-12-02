#!/usr/bin/env python3
"""
Design Generator V2 (Natural Language)

Generate designs using natural language descriptions.
Uses Gemini Flash to parse requests and Local CLIP for style matching.
"""

import sys
import os
import argparse
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["diffusers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Load .env manually to avoid dependency
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"\'')

from src.agents.design_parser import DesignParser
from src.designers.compositional_designer import CompositionalDesigner
from src.intelligence.design_knowledge import DesignKnowledge

def generate(prompt: str, output: str = "output.png"):
    print("="*70)
    print("🎨 AI DESIGN GENERATOR V2")
    print("="*70)

    # 1. Parse Request
    print(f"\n🗣️  Parsing Request: '{prompt}'")
    parser = DesignParser() # Expects GEMINI_API_KEY env var
    spec = parser.parse_request(prompt)

    print(f"\n📋 Generated Specification:")
    print(f"  Goal: {spec.goal_name}")
    print(f"  Format: {spec.format_name}")
    print(f"  Tone: {spec.tone_description} ({spec.tone})")
    print(f"  Style Keywords: '{spec.style_prompt}'")
    print(f"  Elements: {len(spec.content.elements)}")
    for e in spec.content.elements:
        print(f"    - [{e.role.upper()}] {e.text[:40]}...")

    # 2. Design Knowledge Lookup (CLIP)
    print(f"\n🧠 Consulting Design Knowledge (Vector Space)...")
    dk = DesignKnowledge()
    recs = dk.get_recommendations(spec.style_prompt)
    print(f"  Recommended Font: {recs['font']}")
    print(f"  Recommended Color: {recs['color']}")
    print(f"  Recommended Layout: {recs['layout']}")

    # 3. Generate Design (Iterative)
    print(f"\n🎨 Generating Design (Iterative Mode)...")
    from src.designers.iterative_designer import IterativeDesigner

    designer = IterativeDesigner(device="cpu")

    # Run iterative loop
    final_image = designer.create_design_with_refinement(
        spec,
        max_iterations=3,
        output_prefix=output.replace(".png", "")
    )

    if final_image:
        final_image.save(output)
        print(f"\n✅ Final Result Saved to: {output}")
    else:
        print("\n❌ Failed to generate design.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate designs from natural language")
    parser.add_argument("prompt", help="Description of the design you want")
    parser.add_argument("--output", "-o", default="output.png", help="Output filename")

    args = parser.parse_args()
    generate(args.prompt, args.output)
