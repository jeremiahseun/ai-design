#!/usr/bin/env python3
"""
Design Generator - Simple Interface

Generate professional designs by specifying your requirements.
The design system automatically applies:
- OkLCH color theory with emotional mapping
- Modular typography scales
- Adaptive layout positioning
- WCAG accessibility compliance
"""

import sys
import os
from unittest.mock import MagicMock

# Mock heavy dependencies (only needed if you don't have them installed)
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.agents.design_spec import DesignSpec, DesignContent
from src.designers.compositional_designer import CompositionalDesigner

def generate_design(
    headline: str,
    subheading: str = "",
    details: str = "",
    goal: str = "persuade",
    format: str = "poster",
    tone: str = "balanced",
    output_filename: str = "output_design.png"
):
    """
    Generate a design with your specifications.

    Args:
        headline: Main headline text
        subheading: Secondary text (optional)
        details: Body/details text (optional)
        goal: Design goal - "inform", "persuade", "entertain", or "inspire"
        format: Output format - "poster", "social", "flyer", or "banner"
        tone: Design tone - "calm", "balanced", or "energetic"
        output_filename: Name of output file (default: output_design.png)

    Returns:
        PIL Image object
    """

    # Map string inputs to numeric values
    goal_map = {
        "inform": 0,
        "persuade": 1,
        "entertain": 2,
        "inspire": 3
    }

    format_map = {
        "poster": 0,
        "social": 1,
        "flyer": 2,
        "banner": 3
    }

    tone_map = {
        "calm": 0.2,
        "balanced": 0.5,
        "energetic": 0.8
    }

    # Create design spec
    spec = DesignSpec(
        goal=goal_map.get(goal.lower(), 1),
        format=format_map.get(format.lower(), 0),
        tone=tone_map.get(tone.lower(), 0.5),
        style_prompt=f"{tone} {goal} design",
        content=DesignContent(
            headline=headline,
            subheading=subheading,
            details=details
        )
    )

    print("="*70)
    print("🎨 DESIGN GENERATOR")
    print("="*70)
    print(f"\n📋 Your Specifications:")
    print(f"  Goal: {spec.goal_name}")
    print(f"  Format: {spec.format_name}")
    print(f"  Tone: {spec.tone_description}")
    print(f"  Headline: '{headline}'")
    if subheading:
        print(f"  Subheading: '{subheading}'")
    if details:
        print(f"  Details: '{details}'")

    print(f"\n🔧 Initializing Design System...")
    designer = CompositionalDesigner(device="cpu")

    print(f"\n🎨 Generating Design...")
    print("-"*70)

    try:
        image = designer.create_design(spec)

        print("-"*70)
        print(f"\n✅ SUCCESS! Design generated.")
        print(f"  Size: {image.size}")

        # Save
        image.save(output_filename)
        print(f"  Saved to: {output_filename}")

        print(f"\n🎯 Applied Features:")
        print(f"  ✓ Smart layout selection")
        print(f"  ✓ OkLCH color system (emotional mapping)")
        print(f"  ✓ Typography engine (modular scale)")
        print(f"  ✓ Adaptive zones (content-aware)")
        print(f"  ✓ WCAG contrast enforcement")

        return image

    except Exception as e:
        print(f"\n❌ Error generating design: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Example usage - modify these values to generate your design!

    generate_design(
        headline="Genesis Conference",
        subheading="The first of its kind",
        details="This is a way we are working on. Genesis is here for all",
        goal="persuade",      # Options: inform, persuade, entertain, inspire
        format="social",      # Options: poster, social, flyer, banner
        tone="energetic",     # Options: calm, balanced, energetic
        output_filename="genesis_design.png"
    )
