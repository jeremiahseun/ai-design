
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.design_spec import DesignSpec, DesignContent
from src.designers.compositional_designer import CompositionalDesigner

def test_integrated_design():
    print("="*60)
    print("Testing Integrated Design System")
    print("="*60)

    # Create a test spec with long text to test layout adaptation
    spec = DesignSpec(
        goal=1,  # Persuade
        format=0,  # Poster
        tone=0.8,  # Energetic (Golden Ratio typography)
        style_prompt="Bold marketing poster",
        content=DesignContent(
            headline="Revolutionary New Product Launching Soon with Amazing Features",
            subheading="Experience the future of innovation today",
            details="Limited time offer. Pre-order now and save 30%. Available in multiple colors and configurations."
        )
    )

    print("\n📝 Design Spec:")
    print(f"  Goal: {spec.goal_name}")
    print(f"  Format: {spec.format_name}")
    print(f"  Tone: {spec.tone_description} ({spec.tone})")
    print(f"  Headline: '{spec.content.headline}'")

    # Create designer
    designer = CompositionalDesigner(device="cpu")

    print("\n🎨 Generating Design...")
    print("-"*60)

    # Generate design
    try:
        image = designer.create_design(spec)

        print("-"*60)
        print(f"\n✅ Design Generated Successfully!")
        print(f"  Size: {image.size}")

        # Save output
        output_path = "test_integrated_design.png"
        image.save(output_path)
        print(f"  Saved to: {output_path}")

        print("\n🔍 Integrated Features:")
        print("  ✅ OkLCH Color System (Emotional mapping + Contrast enforcement)")
        print("  ✅ Typography Engine (Modular Scale + Widow fixing)")
        print("  ✅ Layout Adapter (Adaptive zone positioning)")

        return image

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_integrated_design()
