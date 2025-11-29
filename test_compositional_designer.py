"""
Test Compositional Designer
"""

from src.designers.compositional_designer import CompositionalDesigner
from src.agents.design_spec import DesignSpec, DesignContent
from pathlib import Path

def main():
    print("="*50)
    print("TESTING COMPOSITIONAL DESIGNER")
    print("="*50)

    designer = CompositionalDesigner(device="mps")

    # Test Case 1: Conference Poster
    spec1 = DesignSpec(
        goal=0,
        format=0, # Poster
        tone=0.8,
        style_prompt="futuristic robot head, cyberpunk style, neon lights",
        content=DesignContent(
            headline="AI REVOLUTION 2025",
            subheading="The Future is Now",
            details="Nov 15-17 | Tokyo, Japan"
        )
    )

    # Test Case 2: Long Text (Overflow Test)
    spec2 = DesignSpec(
        goal=0,
        format=3, # Banner
        tone=0.2,
        style_prompt="calm zen garden, stones and water, minimal",
        content=DesignContent(
            headline="MINDFULNESS AND MEDITATION RETREAT",
            subheading="Join us for a week of absolute silence and inner peace in the mountains",
            details="Register now at www.example.com/retreat"
        )
    )

    output_dir = Path("visualizations/compositional_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Design 1 (Poster)...")
    img1 = designer.create_design(spec1)
    img1.save(output_dir / "poster_test.png")
    print("Saved poster_test.png")

    print("\nGenerating Design 2 (Banner - Overflow Check)...")
    img2 = designer.create_design(spec2)
    img2.save(output_dir / "banner_test.png")
    print("Saved banner_test.png")

    print("\nDone!")

if __name__ == "__main__":
    main()
