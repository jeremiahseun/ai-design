"""
Test Refined Procedural Background Generator
"""

from src.generators.procedural_background import RefinedBackgroundGenerator
from pathlib import Path

def main():
    print("="*50)
    print("TESTING REFINED PROCEDURAL BACKGROUND GENERATOR")
    print("="*50)

    generator = RefinedBackgroundGenerator()
    output_dir = Path("visualizations/procedural_bg_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test different styles
    tests = [
        {"name": "minimalist", "tone": 0.2, "goal": 0},
        {"name": "boho", "tone": 0.5, "goal": 3},
        {"name": "memphis", "tone": 0.9, "goal": 1},
        {"name": "cyber", "tone": 0.7, "goal": 3},
    ]

    for test in tests:
        print(f"\nGenerating {test['name']} background...")
        bg = generator.generate(1080, 1350, tone=test['tone'], goal=test['goal'])

        path = output_dir / f"{test['name']}_bg.png"
        bg.save(path)
        print(f"Saved: {path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
