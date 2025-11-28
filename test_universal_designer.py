"""
Test Universal Designer with Natural Language Prompts
"""

import os
from pathlib import Path
from src.designers.universal_designer import UniversalDesigner

def main():
    print("=" * 70)
    print("UNIVERSAL DESIGNER TEST - Natural Language Interface")
    print("=" * 70)

    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("\n❌ GOOGLE_API_KEY environment variable not set!")
        print("Please set it with: export GOOGLE_API_KEY='your-key-here'")
        return

    # Initialize
    designer = UniversalDesigner(device="mps")

    # Define test prompts
    prompts = [
        # Simple conference
        {
            "name": "Simple Conference",
            "prompt": "Create a poster for DevFest 2025, a tech conference happening June 10-12 in San Francisco. Make it modern and professional with blue tones."
        },
        # Church event
        {
            "name": "Church Event",
            "prompt": "I need a flyer for our annual Harvest Festival at Grace Community Church on October 15th. Should feel warm, welcoming, family-friendly with gold and autumn colors."
        },
        # Product launch
        {
            "name": "Product Launch",
            "prompt": "Design a bold, energetic social media post announcing the launch of our new AI-powered fitness app 'FitGenius'. Use vibrant greens and blacks, make it exciting!"
        },
        # Minimalist poster
        {
            "name": "Minimalist Art",
            "prompt": "Create an elegant, minimal banner for an art gallery exhibition titled 'Silence in Motion'. Use lots of whitespace, calm pastel tones, very sophisticated."
        }
    ]

    # Generate designs
    for i, test in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"Test {i+1}/4: {test['name']}")
        print(f"{'='*70}")
        print(f"Prompt: {test['prompt']}")

        try:
            # Create output directory
            output_dir = Path(f"visualizations/universal_test/{test['name'].lower().replace(' ', '_')}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate (2 variants)
            spec, designs = designer.create_design(test['prompt'], variants=2)

            # Save spec
            spec_path = output_dir / "spec.json"
            with open(spec_path, 'w') as f:
                f.write(spec.to_json())
            print(f"\n📄 Specification saved: {spec_path}")
            print(spec)

            # Save designs
            for j, design in enumerate(designs):
                design_path = output_dir / f"variant_{j+1}.png"
                design.save(design_path)
                print(f"🎨 Design saved: {design_path}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print(f"{'='*70}")
    print("Check visualizations/universal_test/ for all results")

if __name__ == "__main__":
    main()
