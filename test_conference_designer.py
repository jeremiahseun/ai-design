"""
Test Conference Designer with Real-World Use Cases
Generates 4 variants for each of the 4 conferences
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from designers.conference_designer import ConferenceDesigner

def main():
    print("=" * 70)
    print("CONFERENCE DESIGNER TEST - 4 Real-World Events")
    print("=" * 70)

    # Initialize designer
    designer = ConferenceDesigner(device="mps")

    # Define conferences
    conferences = [
        {
            "name": "Lagos DevCon",
            "event_name": "Lagos DevCon 2025",
            "tagline": "Building the Next Billion-User Products",
            "details": "Aug 12–14, 2025 | Landmark Centre, Lagos, Nigeria",
            "logo_path": None,
            "style": "Modern tech aesthetic, deep green and charcoal theme, bold typography, tech conference poster, African tech summit"
        },
        {
            "name": "Revive Africa",
            "event_name": "Revive Africa Conference 2025",
            "tagline": "A Call to Renewal and Purpose",
            "details": "May 2–4, 2025 | Accra International Conference Centre, Ghana",
            "logo_path": None,
            "style": "Warm hopeful visuals, gold and white tones, serene faith-inspired design, church conference poster, spiritual event"
        },
        {
            "name": "Berlin FinTech",
            "event_name": "Berlin FinTech Forum 2025",
            "tagline": "Reinventing Digital Finance in Europe",
            "details": "Sept 18–20, 2025 | Messe Berlin, Germany",
            "logo_path": "assets/logo.png",
            "style": "Professional minimalistic European design, silver and navy palette, corporate finance conference, modern business"
        },
        {
            "name": "Stockholm Creative",
            "event_name": "Stockholm Creative Future Expo 2025",
            "tagline": "Design, Media & The Ideas That Move Us",
            "details": "June 26–28, 2025 | Waterfront Congress Centre, Stockholm, Sweden",
            "logo_path": "assets/logoo.png",
            "style": "Clean Scandinavian look, airy layout, pastel accents, high-contrast photography, creative expo poster, design conference"
        }
    ]

    # Generate designs
    for i, conf in enumerate(conferences):
        print(f"\n{'='*70}")
        print(f"Conference {i+1}/4: {conf['name']}")
        print(f"{'='*70}")

        # Create output directory
        output_dir = Path(f"visualizations/conferences/{conf['name'].lower().replace(' ', '_')}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate 4 variants
        try:
            designs = designer.create_design(
                event_name=conf['event_name'],
                tagline=conf['tagline'],
                details=conf['details'],
                logo_path=conf['logo_path'],
                style=conf['style'],
                variants=4
            )

            # Save each variant
            for j, design in enumerate(designs):
                filename = output_dir / f"variant_{j+1}.png"
                design.save(filename)
                print(f"  ✅ Saved: {filename}")

        except Exception as e:
            print(f"  ❌ Error generating {conf['name']}: {e}")

    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print(f"{'='*70}")
    print("Check visualizations/conferences/ for all designs")

if __name__ == "__main__":
    main()
