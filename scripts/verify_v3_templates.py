
import os
import sys
from PIL import Image

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.templates.template_library import TemplateLibrary
from src.designers.enhanced_compositional_designer import EnhancedCompositionalDesigner
from src.analysis.style_profiles import StyleProfile

# Load .env manually
import os
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("CLAUDE_API_KEY="):
                os.environ["CLAUDE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")

def verify_learned_templates():
    print("Initializing Designer and Library...")
    library = TemplateLibrary()
    designer = EnhancedCompositionalDesigner()

    # Reload templates to be sure (though init does it)
    print(f"Loaded {len(library.templates)} templates.")

    # We specifically look for "v3" tags
    learned = library.find_templates(tags=["v3"])
    print(f"Found {len(learned)} learned V3 templates.")

    if not learned:
        print("No learned V3 templates found! Check loading logic.")
        return

    output_dir = "data/verification/learned_v3"
    os.makedirs(output_dir, exist_ok=True)

    dummy_Brief = {
        "headline": "LEARN DESIGN",
        "subheading": "Data Driven Art",
        "body": "This layout was extracted from real analysis.",
        "cta": "Start Now",
        "caption": "Verification v1",
        "recommended_style": "Modern"
    }

    import json
    for tmpl in learned:
        print(f"Generating image for {tmpl.id}...")
        try:
            # design_from_template expects (brief: str, template_id: str)
            image = designer.design_from_template(json.dumps(dummy_Brief), template_id=tmpl.id)

            # Save
            filename = f"{output_dir}/{tmpl.id}.png"
            image.save(filename)
            print(f"Saved to {filename}")
        except Exception as e:
            print(f"Failed to generate {tmpl.id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_learned_templates()
