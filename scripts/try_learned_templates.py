
"""
Simple script to test the new Data-Driven "Exemplar" Templates.
Run this to generate designs using the V3 learned patterns.
"""

import os
from src.designers.enhanced_compositional_designer import EnhancedCompositionalDesigner
from src.templates.template_library import TemplateLibrary

def test_templates():
    # 1. Initialize Designer
    print("Initializing Designer...")
    designer = EnhancedCompositionalDesigner()

    # 2. List available V3 (Exemplar) templates
    print("\nAvailable V3 Exemplar Templates:")
    v3_templates = designer.template_library.find_templates(tags=["v3"])
    for t in v3_templates:
        print(f" - {t.id}: {t.description[:50]}...")

    if not v3_templates:
        print("No V3 templates found. Run analysis first.")
        return

    # 3. Define a generic content brief
    brief = {
        "headline": "DESIGN INTELLIGENCE",
        "subheading": "Real-World Patterns",
        "body": "This layout was learned from professional designs using exemplar extraction.",
        "cta": "Explore Now",
        "caption": "V3 Exemplar Mode"
    }

    # 4. Generate a design for the first V3 template
    target_template = v3_templates[0] # Try the first one
    # Or specify ID: target_template = designer.template_library.get_template("learned_asymmetric_single_hero_center")

    print(f"\nGenerating design for: {target_template.id}")
    import json
    image = designer.design_from_template(json.dumps(brief), template_id=target_template.id)

    # 5. Save output
    output_path = f"generated_test_{target_template.id}.png"
    image.save(output_path)
    print(f"\nSuccess! Saved design to: {output_path}")

if __name__ == "__main__":
    test_templates()
