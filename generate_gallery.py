"""
Generate Gallery
Visual verification for the Layout Engine.

Generates a 3x3 grid of designs using the same content but different layouts.
"""

import sys
import os
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["diffusers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.agents.design_spec import DesignSpec, DesignContent, DesignElement
from src.designers.compositional_designer import CompositionalDesigner
from src.layouts.layout_engine import LayoutEngine

def generate_gallery():
    print("="*60)
    print("🖼️  GENERATING LAYOUT GALLERY")
    print("="*60)

    # 1. Create a standard spec
    spec = DesignSpec(
        goal=1, # Persuade
        format=0, # Poster
        tone=0.8, # Energetic
        style_prompt="Modern tech design",
        content=DesignContent(elements=[
            DesignElement("Future of AI", "primary"),
            DesignElement("Experience the revolution", "secondary"),
            DesignElement("Join us for a deep dive into generative design systems.", "tertiary"),
            DesignElement("Dec 2025 • San Francisco", "tertiary")
        ])
    )

    # 2. Initialize Designer
    designer = CompositionalDesigner(device="cpu")

    # 3. Get all layouts
    layout_names = [
        "split_horizontal", "split_vertical", "central_hero",
        "typographic_bold", "modern_clean", "magazine_grid",
        "diagonal_split", "asymmetric_editorial", "modern_geometric"
    ]

    # 4. Generate images
    images = []
    for name in layout_names:
        print(f"🎨 Generating '{name}'...")
        # Force specific layout by mocking selector or just modifying spec context if possible
        # Since CompositionalDesigner uses LayoutSelector internally, we need a way to force it.
        # For this gallery, we'll temporarily override the layout selector's choice
        # OR better: We can manually use LayoutEngine and render using designer's components
        # BUT easiest is to just let the designer do it if we can influence it.

        # Actually, CompositionalDesigner.create_design calls layout_selector.select_layout(spec)
        # We can monkeypatch it for this script
        designer.layout_selector.select_layout = lambda s: name

        img = designer.create_design(spec)
        # Resize for grid (e.g. 400x500)
        img.thumbnail((400, 500))
        images.append((name, img))

    # 5. Create Grid
    # 3 columns, ceil(len/3) rows
    cols = 3
    rows = (len(images) + cols - 1) // cols

    thumb_w, thumb_h = images[0][1].size
    padding = 20
    text_h = 30

    grid_w = cols * (thumb_w + padding) + padding
    grid_h = rows * (thumb_h + padding + text_h) + padding

    gallery = Image.new('RGB', (grid_w, grid_h), "#F0F0F0")
    draw = ImageDraw.Draw(gallery)

    try:
        font = ImageFont.truetype("Arial", 16)
    except:
        font = ImageFont.load_default()

    for i, (name, img) in enumerate(images):
        r = i // cols
        c = i % cols

        x = padding + c * (thumb_w + padding)
        y = padding + r * (thumb_h + padding + text_h)

        # Paste image
        gallery.paste(img, (x, y))

        # Draw label
        draw.text((x, y + thumb_h + 5), name, fill="#000000", font=font)

    output_path = "layout_gallery.png"
    gallery.save(output_path)
    print(f"\n✅ Gallery saved to: {output_path}")

if __name__ == "__main__":
    generate_gallery()
