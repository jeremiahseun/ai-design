"""
Compositional Designer

The "Art Director" that orchestrates the layout-driven design process.
"""

from typing import List, Optional
from PIL import Image
from src.layouts.layout_engine import LayoutEngine, ZoneType
from src.generators.asset_generator import AssetGenerator
from src.generators.constraint_renderer import ConstraintTextRenderer
from src.agents.design_spec import DesignSpec

from src.designers.layout_selector import LayoutSelector

class CompositionalDesigner:
    def __init__(self, device: str = "mps"):
        self.layout_engine = LayoutEngine()
        self.layout_selector = LayoutSelector()
        self.asset_generator = AssetGenerator(device=device)
        self.text_renderer = ConstraintTextRenderer()

    def create_design(self, spec: DesignSpec) -> Image.Image:
        """
        Create a design based on the spec using compositional logic.
        """
        # 1. Select Layout (Smart Selection)
        layout_name = self.layout_selector.select_layout(spec)
        layout = self.layout_engine.get_layout(layout_name)
        print(f"📐 Selected Layout: {layout.name}")

        # 2. Create Canvas
        width, height = (1080, 1350) # Default portrait
        if spec.format == 1: width, height = (1080, 1080) # Square
        elif spec.format == 3: width, height = (1920, 600) # Banner

        canvas = Image.new('RGB', (width, height), "#FFFFFF")

        # 3. Generate & Place Assets (Background first)
        # For now, generate a gradient background for the whole canvas
        bg = self.asset_generator.generate_background(width, height)
        canvas.paste(bg, (0, 0))

        # 4. Process Zones
        for zone in sorted(layout.zones, key=lambda z: z.z_index):
            x1, y1, x2, y2 = zone.to_pixel_rect(width, height)
            zone_w = x2 - x1
            zone_h = y2 - y1

            if zone.type == ZoneType.IMAGE:
                print(f"🎨 Generating Hero Image for zone: {zone.name}")
                # Generate hero
                hero = self.asset_generator.generate_hero_image(spec.style_prompt, zone_w, zone_h)
                # Resize to fit exactly (or crop)
                hero = hero.resize((zone_w, zone_h), Image.Resampling.LANCZOS)
                canvas.paste(hero, (x1, y1))

            elif zone.type == ZoneType.TEXT:
                text_content = ""
                if zone.name == "headline": text_content = spec.content.headline
                elif zone.name == "subheading": text_content = spec.content.subheading
                elif zone.name == "details": text_content = spec.content.details

                if text_content:
                    print(f"✍️ Rendering Text: {zone.name}")
                    self.text_renderer.render_text(
                        canvas,
                        text_content,
                        (x1, y1, x2, y2),
                        align="center" if "central" in layout.name else "left",
                        tone=spec.tone  # Pass tone for smart shadow decisions
                    )

            elif zone.type == ZoneType.LOGO and spec.logo_path:
                # Add logo logic here
                pass

        return canvas
