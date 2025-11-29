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

from src.generators.palette_manager import PaletteManager

class CompositionalDesigner:
    def __init__(self, device: str = "mps"):
        self.layout_engine = LayoutEngine()
        self.layout_selector = LayoutSelector()
        self.asset_generator = AssetGenerator(device=device)
        self.text_renderer = ConstraintTextRenderer()
        self.palette_manager = PaletteManager()

        # State for palette consistency
        self.current_base_palette = None
        self.current_spec_id = None
        self.variant_counter = 0

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
        # Collect text zones for masking
        text_zones = [(z.x, z.y, z.w, z.h) for z in layout.zones if z.type == ZoneType.TEXT]

        # Generate variant-specific palette
        # We assume self.palette_manager exists (added in __init__)
        # If not, we'll just let the generator pick one (fallback)
        variant_palette = None
        if hasattr(self, 'palette_manager'):
            # If this is the first variant, generate base palette
            if not hasattr(self, 'current_base_palette') or self.current_spec_id != id(spec):
                self.current_base_palette = self.palette_manager.generate_base_palette(spec.tone, spec.goal)
                self.current_spec_id = id(spec)
                self.variant_counter = 0

            # Generate shifted palette for this variant
            variant_palette = self.palette_manager.generate_variant_palette(
                self.current_base_palette, self.variant_counter
            )
            self.variant_counter += 1

        # Generate procedural background
        bg = self.asset_generator.generate_background(width, height, tone=spec.tone,
                                                      goal=spec.goal, text_zones=text_zones,
                                                      palette=variant_palette)
        canvas = bg

        # 4. Process Zones
        for zone in sorted(layout.zones, key=lambda z: z.z_index):
            x1, y1, x2, y2 = zone.to_pixel_rect(width, height)
            zone_w = x2 - x1
            zone_h = y2 - y1

            # SKIP IMAGE ZONES FOR NOW (SD is too slow)
            # We'll add procedural hero images later
            if zone.type == ZoneType.IMAGE:
                print(f"⏭️  Skipping hero image for zone: {zone.name} (SD disabled)")
                continue

            elif zone.type == ZoneType.TEXT:
                text_content = ""
                element_type = "body"  # Default

                if zone.name == "headline":
                    text_content = spec.content.headline
                    element_type = "headline"
                elif zone.name == "subheading":
                    text_content = spec.content.subheading
                    element_type = "subheading"
                elif zone.name == "details" or zone.name == "body_text":
                    text_content = spec.content.details
                    element_type = "body"

                if text_content:
                    print(f"✍️ Rendering Text: {zone.name}")
                    self.text_renderer.render_text(
                        canvas,
                        text_content,
                        (x1, y1, x2, y2),
                        align="center" if "central" in layout.name else "left",
                        tone=spec.tone,
                        goal=spec.goal,
                        element=element_type
                    )

            elif zone.type == ZoneType.LOGO and spec.logo_path:
                # Add logo logic here
                pass

        return canvas
