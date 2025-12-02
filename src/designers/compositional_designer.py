"""
Compositional Designer

The "Art Director" that orchestrates the layout-driven design process.
"""

from typing import List, Optional
from PIL import Image
from src.layouts.layout_engine import LayoutEngine, ZoneType
from src.generators.asset_generator import AssetGenerator
from src.generators.element_generator import ElementGenerator
from src.generators.constraint_renderer import ConstraintTextRenderer
from src.agents.design_spec import DesignSpec

from src.designers.layout_selector import LayoutSelector

from src.generators.palette_manager import PaletteManager

class CompositionalDesigner:
    def __init__(self, device: str = "mps"):
        self.layout_engine = LayoutEngine()
        self.layout_selector = LayoutSelector()
        self.asset_generator = AssetGenerator(device=device)
        self.element_generator = ElementGenerator()
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
        from src.utils.typography_engine import ModularScale, TextAnalyzer
        from src.layouts.layout_adapter import LayoutAdapter
        from src.layouts.layout_engine import DynamicLayoutEngine

        # 1. Select Base Layout (Smart Selection)
        base_layout_name = self.layout_selector.select_layout(spec)
        print(f"📐 Selected Base Layout: {base_layout_name}")

        # 2. Generate Dynamic Layout based on content elements
        layout = DynamicLayoutEngine.generate_layout(spec.content.elements, base_layout_name)
        print(f"🧩 Generated Dynamic Layout: {len(layout.zones)} zones")

        # 3. Create Canvas
        width, height = (1080, 1350) # Default portrait
        if spec.format == 1: width, height = (1080, 1080) # Square
        elif spec.format == 3: width, height = (1920, 600) # Banner

        # 4. Setup Typography Engine
        # Choose ratio based on tone
        if spec.tone > 0.7:
            ratio = ModularScale.GOLDEN_RATIO  # Energetic -> High contrast
        elif spec.tone < 0.3:
            ratio = ModularScale.MAJOR_SECOND  # Calm -> Low contrast
        else:
            ratio = ModularScale.MAJOR_THIRD   # Balanced

        type_scale = ModularScale(base_size=16, ratio=ratio)
        print(f"🔤 Typography: Ratio {ratio:.3f}")

        # Fix widows in text content
        for element in spec.content.elements:
            element.text = TextAnalyzer.fix_widows(element.text)

        # 5. Adapt Layout based on content (Fine-tuning)
        # Note: DynamicLayoutEngine already did a rough pass, but LayoutAdapter does precise text height
        adapted_layout = LayoutAdapter.adapt_zones(layout, spec, width, height)
        print(f"📏 Layout Adapted: {len(adapted_layout.zones)} zones")

        # 6. Create Canvas
        canvas = Image.new('RGB', (width, height), "#FFFFFF")

        # 7. Generate & Place Assets (Background first)
        # Collect text zones for masking
        text_zones = [(z.x, z.y, z.w, z.h) for z in adapted_layout.zones if z.type == ZoneType.TEXT]

        # Generate variant-specific palette
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
            print(f"🎨 Palette: Base {variant_palette['base']}, Accents {variant_palette['accents']}")

        # Generate procedural background
        try:
            bg = self.asset_generator.generate_background(width, height, tone=spec.tone,
                                                          goal=spec.goal, text_zones=text_zones,
                                                          palette=variant_palette)
            if bg:
                canvas = bg
            else:
                print("⚠️ Background generator returned None. Using white background.")
                canvas = Image.new('RGB', (width, height), "#FFFFFF")
        except Exception as e:
            print(f"❌ Error generating background: {e}. Using white background.")
            canvas = Image.new('RGB', (width, height), "#FFFFFF")

        # 8. Process Zones (using adapted layout)
        for zone in sorted(adapted_layout.zones, key=lambda z: z.z_index):
            x1, y1, x2, y2 = zone.to_pixel_rect(width, height)
            zone_w = x2 - x1
            zone_h = y2 - y1

            # SKIP IMAGE ZONES FOR NOW (SD is too slow)
            if zone.type == ZoneType.IMAGE:
                print(f"⏭️  Skipping hero image for zone: {zone.name} (SD disabled)")
                continue

            elif zone.type == ZoneType.TEXT:
                # Parse zone name to find corresponding element index
                # Name format: text_{index}_{role}
                try:
                    parts = zone.name.split('_')
                    if len(parts) >= 3 and parts[0] == "text":
                        idx = int(parts[1])
                        if 0 <= idx < len(spec.content.elements):
                            element = spec.content.elements[idx]

                            # Determine font size step based on role
                            font_size_step = 0
                            if element.role == "primary": font_size_step = 3
                            elif element.role == "secondary": font_size_step = 1

                            print(f"✍️  Rendering Text: {element.text[:20]}... ({element.role})")

                            self.text_renderer.render_text(
                                canvas,
                                element.text,
                                (x1, y1, x2, y2),
                                align="center" if "central" in base_layout_name else "left",
                                tone=spec.tone,
                                goal=spec.goal,
                                element="headline" if element.role == "primary" else "body",
                                optical_adjustment=True # Enable optical alignment
                            )
                except (ValueError, IndexError):
                    print(f"⚠️ Could not map zone {zone.name} to content element")

            elif zone.type == ZoneType.SHAPE:
                print(f"🔷 Rendering Shape: {zone.name}")
                # Determine shape properties
                shape_type = "circle" if "circle" in zone.name else "rect"
                if "pill" in zone.name: shape_type = "pill"
                if "grid" in zone.name: shape_type = "grid"

                # Use accent color
                color = variant_palette['accents'][0] if variant_palette else "#000000"

                shape_img = self.element_generator.generate_shape(
                    shape_type=shape_type,
                    width=zone_w,
                    height=zone_h,
                    color=color,
                    opacity=0.5 # Semi-transparent for accents
                )

                # Paste with alpha
                canvas.paste(shape_img, (x1, y1), shape_img)

            elif zone.type == ZoneType.ICON:
                print(f"🔶 Rendering Icon: {zone.name}")
                # Use accent color
                color = variant_palette['accents'][1] if variant_palette and len(variant_palette['accents']) > 1 else "#000000"

                # Determine icon type from name or random
                icon_name = "star" if "star" in zone.name else "arrow"

                icon_img = self.element_generator.generate_icon(
                    icon_name=icon_name,
                    size=min(zone_w, zone_h),
                    color=color
                )

                # Center icon in zone
                ix = x1 + (zone_w - icon_img.width) // 2
                iy = y1 + (zone_h - icon_img.height) // 2
                canvas.paste(icon_img, (ix, iy), icon_img)

            elif zone.type == ZoneType.LOGO and spec.logo_path:
                # Add logo logic here
                pass

        return canvas
