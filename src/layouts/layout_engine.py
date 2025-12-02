"""
Layout Engine

Defines the mathematical structure of a design using Zones.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

class ZoneType(Enum):
    TEXT = "text"
    IMAGE = "image"
    LOGO = "logo"
    BACKGROUND = "background"
    SHAPE = "shape"
    ICON = "icon"

@dataclass
class Zone:
    """
    A defined area in the design.
    Coordinates are normalized (0.0 to 1.0).
    """
    x: float
    y: float
    w: float
    h: float
    type: ZoneType
    name: str  # e.g., "headline", "hero_image"
    z_index: int = 0

    def to_pixel_rect(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel rectangle (left, top, right, bottom)"""
        return (
            int(self.x * width),
            int(self.y * height),
            int((self.x + self.w) * width),
            int((self.y + self.h) * height)
        )

@dataclass
class Layout:
    """
    A complete layout blueprint consisting of multiple zones.
    """
    name: str
    zones: List[Zone]
    description: str

class LayoutEngine:
    """
    Manages and provides layout templates.
    """

    @staticmethod
    def get_layout(name: str) -> Layout:
        """Get a layout by name"""
        layouts = {
            "split_horizontal": LayoutEngine._split_horizontal(),
            "split_vertical": LayoutEngine._split_vertical(),
            "central_hero": LayoutEngine._central_hero(),
            "typographic_bold": LayoutEngine._typographic_bold(),
            "modern_clean": LayoutEngine._modern_clean(),
            "magazine_grid": LayoutEngine._magazine_grid(),
            "modern_geometric": LayoutEngine._modern_geometric(),
            "diagonal_split": LayoutEngine._diagonal_split(),
            "asymmetric_editorial": LayoutEngine._asymmetric_editorial()
        }
        return layouts.get(name, LayoutEngine._modern_clean())

    @staticmethod
    def _split_horizontal() -> Layout:
        """Image top 50%, Text bottom 50%"""
        return Layout(
            name="split_horizontal",
            description="Classic split: Image on top, text below.",
            zones=[
                Zone(0.0, 0.0, 1.0, 0.5, ZoneType.IMAGE, "hero_image", 1),
                Zone(0.05, 0.55, 0.9, 0.15, ZoneType.TEXT, "headline", 2),
                Zone(0.05, 0.72, 0.9, 0.08, ZoneType.TEXT, "subheading", 2),
                Zone(0.05, 0.85, 0.9, 0.1, ZoneType.TEXT, "details", 2),
                Zone(0.05, 0.05, 0.2, 0.1, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _split_vertical() -> Layout:
        """Image left 50%, Text right 50%"""
        return Layout(
            name="split_vertical",
            description="Modern split: Image left, text right.",
            zones=[
                Zone(0.0, 0.0, 0.5, 1.0, ZoneType.IMAGE, "hero_image", 1),
                Zone(0.55, 0.2, 0.4, 0.2, ZoneType.TEXT, "headline", 2),
                Zone(0.55, 0.45, 0.4, 0.1, ZoneType.TEXT, "subheading", 2),
                Zone(0.55, 0.6, 0.4, 0.2, ZoneType.TEXT, "details", 2),
                Zone(0.85, 0.05, 0.1, 0.1, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _central_hero() -> Layout:
        """Hero image in center, text overlay"""
        return Layout(
            name="central_hero",
            description="Focus on central subject with overlay text.",
            zones=[
                Zone(0.1, 0.1, 0.8, 0.6, ZoneType.IMAGE, "hero_image", 1),
                Zone(0.1, 0.75, 0.8, 0.1, ZoneType.TEXT, "headline", 2),
                Zone(0.1, 0.87, 0.8, 0.08, ZoneType.TEXT, "details", 2),
                Zone(0.45, 0.05, 0.1, 0.1, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _typographic_bold() -> Layout:
        """Text dominant, small accent image"""
        return Layout(
            name="typographic_bold",
            description="Bold typography takes center stage.",
            zones=[
                Zone(0.05, 0.15, 0.9, 0.25, ZoneType.TEXT, "headline", 2),
                Zone(0.05, 0.45, 0.9, 0.1, ZoneType.TEXT, "subheading", 2),
                Zone(0.05, 0.6, 0.9, 0.3, ZoneType.IMAGE, "hero_image", 1),
                Zone(0.05, 0.92, 0.9, 0.05, ZoneType.TEXT, "details", 2),
                Zone(0.05, 0.05, 0.15, 0.08, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _modern_clean() -> Layout:
        """Balanced layout with whitespace"""
        return Layout(
            name="modern_clean",
            description="Clean, balanced layout with ample whitespace.",
            zones=[
                Zone(0.1, 0.1, 0.8, 0.4, ZoneType.IMAGE, "hero_image", 1),
                Zone(0.1, 0.55, 0.8, 0.15, ZoneType.TEXT, "headline", 2),
                Zone(0.1, 0.72, 0.8, 0.08, ZoneType.TEXT, "subheading", 2),
                Zone(0.1, 0.85, 0.8, 0.05, ZoneType.TEXT, "details", 2),
                Zone(0.05, 0.05, 0.1, 0.1, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _magazine_grid() -> Layout:
        """High-fashion magazine style with overlapping text"""
        return Layout(
            name="magazine_grid",
            description="Editorial grid with overlapping headline.",
            zones=[
                Zone(0.0, 0.0, 1.0, 0.65, ZoneType.IMAGE, "hero_image", 1),
                Zone(0.05, 0.55, 0.9, 0.15, ZoneType.TEXT, "headline", 2), # Overlaps image
                Zone(0.05, 0.72, 0.4, 0.05, ZoneType.TEXT, "subheading", 2),
                Zone(0.5, 0.72, 0.45, 0.2, ZoneType.TEXT, "details", 2),
                Zone(0.8, 0.05, 0.15, 0.05, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _modern_geometric() -> Layout:
        """Modern layout with geometric accents"""
        return Layout(
            name="modern_geometric",
            description="Clean layout with decorative geometric shapes.",
            zones=[
                Zone(0.0, 0.0, 1.0, 1.0, ZoneType.BACKGROUND, "bg", 0),
                Zone(0.6, 0.1, 0.3, 0.3, ZoneType.SHAPE, "circle_accent", 1), # Decorative circle
                Zone(0.1, 0.1, 0.8, 0.4, ZoneType.IMAGE, "hero_image", 2),
                Zone(0.1, 0.55, 0.8, 0.15, ZoneType.TEXT, "headline", 3),
                Zone(0.1, 0.72, 0.8, 0.08, ZoneType.TEXT, "subheading", 3),
                Zone(0.1, 0.85, 0.8, 0.05, ZoneType.TEXT, "details", 3),
                Zone(0.05, 0.05, 0.1, 0.1, ZoneType.LOGO, "logo", 4)
            ]
        )

    @staticmethod
    def _diagonal_split() -> Layout:
        """Dynamic diagonal flow"""
        return Layout(
            name="diagonal_split",
            description="Dynamic layout with diagonal visual flow.",
            zones=[
                Zone(0.05, 0.05, 0.6, 0.5, ZoneType.IMAGE, "hero_image", 1), # Top-left
                Zone(0.4, 0.5, 0.55, 0.15, ZoneType.TEXT, "headline", 2), # Center-right
                Zone(0.4, 0.68, 0.55, 0.08, ZoneType.TEXT, "subheading", 2),
                Zone(0.4, 0.8, 0.55, 0.15, ZoneType.TEXT, "details", 2),
                Zone(0.05, 0.85, 0.1, 0.1, ZoneType.LOGO, "logo", 3)
            ]
        )

    @staticmethod
    def _asymmetric_editorial() -> Layout:
        """Artistic asymmetric layout"""
        return Layout(
            name="asymmetric_editorial",
            description="Sophisticated off-center composition.",
            zones=[
                Zone(0.4, 0.0, 0.6, 1.0, ZoneType.IMAGE, "hero_image", 1), # Right vertical strip
                Zone(0.05, 0.15, 0.3, 0.3, ZoneType.TEXT, "headline", 2), # Left heavy
                Zone(0.05, 0.5, 0.3, 0.1, ZoneType.TEXT, "subheading", 2),
                Zone(0.05, 0.65, 0.3, 0.3, ZoneType.TEXT, "details", 2),
                Zone(0.05, 0.05, 0.1, 0.05, ZoneType.LOGO, "logo", 3)
            ]
        )
class DynamicLayoutEngine:
    """
    Generates layouts dynamically based on content elements.
    """
    @staticmethod
    def generate_layout(elements: List['DesignElement'], base_layout_name: str = "modern_clean") -> Layout:
        """
        Generate a layout that fits the specific elements provided.
        """
        # Start with a base template
        base_layout = LayoutEngine.get_layout(base_layout_name)

        # Separate elements by role
        primary = [e for e in elements if e.role == "primary"]
        secondary = [e for e in elements if e.role == "secondary"]
        tertiary = [e for e in elements if e.role == "tertiary"]

        # Create new zones list
        new_zones = []

        # 1. Keep non-text zones (images, logos)
        for zone in base_layout.zones:
            if zone.type != ZoneType.TEXT:
                new_zones.append(zone)

        # 2. Allocate Text Zones
        # We need to find where text usually goes in this layout
        text_area_y_start = 0.5  # Default
        text_area_height = 0.4

        # Find the bounding box of original text zones to know where to put new ones
        orig_text_zones = [z for z in base_layout.zones if z.type == ZoneType.TEXT]
        if orig_text_zones:
            min_y = min(z.y for z in orig_text_zones)
            max_y = max(z.y + z.h for z in orig_text_zones)
            min_x = min(z.x for z in orig_text_zones)
            max_x = max(z.x + z.w for z in orig_text_zones)

            text_area_y_start = min_y
            text_area_height = max_y - min_y
            text_area_x = min_x
            text_area_w = max_x - min_x
        else:
            text_area_x = 0.1
            text_area_w = 0.8

        # Distribute available height among elements
        total_elements = len(elements)
        if total_elements == 0:
            return Layout(base_layout.name + "_dynamic", new_zones, "Dynamic layout with no text")

        # Simple vertical stack strategy for now
        # (A real solver would be better, but this works for "list of headings")
        current_y = text_area_y_start

        # Calculate weights for height distribution
        # Primary gets 2x space, Secondary 1.5x, Tertiary 1x
        weights = []
        for e in elements:
            if e.role == "primary": weights.append(2.0)
            elif e.role == "secondary": weights.append(1.5)
            else: weights.append(1.0)

        total_weight = sum(weights)
        unit_height = text_area_height / total_weight

        for i, element in enumerate(elements):
            h = weights[i] * unit_height

            # Add padding
            h_actual = h * 0.9
            y_actual = current_y + (h * 0.05)

            new_zones.append(Zone(
                x=text_area_x,
                y=y_actual,
                w=text_area_w,
                h=h_actual,
                type=ZoneType.TEXT,
                name=f"text_{i}_{element.role}", # Unique name
                z_index=2
            ))
            current_y += h

        return Layout(
            name=base_layout.name + "_dynamic",
            zones=new_zones,
            description=f"Dynamic layout for {total_elements} elements based on {base_layout.name}"
        )
