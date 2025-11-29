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
            "modern_clean": LayoutEngine._modern_clean()
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
