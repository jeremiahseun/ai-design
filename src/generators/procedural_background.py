"""
Refined Procedural Background Generator - Professional Quality

Generates Pinterest-style aesthetic backgrounds using gradient-first composition
with strategic element placement and sophisticated visual effects.
"""

import random
import math
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import colorsys




class GradientEngine:
    """Advanced gradient rendering engine."""

    @staticmethod
    def create_linear_gradient(width: int, height: int, colors: List[str],
                              angle: int = 45) -> Image.Image:
        """Create a multi-stop linear gradient."""
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)

        # Convert angle to radians
        rad = math.radians(angle)

        # Calculate gradient direction
        if angle == 0:  # Horizontal
            for x in range(width):
                ratio = x / width
                color = GradientEngine._interpolate_colors(colors, ratio)
                draw.line([(x, 0), (x, height)], fill=color)
        elif angle == 90:  # Vertical
            for y in range(height):
                ratio = y / height
                color = GradientEngine._interpolate_colors(colors, ratio)
                draw.line([(0, y), (width, y)], fill=color)
        else:  # Diagonal
            # Create oversized gradient and rotate
            diagonal = int(math.sqrt(width**2 + height**2))
            temp = Image.new('RGB', (diagonal, diagonal))
            temp_draw = ImageDraw.Draw(temp)

            for y in range(diagonal):
                ratio = y / diagonal
                color = GradientEngine._interpolate_colors(colors, ratio)
                temp_draw.line([(0, y), (diagonal, y)], fill=color)

            # Rotate and crop
            temp = temp.rotate(angle, expand=False, fillcolor=colors[0])
            # Center crop
            left = (diagonal - width) // 2
            top = (diagonal - height) // 2
            img = temp.crop((left, top, left + width, top + height))

        return img

    @staticmethod
    def create_radial_gradient(width: int, height: int, center: Tuple[float, float],
                              color: str, max_radius_ratio: float = 0.5) -> Image.Image:
        """Create a radial gradient with transparency (glow effect)."""
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        cx = int(center[0] * width)
        cy = int(center[1] * height)
        max_radius = int(min(width, height) * max_radius_ratio)

        # Parse hex color
        hex_color = color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Create radial gradient
        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < max_radius:
                    ratio = distance / max_radius
                    alpha = int(255 * (1 - ratio))
                    img.putpixel((x, y), (r, g, b, alpha))

        return img

    @staticmethod
    def _interpolate_colors(colors: List[str], ratio: float) -> str:
        """Interpolate between multiple colors."""
        n = len(colors) - 1
        if n == 0:
            return colors[0]

        segment = ratio * n
        idx = int(segment)
        if idx >= n:
            return colors[-1]

        local_ratio = segment - idx
        return GradientEngine._blend_colors(colors[idx], colors[idx + 1], local_ratio)

    @staticmethod
    def _blend_colors(color1: str, color2: str, ratio: float) -> str:
        """Blend two hex colors."""
        c1 = color1.lstrip('#')
        c2 = color2.lstrip('#')

        r1, g1, b1 = tuple(int(c1[i:i+2], 16) for i in (0, 2, 4))
        r2, g2, b2 = tuple(int(c2[i:i+2], 16) for i in (0, 2, 4))

        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)

        return f"#{r:02x}{g:02x}{b:02x}"


from .palette_manager import PaletteManager

class RefinedBackgroundGenerator:
    """Professional-quality background generator using gradient-first composition."""

    ANCHOR_POSITIONS = {
        "top_left": (0.15, 0.15),
        "top_right": (0.85, 0.15),
        "bottom_left": (0.15, 0.85),
        "bottom_right": (0.85, 0.85),
        "center": (0.5, 0.5),
        "top_center": (0.5, 0.2),
        "bottom_center": (0.5, 0.8)
    }

    def __init__(self):
        self.palette_manager = PaletteManager()
        self.gradient_engine = GradientEngine()

    def generate(self, width: int, height: int, tone: float, goal: int,
                 text_zones: List[Tuple[float, float, float, float]] = None,
                 palette: Optional[dict] = None) -> Image.Image:
        """
        Generate a professional-quality background.

        Args:
            width: Image width
            height: Image height
            tone: Design tone (0.0 = calm, 1.0 = energetic)
            goal: Design goal (0-3)
            text_zones: Text zones to avoid
            palette: Optional custom palette dictionary

        Returns:
            PIL Image
        """
        # Get color palette (use provided or generate new)
        if palette is None:
            palette = self.palette_manager.generate_base_palette(tone, goal)

        # Determine style
        if tone < 0.3:
            return self._generate_minimalist(width, height, palette)
        elif tone > 0.7:
            return self._generate_memphis(width, height, palette, text_zones)
        elif goal == 3:
            return self._generate_cyber(width, height, palette)
        else:
            return self._generate_boho(width, height, palette, text_zones)

    def _generate_minimalist(self, width: int, height: int, palette: dict) -> Image.Image:
        """Generate minimalist background."""
        # Foundation: Vertical gradient
        canvas = self.gradient_engine.create_linear_gradient(
            width, height, palette["gradients"], angle=90
        )

        # Single radial glow (center, very subtle)
        glow = self.gradient_engine.create_radial_gradient(
            width, height, self.ANCHOR_POSITIONS["center"],
            palette["accents"][0], max_radius_ratio=0.4
        )
        # Very low opacity
        glow.putalpha(int(255 * 0.10))
        canvas.paste(glow, (0, 0), glow)

        # Minimal texture
        canvas = self._add_texture(canvas, intensity=2)

        return canvas

    def _generate_boho(self, width: int, height: int, palette: dict,
                      text_zones: List) -> Image.Image:
        """Generate boho/organic background."""
        # Foundation: Diagonal gradient
        canvas = self.gradient_engine.create_linear_gradient(
            width, height, palette["gradients"], angle=45
        )

        # Large blurred blob (bottom-left)
        blob1 = self._create_soft_blob(width, height,
                                       self.ANCHOR_POSITIONS["bottom_left"],
                                       palette["accents"][0], size_ratio=0.5, opacity=0.15)
        canvas.paste(blob1, (0, 0), blob1)

        # Medium blurred blob (top-right)
        if len(palette["accents"]) > 1:
            blob2 = self._create_soft_blob(width, height,
                                          self.ANCHOR_POSITIONS["top_right"],
                                          palette["accents"][1], size_ratio=0.35, opacity=0.20)
            canvas.paste(blob2, (0, 0), blob2)

        # Paper grain texture
        canvas = self._add_texture(canvas, intensity=3)

        return canvas

    def _generate_memphis(self, width: int, height: int, palette: dict,
                         text_zones: List) -> Image.Image:
        """Generate Memphis-style background."""
        # Foundation: Diagonal gradient
        canvas = self.gradient_engine.create_linear_gradient(
            width, height, palette["gradients"], angle=135
        )

        # Large geometric element (triangle, top-right, low opacity)
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Triangle at top-right
        tri_color = palette["accents"][0] if palette["accents"] else "#00CED1"
        tri_size = int(min(width, height) * 0.4)
        points = [
            (width, 0),
            (width - tri_size, 0),
            (width, tri_size)
        ]
        draw.polygon(points, fill=tri_color + "4D")  # 30% opacity

        # Circle at bottom-left
        if len(palette["accents"]) > 1:
            circle_color = palette["accents"][1]
            radius = int(min(width, height) * 0.15)
            cx, cy = int(width * 0.2), int(height * 0.8)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                        fill=circle_color + "80")  # 50% opacity

        canvas.paste(overlay, (0, 0), overlay)

        return canvas

    def _generate_cyber(self, width: int, height: int, palette: dict) -> Image.Image:
        """Generate cyber/gradient style background."""
        # Foundation: Radial dark gradient
        canvas = self.gradient_engine.create_linear_gradient(
            width, height, palette["gradients"], angle=90
        )

        # Central glow (cyan/purple)
        glow1 = self.gradient_engine.create_radial_gradient(
            width, height, self.ANCHOR_POSITIONS["center"],
            palette["accents"][0], max_radius_ratio=0.6
        )
        glow1.putalpha(int(255 * 0.4))
        canvas.paste(glow1, (0, 0), glow1)

        # Top-left accent glow
        if len(palette["accents"]) > 1:
            glow2 = self.gradient_engine.create_radial_gradient(
                width, height, self.ANCHOR_POSITIONS["top_left"],
                palette["accents"][1], max_radius_ratio=0.35
            )
            glow2.putalpha(int(255 * 0.3))
            canvas.paste(glow2, (0, 0), glow2)

        # Light noise
        canvas = self._add_texture(canvas, intensity=1)

        return canvas

    def _create_soft_blob(self, width: int, height: int, position: Tuple[float, float],
                         color: str, size_ratio: float = 0.4, opacity: float = 0.2) -> Image.Image:
        """Create a soft, blurred blob."""
        blob_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(blob_img)

        cx = int(position[0] * width)
        cy = int(position[1] * height)
        radius = int(min(width, height) * size_ratio)

        # Parse color
        hex_color = color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        alpha = int(255 * opacity)

        # Draw circle
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=(r, g, b, alpha))

        # Apply blur
        blob_img = blob_img.filter(ImageFilter.GaussianBlur(radius=80))

        return blob_img

    def _add_texture(self, image: Image.Image, intensity: int = 3) -> Image.Image:
        """Add subtle noise texture."""
        width, height = image.size
        noise = np.random.randint(0, intensity, (height, width, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise, mode='RGB')
        return Image.blend(image, noise_img, alpha=0.03)
