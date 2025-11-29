"""
Palette Manager

Manages color palette generation and consistency across design variants.
Ensures that multiple variants of a design share a cohesive color family
while having distinct visual identities.
"""

import random
import colorsys
from typing import List, Dict, Tuple

class PaletteManager:
    """Manages color palette generation and consistency."""

    @staticmethod
    def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to HSL."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return colorsys.rgb_to_hls(r, g, b)

    @staticmethod
    def hsl_to_hex(h: float, s: float, l: float) -> str:
        """Convert HSL to hex color."""
        # Normalize hue to 0-1
        h = h % 1.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def generate_base_palette(self, tone: float, goal: int) -> Dict[str, List[str]]:
        """
        Generate a harmonious base color palette based on tone and goal.

        Args:
            tone: Design tone (0.0 = calm, 1.0 = energetic)
            goal: Design goal (0-3)

        Returns:
            Dictionary with 'base', 'gradients', and 'accents' keys.
        """
        if tone < 0.3:  # Minimalist
            return {
                "base": "#FFFFFF",
                "gradients": ["#F5F5F5", "#EBEBEB", "#E0E0E0"],
                "accents": ["#CCCCCC"]
            }

        elif tone > 0.7:  # Memphis (Bold)
            return {
                "base": "#FFFFFF",
                "gradients": ["#FFD700", "#FF69B4", "#FF1493"],
                "accents": ["#00CED1", "#000000", "#FFD700"]
            }

        elif goal == 3:  # Inspire (Cyber)
            return {
                "base": "#0A0A0A",
                "gradients": ["#1A0A2E", "#2E1A47", "#0A0A0A"],
                "accents": ["#9D4EDD", "#00F5FF", "#7209B7"]
            }

        else:  # Boho/Balanced
            # Pick a random warm hue for the base
            base_hue = random.choice([0.05, 0.08, 0.12])  # Warm hues
            return {
                "base": self.hsl_to_hex(base_hue, 0.2, 0.95),
                "gradients": [
                    self.hsl_to_hex(base_hue, 0.3, 0.85),
                    self.hsl_to_hex(base_hue + 0.05, 0.4, 0.70),
                    self.hsl_to_hex(base_hue + 0.08, 0.5, 0.60)
                ],
                "accents": [
                    self.hsl_to_hex(base_hue + 0.15, 0.4, 0.65),
                    self.hsl_to_hex(base_hue - 0.1, 0.3, 0.70)
                ]
            }

    def generate_variant_palette(self, base_palette: Dict[str, List[str]],
                               variant_index: int) -> Dict[str, List[str]]:
        """
        Generate a variation of the base palette by shifting hues.

        Args:
            base_palette: The original palette dictionary
            variant_index: Index of the variant (0 = original, 1+ = shifted)

        Returns:
            A new palette dictionary with shifted hues
        """
        if variant_index == 0:
            return base_palette

        # Determine shift amount (e.g., 15 degrees per variant)
        # 1.0 = 360 degrees, so 15 degrees ~= 0.04
        shift = (variant_index * 0.04) % 1.0

        new_palette = {
            "base": self._shift_color(base_palette["base"], shift),
            "gradients": [self._shift_color(c, shift) for c in base_palette["gradients"]],
            "accents": [self._shift_color(c, shift) for c in base_palette["accents"]]
        }

        return new_palette

    def _shift_color(self, hex_color: str, shift_amount: float) -> str:
        """Shift the hue of a single color."""
        h, l, s = self.hex_to_hsl(hex_color)

        # Don't shift grays/whites/blacks significantly
        if s < 0.1:
            return hex_color

        new_h = (h + shift_amount) % 1.0
        return self.hsl_to_hex(new_h, s, l)
