"""
Palette Manager

Manages color palette generation and consistency across design variants.
Ensures that multiple variants of a design share a cohesive color family
while having distinct visual identities.
Uses OkLCH color space for perceptual uniformity and accessibility.
"""

import random
import json
import os
from typing import List, Dict, Tuple
from src.utils.color_science import (
    hex_to_oklch, oklch_to_hex, calculate_contrast_ratio
)

class PaletteManager:
    """Manages color palette generation and consistency using OkLCH."""

    def __init__(self):
        self.emotional_map = self._load_emotional_mapping()

    def _load_emotional_mapping(self) -> Dict:
        """Load emotional mapping configuration."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "../config/emotional_mapping.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Emotional mapping not found, using fallback.")
            return {}

    def generate_base_palette(self, tone: float, goal: int) -> Dict[str, List[str]]:
        """
        Generate a harmonious base color palette based on tone and goal.

        Args:
            tone: Design tone (0.0 = calm, 1.0 = energetic)
            goal: Design goal (0-3)

        Returns:
            Dictionary with 'base', 'gradients', and 'accents' keys.
        """
        # 1. Determine Emotion/Style
        emotion = "trust" # Default
        if tone < 0.3: emotion = "calm"
        elif tone > 0.7: emotion = "energetic"
        elif goal == 3: emotion = "luxury" # Inspire
        elif goal == 2: emotion = "playful" # Entertain
        elif goal == 0: emotion = "tech" # Inform

        # 2. Sample Base Color from OkLCH Range
        base_hex = self._sample_color_from_emotion(emotion)

        # 3. Generate Harmonies (Monochromatic / Analogous / Complementary)
        # We'll use simple hue shifts in OkLCH space
        l, c, h = hex_to_oklch(base_hex)

        # Gradients: Lighter and Darker versions
        grad1 = oklch_to_hex(min(l + 0.15, 0.95), c, h)
        grad2 = oklch_to_hex(max(l - 0.15, 0.1), c, h)

        # Accents: Complementary (180 deg) or Split Comp (150/210)
        accent_h = (h + 180) % 360
        accent1 = oklch_to_hex(l, c, accent_h)

        # Secondary Accent: Analogous to primary accent
        accent2 = oklch_to_hex(l, c, (accent_h + 30) % 360)

        palette = {
            "base": base_hex,
            "gradients": [base_hex, grad1, grad2],
            "accents": [accent1, accent2]
        }

        # 4. Ensure Contrast
        return self.ensure_contrast(palette)

    def _sample_color_from_emotion(self, emotion: str) -> str:
        """Sample a random color from the emotion's OkLCH range."""
        ranges = self.emotional_map.get(emotion, self.emotional_map.get("trust"))

        l = random.uniform(*ranges["l"])
        c = random.uniform(*ranges["c"])
        h = random.uniform(*ranges["h"])

        return oklch_to_hex(l, c, h)

    def generate_variant_palette(self, base_palette: Dict[str, List[str]],
                               variant_index: int) -> Dict[str, List[str]]:
        """
        Generate a variation of the base palette by shifting hues.
        """
        if variant_index == 0:
            return base_palette

        # Shift Hue by 15 degrees per variant
        shift_deg = variant_index * 15.0

        new_palette = {
            "base": self._shift_hue(base_palette["base"], shift_deg),
            "gradients": [self._shift_hue(c, shift_deg) for c in base_palette["gradients"]],
            "accents": [self._shift_hue(c, shift_deg) for c in base_palette["accents"]]
        }

        return self.ensure_contrast(new_palette)

    def _shift_hue(self, hex_color: str, degrees: float) -> str:
        """Shift the hue of a single color in OkLCH space."""
        l, c, h = hex_to_oklch(hex_color)

        # Don't shift neutrals (low chroma)
        if c < 0.02:
            return hex_color

        new_h = (h + degrees) % 360
        return oklch_to_hex(l, c, new_h)

    def ensure_contrast(self, palette: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Ensure sufficient contrast between base (background) and accents (text/elements).
        Target: WCAG AA (4.5:1)
        """
        bg = palette["base"]
        bg_l, _, _ = hex_to_oklch(bg)

        # Check accents
        new_accents = []
        for accent in palette["accents"]:
            ratio = calculate_contrast_ratio(bg, accent)

            if ratio < 4.5:
                # Iteratively adjust lightness until contrast is met
                acc_l, acc_c, acc_h = hex_to_oklch(accent)

                # Decide direction: if BG is dark (<0.5), lighten accent. Else darken.
                direction = 1 if bg_l < 0.5 else -1

                # Safety break
                for _ in range(10):
                    acc_l += (0.1 * direction)
                    acc_l = max(0.01, min(0.99, acc_l)) # Clamp

                    new_hex = oklch_to_hex(acc_l, acc_c, acc_h)
                    new_ratio = calculate_contrast_ratio(bg, new_hex)

                    if new_ratio >= 4.5:
                        new_accents.append(new_hex)
                        break
                else:
                    # If failed to meet contrast after iterations, try explicit Black or White
                    black = "#000000"
                    white = "#ffffff"
                    ratio_b = calculate_contrast_ratio(bg, black)
                    ratio_w = calculate_contrast_ratio(bg, white)

                    if ratio_b >= 4.5:
                        new_accents.append(black)
                    elif ratio_w >= 4.5:
                        new_accents.append(white)
                    else:
                        # Both fail (rare, but possible with mid-tone saturated colors)
                        # Pick the best of the two
                        best = black if ratio_b > ratio_w else white
                        print(f"⚠️ Could not meet 4.5:1 contrast. Best fallback: {best} (Ratio: {max(ratio_b, ratio_w):.2f})")
                        new_accents.append(best)
            else:
                new_accents.append(accent)

        palette["accents"] = new_accents
        return palette
