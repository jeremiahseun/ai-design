"""
Typography Engine

Implements advanced typographic systems including Modular Scales,
Optical Sizing, and Text Analysis for readability.
"""

import math
from typing import List, Tuple, Optional

class ModularScale:
    """
    Generates a harmonious type scale based on a base size and ratio.
    """

    # Common Ratios
    MINOR_SECOND = 1.067
    MAJOR_SECOND = 1.125
    MINOR_THIRD = 1.200
    MAJOR_THIRD = 1.250
    PERFECT_FOURTH = 1.333
    AUGMENTED_FOURTH = 1.414
    PERFECT_FIFTH = 1.500
    GOLDEN_RATIO = 1.618

    def __init__(self, base_size: float = 16.0, ratio: float = 1.25):
        self.base_size = base_size
        self.ratio = ratio

    def get_size(self, step: int) -> float:
        """
        Get font size at a specific step in the scale.
        Step 0 = base_size.
        Positive steps = larger sizes.
        Negative steps = smaller sizes.
        """
        return self.base_size * (self.ratio ** step)

    def get_scale(self, steps: List[int]) -> List[float]:
        """Get a list of sizes for the given steps."""
        return [self.get_size(s) for s in steps]


class OpticalSizing:
    """
    Calculates optical adjustments for typography.
    """

    @staticmethod
    def calculate_tracking(font_size: float) -> float:
        """
        Calculate suggested letter-spacing (tracking) in ems.

        Rule of thumb:
        - Large text (Display) needs tighter tracking (negative).
        - Small text (Caption) needs looser tracking (positive).
        - Body text (approx 16px) is neutral.

        Formula approximation based on San Francisco font guidelines:
        tracking = a + b * e^(c * size) ... simplified linear/log model here.
        """
        # Simplified model:
        # 10px -> +0.03em
        # 16px -> 0.00em
        # 72px -> -0.03em

        if font_size < 16:
            # Increase tracking for small sizes
            return 0.03 * (1 - (font_size / 16))
        else:
            # Decrease tracking for large sizes (clamp at -0.05)
            # Logarithmic decay starting from 16px
            # math.log(1) is 0, so at 16px this returns 0.
            return max(-0.05, -0.01 * math.log(font_size / 16, 1.5))


class TextAnalyzer:
    """
    Analyzes and fixes text for better readability.
    """

    @staticmethod
    def fix_widows(text: str) -> str:
        """
        Prevents widows (single words on the last line) by replacing
        the last space with a non-breaking space.
        """
        if not text:
            return text

        words = text.split()
        if len(words) < 2:
            return text

        # Join the last two words with a non-breaking space (\u00A0)
        last_two = words[-2] + "\u00A0" + words[-1]

        # Reconstruct text
        return " ".join(words[:-2] + [last_two])

    @staticmethod
    def estimate_line_length(text: str, font_size: float, width_px: float) -> int:
        """
        Estimate characters per line (CPL) for a given width and font size.
        Assumes average character width is ~0.5em.
        """
        avg_char_width = font_size * 0.5
        cpl = width_px / avg_char_width
        return int(cpl)

    @staticmethod
    def is_optimal_line_length(cpl: int) -> bool:
        """
        Check if CPL is within the optimal range (45-75).
        """
        return 45 <= cpl <= 75
