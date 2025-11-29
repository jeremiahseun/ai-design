"""
Constraint-Based Text Renderer

Renders text that strictly fits within a defined bounding box (Zone).
"""

from PIL import Image, ImageDraw, ImageFont, ImageStat
from typing import Tuple, Optional
import textwrap

class ConstraintTextRenderer:
    def __init__(self):
        self.default_font = "Arial" # Fallback

        # Initialize FontManager
        try:
            from src.utils.font_manager import FontManager
            self.font_manager = FontManager()
            self.use_google_fonts = True
            print("✅ FontManager initialized")
        except Exception as e:
            print(f"⚠️  FontManager not available: {e}")
            self.font_manager = None
            self.use_google_fonts = False

    def _sample_background_brightness(self, image: Image.Image, box: Tuple[int, int, int, int]) -> float:
        """
        Sample the background within the box and return brightness (0.0 = black, 1.0 = white).
        """
        x1, y1, x2, y2 = box
        # Crop to the text zone
        zone = image.crop((x1, y1, x2, y2))
        # Convert to grayscale
        gray = zone.convert('L')
        # Calculate average brightness
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0] / 255.0  # Normalize to 0-1
        return brightness

    def _calculate_contrast_ratio(self, color1_brightness: float, color2_brightness: float) -> float:
        """Calculate WCAG contrast ratio between two colors."""
        # Add 0.05 to avoid division by zero and for WCAG formula
        lighter = max(color1_brightness, color2_brightness)
        darker = min(color1_brightness, color2_brightness)
        return (lighter + 0.05) / (darker + 0.05)

    def render_text(self,
                   image: Image.Image,
                   text: str,
                   box: Tuple[int, int, int, int],
                   font_path: Optional[str] = None,
                   color: Optional[str] = None,
                   align: str = "center",
                   tone: float = 0.5,
                   goal: int = 0,
                   element: str = "headline") -> Image.Image:
        """
        Render text fitting strictly inside the box.
        Smart color selection based on background.

        Args:
            image: PIL Image to render on
            text: Text to render
            box: (x1, y1, x2, y2) bounding box
            font_path: Optional custom font path
            color: Optional text color
            align: Text alignment
            tone: Design tone (0-1) for font selection
            goal: Design goal (0-3) for font selection
            element: 'headline', 'subheading', or 'body' for font selection
        """
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        # 0. Smart Color Selection
        bg_brightness = self._sample_background_brightness(image, box)

        if color is None:
            # Auto-select text color for optimal contrast
            if bg_brightness > 0.5:  # Light background
                text_color = "#000000"  # Black text
                text_brightness = 0.0
            else:  # Dark background
                text_color = "#FFFFFF"  # White text
                text_brightness = 1.0
        else:
            text_color = color
            # Rough brightness estimate for provided color
            if color in ["#FFFFFF", "#FFF"]:
                text_brightness = 1.0
            elif color in ["#000000", "#000"]:
                text_brightness = 0.0
            else:
                text_brightness = 0.5  # Assume medium

        # Calculate if we need a shadow
        contrast_ratio = self._calculate_contrast_ratio(bg_brightness, text_brightness)
        needs_shadow = contrast_ratio < 3.0  # Low contrast

        # Only add shadow if needed OR if tone is bold/energetic
        use_shadow = needs_shadow or tone > 0.6

        # Get professional font
        if font_path:
            selected_font_path = font_path
        elif self.use_google_fonts and self.font_manager:
            selected_font_path = self.font_manager.get_font_for_tone(tone, goal, element)
        else:
            selected_font_path = "/System/Library/Fonts/Helvetica.ttc"

        # 1. Find optimal font size
        font_size = 100 # Start large
        min_font_size = 10

        final_font = None
        final_lines = []

        # Binary search-ish approach (iterative reduction)
        while font_size >= min_font_size:
            try:
                font = ImageFont.truetype(selected_font_path, font_size)
            except:
                font = ImageFont.load_default()
                break

            # Estimate chars per line
            avg_char_w = font.getlength("x")
            chars_per_line = int(box_w / (avg_char_w * 0.8)) # 0.8 factor for safety
            if chars_per_line < 1: chars_per_line = 1

            lines = textwrap.wrap(text, width=chars_per_line)

            # Check height
            # ascent, descent = font.getmetrics()
            # line_height = ascent + descent
            bbox = font.getbbox("Ay")
            line_height = bbox[3] - bbox[1] + 10 # padding

            total_h = line_height * len(lines)

            # Check width of longest line
            max_w = 0
            for line in lines:
                w = font.getlength(line)
                if w > max_w: max_w = w

            if total_h <= box_h and max_w <= box_w:
                final_font = font
                final_lines = lines
                break

            font_size -= 2 # Decrease size

        if final_font is None:
            # Fallback if text is just too massive
            final_font = ImageFont.load_default()
            final_lines = [text]

        # 2. Render lines
        bbox = final_font.getbbox("Ay")
        line_height = bbox[3] - bbox[1] + 10
        total_text_h = line_height * len(final_lines)

        # Vertical center
        start_y = y1 + (box_h - total_text_h) // 2

        for i, line in enumerate(final_lines):
            line_w = final_font.getlength(line)

            # Horizontal align
            if align == "center":
                start_x = x1 + (box_w - line_w) // 2
            elif align == "right":
                start_x = x2 - line_w
            else: # left
                start_x = x1

            curr_y = start_y + (i * line_height)

            # Draw shadow (conditional)
            if use_shadow:
                shadow_color = "#000000" if text_brightness > 0.5 else "#FFFFFF"
                draw.text((start_x+2, curr_y+2), line, font=final_font, fill=shadow_color)

            # Draw text
            draw.text((start_x, curr_y), line, font=final_font, fill=text_color)

        return image
