"""
Smart Text Renderer (Module 9)

Overlays high-quality, readable text onto AI-generated designs.
Handles:
- Font downloading (Google Fonts)
- Smart layout based on Design Goal
- Contrast-aware color selection
"""

import os
import requests
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageFilter
import math

class TextRenderer:
    def __init__(self, font_dir: str = "data/fonts"):
        self.font_dir = Path(font_dir)
        self.font_dir.mkdir(parents=True, exist_ok=True)

        # Google Fonts URLs (Direct TTF links - using static paths)
        self.fonts = {
            "Roboto": "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Regular.ttf",
            "Roboto-Bold": "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Bold.ttf",
            "PlayfairDisplay": "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/static/PlayfairDisplay-Regular.ttf",
            "Montserrat": "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf",
            "Oswald": "https://github.com/google/fonts/raw/main/ofl/oswald/static/Oswald-Bold.ttf"
        }

        # System Font Fallbacks (Mac)
        self.system_fonts = {
            "Roboto": "/System/Library/Fonts/Helvetica.ttc",
            "Roboto-Bold": "/System/Library/Fonts/Helvetica.ttc", # Helvetica doesn't have separate bold file usually, handled by index
            "PlayfairDisplay": "/System/Library/Fonts/Times.ttc",
            "Montserrat": "/System/Library/Fonts/Supplemental/Arial.ttf",
            "Oswald": "/System/Library/Fonts/Supplemental/Impact.ttf"
        }

        # Download fonts if missing
        self._ensure_fonts()

    def _ensure_fonts(self):
        """Download missing fonts or setup fallbacks"""
        for name, url in self.fonts.items():
            font_path = self.font_dir / f"{name}.ttf"
            if not font_path.exists():
                # Try download
                print(f"Downloading font: {name}...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"❌ Failed to download {name}: {e}")
                    # Fallback to system font if download failed
                    if name in self.system_fonts and os.path.exists(self.system_fonts[name]):
                        print(f"  Using system fallback: {self.system_fonts[name]}")

    def _get_font_path(self, font_name: str) -> str:
        """Get path to font file (local download or system fallback)"""
        local_path = self.font_dir / f"{font_name}.ttf"
        if local_path.exists():
            return str(local_path)

        # Fallback
        if font_name in self.system_fonts and os.path.exists(self.system_fonts[font_name]):
            return self.system_fonts[font_name]

        return "default"

    def _get_font_for_tone(self, tone: float) -> str:
        """Select font based on emotional tone"""
        if tone < 0.4:
            return "PlayfairDisplay" # Elegant, Calm
        elif tone < 0.7:
            return "Roboto-Bold"     # Professional, Neutral
        else:
            return "Montserrat"      # Bold, Energetic

    def _get_text_color(self, image: Image.Image, box: Tuple[int, int, int, int]) -> str:
        """
        Determine text color (White/Black) based on background brightness
        box: (left, top, right, bottom)
        """
        # Crop the area where text will be
        crop = image.crop(box)
        stat = ImageStat.Stat(crop)
        avg_brightness = sum(stat.mean) / 3

        # If bright background -> Black text, else White
        return "#000000" if avg_brightness > 128 else "#FFFFFF"

    def _analyze_layout(self, image: Image.Image) -> Dict:
        """
        Analyze image to find best text placement.
        Returns a score grid (3x3) where higher score = better for text.
        """
        w, h = image.size
        # Convert to grayscale for analysis
        gray = image.convert('L')

        # Find edges (clutter)
        edges = gray.filter(ImageFilter.FIND_EDGES)

        # Grid size
        rows, cols = 3, 3
        cell_w, cell_h = w // cols, h // rows

        scores = []

        for r in range(rows):
            row_scores = []
            for c in range(cols):
                # Define cell box
                box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)

                # 1. Edge Density (Lower is better)
                edge_crop = edges.crop(box)
                edge_stat = ImageStat.Stat(edge_crop)
                edge_density = sum(edge_stat.mean) # 0-255

                # 2. Brightness Variance (Lower is better - uniform background)
                gray_crop = gray.crop(box)
                gray_stat = ImageStat.Stat(gray_crop)
                variance = sum(gray_stat.stddev)

                # 3. Center Bias (Slight penalty for center to avoid covering subject)
                # Center cell is (1, 1)
                center_penalty = 0
                if r == 1 and c == 1:
                    center_penalty = 50

                # Final Score (Lower is better, so we invert later)
                # Weighted sum: Edges matter most, then variance
                score = (edge_density * 2) + variance + center_penalty
                row_scores.append(score)
            scores.append(row_scores)

        return scores

    def _find_best_position(self, scores: List[List[float]], w: int, h: int) -> Tuple[int, int, str]:
        """
        Find the cell with the lowest clutter score.
        Returns (x, y, align)
        """
        min_score = float('inf')
        best_r, best_c = 0, 0

        for r in range(3):
            for c in range(3):
                if scores[r][c] < min_score:
                    min_score = scores[r][c]
                    best_r, best_c = r, c

        # Convert grid to coordinates
        cell_w, cell_h = w // 3, h // 3

        # Determine alignment based on column
        if best_c == 0:
            align = "left"
            x = int(cell_w * 0.1) # Padding
        elif best_c == 1:
            align = "center"
            x = w // 2
        else:
            align = "right"
            x = w - int(cell_w * 0.1)

        # Y coordinate (center of cell)
        y = (best_r * cell_h) + (cell_h // 4) # Slightly top-biased in the cell

        return x, y, align

    def _draw_text_with_shadow(self, draw, xy, text, font, fill, shadow_color="#000000", anchor=None):
        """Helper to draw text with drop shadow"""
        x, y = xy
        # Shadow offset
        off = int(font.size / 15)
        draw.text((x + off, y + off), text, font=font, fill=shadow_color, anchor=anchor)
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

    def render_text(self,
                   image: Image.Image,
                   metadata: Dict) -> Image.Image:
        """
        Overlay text onto the image based on metadata and smart analysis
        """
        draw = ImageDraw.Draw(image)
        w, h = image.size

        # Extract metadata
        goal_id = metadata.get('v_Goal', 0)
        tone = metadata.get('v_Tone', 0.5)

        # 1. Analyze Image for Layout
        layout_scores = self._analyze_layout(image)
        best_x, best_y, align = self._find_best_position(layout_scores, w, h)

        # Select Font
        font_name = self._get_font_for_tone(tone)
        font_path = self._get_font_path(font_name)

        # Define Content
        if goal_id == 0: # Inform
            headline = "INFOGRAPHIC"
            subhead = "Key Data Points"
        elif goal_id == 1: # Persuade
            headline = "SALE 50% OFF"
            subhead = "Limited Time Offer"
        elif goal_id == 2: # Entertain
            headline = "PARTY TIME"
            subhead = "Join the Fun"
        else: # Inspire
            headline = "DREAM BIG"
            subhead = "Make it Happen"

        # Load Font
        font_size = int(h * 0.12) # 12% of height (bigger)
        try:
            if font_path == "default":
                raise Exception("Use default")
            font = ImageFont.truetype(font_path, font_size)
            sub_font_path = self._get_font_path("Roboto")
            sub_font = ImageFont.truetype(sub_font_path if sub_font_path != "default" else font_path, int(font_size * 0.5))
        except:
            font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        # Determine Anchor
        if align == "left":
            anchor = "la" # Left-Ascender
        elif align == "center":
            anchor = "ma" # Middle-Ascender
        else:
            anchor = "ra" # Right-Ascender

        # Check Contrast at best position
        # Sample a region around the text
        sample_box = (
            max(0, best_x - 100),
            max(0, best_y - 50),
            min(w, best_x + 100),
            min(h, best_y + 100)
        )
        crop = image.crop(sample_box)
        stat = ImageStat.Stat(crop)
        avg_brightness = sum(stat.mean) / 3

        text_color = "#FFFFFF" if avg_brightness < 180 else "#000000"
        shadow_color = "#000000" if text_color == "#FFFFFF" else "#FFFFFF"

        # Draw Text with Shadow
        self._draw_text_with_shadow(draw, (best_x, best_y), headline, font, text_color, shadow_color, anchor=anchor)

        # Draw Subhead
        # Offset for subhead
        bbox = draw.textbbox((best_x, best_y), headline, font=font, anchor=anchor)
        text_h = bbox[3] - bbox[1]
        sub_y = best_y + text_h + 10

        self._draw_text_with_shadow(draw, (best_x, sub_y), subhead, sub_font, text_color, shadow_color, anchor=anchor)

        return image
