"""
Font Manager

Downloads and manages Google Fonts for professional typography.
"""

import os
import requests
from pathlib import Path
from typing import Dict, Tuple, Optional
import re

class FontManager:
    """Manages Google Fonts download and selection."""

    # Google Fonts direct download URLs (using Google Fonts CDN)
    FONT_URLS = {
        "Inter": {
            "Regular": "https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiA.woff2",
            "SemiBold": "https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuGKYAZ9hiA.woff2",
            "Bold": "https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuDyfAZ9hiA.woff2"
        },
        "Montserrat": {
            "Regular": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf",
            "SemiBold": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-SemiBold.ttf",
            "Bold": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"
        },
        "Poppins": {
            "Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf",
            "SemiBold": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-SemiBold.ttf",
            "Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Bold.ttf"
        }
    }

    # Tone → Font mapping
    FONT_MAP = {
        "minimalist": {
            "headline": ("Inter", "Bold"),
            "subheading": ("Inter", "SemiBold"),
            "body": ("Inter", "Regular")
        },
        "boho": {
            "headline": ("Poppins", "SemiBold"),
            "subheading": ("Poppins", "Regular"),
            "body": ("Inter", "Regular")
        },
        "memphis": {
            "headline": ("Montserrat", "Bold"),
            "subheading": ("Montserrat", "SemiBold"),
            "body": ("Poppins", "Regular")
        },
        "cyber": {
            "headline": ("Montserrat", "Bold"),
            "subheading": ("Inter", "SemiBold"),
            "body": ("Inter", "Regular")
        }
    }

    def __init__(self, fonts_dir: str = "fonts"):
        self.fonts_dir = Path(fonts_dir)
        self.fonts_dir.mkdir(exist_ok=True)

        # System font fallback
        self.fallback_font = self._get_system_font_fallback()

    def get_font_for_tone(self, tone: float, goal: int, element: str = "headline") -> str:
        """
        Get font path for a given tone and element type.

        Args:
            tone: Design tone (0-1)
            goal: Design goal (0-3)
            element: 'headline', 'subheading', or 'body'

        Returns:
            Path to TTF file
        """
        # Determine style
        if tone < 0.3:
            style = "minimalist"
        elif tone > 0.7:
            style = "memphis"
        elif goal == 3:
            style = "cyber"
        else:
            style = "boho"

        # Get font family and weight
        font_family, weight = self.FONT_MAP[style].get(element, ("Inter", "Regular"))

        # Get or download font
        font_path = self._get_font_path(font_family, weight)

        if font_path and os.path.exists(font_path):
            return font_path
        else:
            print(f"⚠️  Font {font_family}-{weight} not available, using fallback")
            return self.fallback_font

    def _get_font_path(self, family: str, weight: str) -> Optional[str]:
        """Get path to font file, downloading if necessary."""
        font_file = f"{family}-{weight}.ttf"
        font_path = self.fonts_dir / family.lower() / font_file

        # Check if already cached
        if font_path.exists():
            return str(font_path)

        # Try to download
        if family in self.FONT_URLS and weight in self.FONT_URLS[family]:
            try:
                self._download_font(family, weight)
                if font_path.exists():
                    return str(font_path)
            except Exception as e:
                print(f"⚠️  Failed to download {family}-{weight}: {e}")

        return None

    def _download_font(self, family: str, weight: str):
        """Download a font from Google Fonts GitHub."""
        url = self.FONT_URLS[family][weight]
        font_file = f"{family}-{weight}.ttf"

        # Create family directory
        family_dir = self.fonts_dir / family.lower()
        family_dir.mkdir(exist_ok=True)

        font_path = family_dir / font_file

        print(f"📥 Downloading {family} {weight}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(font_path, 'wb') as f:
                f.write(response.content)

            print(f"✅ Downloaded {font_file}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

    def ensure_essential_fonts(self):
        """Pre-download essential fonts."""
        essential = [
            ("Inter", "Regular"),
            ("Inter", "SemiBold"),
            ("Inter", "Bold"),
            ("Montserrat", "Regular"),
            ("Montserrat", "SemiBold"),
            ("Montserrat", "Bold"),
            ("Poppins", "Regular"),
            ("Poppins", "SemiBold")
        ]

        print("🔤 Ensuring essential fonts are available...")

        for family, weight in essential:
            font_path = self._get_font_path(family, weight)
            if not font_path:
                try:
                    self._download_font(family, weight)
                except:
                    pass  # Skip failed downloads

        print("✅ Font setup complete")

    def _get_system_font_fallback(self) -> str:
        """Get system font as fallback."""
        import platform

        system = platform.system()

        if system == "Darwin":  # macOS
            candidates = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSDisplay.ttf"
            ]
        elif system == "Windows":
            candidates = [
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\calibri.ttf"
            ]
        else:  # Linux
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf"
            ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return "default"  # PIL will use default font
