"""
Asset Generator

Generates individual design components:
1. Backgrounds (Gradients, Textures)
2. Hero Images (via Stable Diffusion)
"""

import math
import random
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
import torch

# Optional import - SD decoder not strictly required
try:
    from ..models.sd_decoder import SDDecoder
    HAS_SD_DECODER = True
except ImportError:
    HAS_SD_DECODER = False

from .procedural_background import RefinedBackgroundGenerator

class AssetGenerator:
    def __init__(self, device: str = "mps"):
        self.device = device
        # SD decoder removed - using pure procedural generation
        self.bg_generator = RefinedBackgroundGenerator()

    def generate_background(self, width: int, height: int, tone: float = 0.5,
                          goal: int = 0, text_zones: List[Tuple[float, float, float, float]] = None,
                          palette: Optional[dict] = None) -> Image.Image:
        """
        Generate a background asset using refined procedural generation.
        """
        return self.bg_generator.generate(width, height, tone, goal, text_zones, palette)

    def generate_hero_image(self, prompt: str, width: int, height: int) -> Image.Image:
        """
        Generate a hero image using SD.
        """
        # Enhance prompt for isolation
        enhanced_prompt = f"isolated {prompt}, white background, high quality, 3d render, detailed, professional"
        negative_prompt = "text, watermark, blurry, distorted, complex background, busy"

        # Ensure dimensions are divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        image = self.decoder.pipe(
            enhanced_prompt,
            num_inference_steps=30,
            negative_prompt=negative_prompt,
            height=height,
            width=width
        ).images[0]

        return image

    def _generate_gradient(self, width: int, height: int) -> Image.Image:
        """Generate a smooth linear gradient"""
        base = Image.new('RGB', (width, height), "#FFFFFF")
        draw = ImageDraw.Draw(base)

        # Random professional colors
        colors = [
            ((10, 20, 40), (40, 60, 100)),   # Deep Blue
            ((20, 40, 20), (50, 100, 50)),   # Forest Green
            ((40, 10, 10), (100, 40, 40)),   # Deep Red
            ((10, 10, 10), (50, 50, 50)),    # Charcoal
            ((240, 240, 250), (200, 200, 220)) # Light Gray
        ]
        c1, c2 = random.choice(colors)

        for y in range(height):
            r = int(c1[0] + (c2[0] - c1[0]) * y / height)
            g = int(c1[1] + (c2[1] - c1[1]) * y / height)
            b = int(c1[2] + (c2[2] - c1[2]) * y / height)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        return base

    def _generate_solid(self, width: int, height: int) -> Image.Image:
        """Generate solid color"""
        color = random.choice(["#FFFFFF", "#F0F0F0", "#1A1A1A", "#000000"])
        return Image.new('RGB', (width, height), color)

class EnhancedAssetGenerator(AssetGenerator):
    def generate_advanced_background(self, spec: dict, width: int, height: int) -> Image.Image:
        """
        Generate complex, layered backgrounds based on spec
        """
        # Base gradient
        base = self._create_multipoint_gradient(spec.get("gradient", {}), width, height)

        # Texture
        if spec.get("texture"):
            texture = self._generate_noise_texture(
                spec["texture"].get("type", "grain"),
                spec["texture"].get("intensity", 0.1),
                width, height
            )
            base = Image.blend(base, texture, alpha=0.15)

        return base

    def _create_multipoint_gradient(self, gradient_spec: dict, width: int, height: int) -> Image.Image:
        """
        Create gradient with custom stops
        """
        stops = gradient_spec.get("stops", [(0.0, "#FF6B6B"), (1.0, "#4ECDC4")])
        angle = gradient_spec.get("angle", 45)

        # Simplified implementation: Linear interpolation between stops
        # In a real implementation, we'd handle angle and complex easing
        base = Image.new('RGB', (width, height), "#FFFFFF")
        draw = ImageDraw.Draw(base)

        # Convert hex stops to RGB
        rgb_stops = []
        for pos, hex_c in stops:
            hex_c = hex_c.lstrip('#')
            rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            rgb_stops.append((pos, rgb))

        # Sort by position
        rgb_stops.sort(key=lambda x: x[0])

        # Draw vertical gradient for now (simplification)
        for y in range(height):
            norm_y = y / height

            # Find surrounding stops
            start_stop = rgb_stops[0]
            end_stop = rgb_stops[-1]

            for i in range(len(rgb_stops) - 1):
                if rgb_stops[i][0] <= norm_y <= rgb_stops[i+1][0]:
                    start_stop = rgb_stops[i]
                    end_stop = rgb_stops[i+1]
                    break

            # Interpolate
            t = (norm_y - start_stop[0]) / (end_stop[0] - start_stop[0]) if end_stop[0] != start_stop[0] else 0

            r = int(start_stop[1][0] + (end_stop[1][0] - start_stop[1][0]) * t)
            g = int(start_stop[1][1] + (end_stop[1][1] - start_stop[1][1]) * t)
            b = int(start_stop[1][2] + (end_stop[1][2] - start_stop[1][2]) * t)

            draw.line([(0, y), (width, y)], fill=(r, g, b))

        return base

    def _generate_noise_texture(self, type: str, intensity: float, width: int, height: int) -> Image.Image:
        """
        Generate procedural texture
        """
        img = Image.new('RGB', (width, height), (128, 128, 128))
        pixels = img.load()

        import random
        for y in range(height):
            for x in range(width):
                noise = random.randint(-int(255*intensity), int(255*intensity))
                r, g, b = pixels[x, y]
                pixels[x, y] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                )
        return img
