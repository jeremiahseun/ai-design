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
from ..models.sd_decoder import SDDecoder
from .procedural_background import RefinedBackgroundGenerator

class AssetGenerator:
    def __init__(self, device: str = "mps"):
        self.device = device
        # SD decoder removed - using pure procedural generation
        self.bg_generator = RefinedBackgroundGenerator()

    def generate_background(self, width: int, height: int, tone: float = 0.5,
                          goal: int = 0, text_zones: List[Tuple[float, float, float, float]] = None) -> Image.Image:
        """
        Generate a background asset using refined procedural generation.
        """
        return self.bg_generator.generate(width, height, tone, goal, text_zones)

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
