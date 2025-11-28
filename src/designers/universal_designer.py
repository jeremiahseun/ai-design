"""
Universal Designer

Combines the Design Agent with the existing pipeline to create designs from
natural language prompts.
"""

import torch
from pathlib import Path
from typing import Optional, List
from PIL import Image

from src.agents.design_agent import UniversalDesignAgent
from src.agents.design_spec import DesignSpec
from src.models.sd_decoder import SDDecoder
from src.generators.text_renderer import TextRenderer

class UniversalDesigner:
    """
    End-to-end designer that takes natural language and outputs designs
    """

    def __init__(self,
                 device: str = "mps",
                 api_key: Optional[str] = None):
        """
        Initialize the universal designer

        Args:
            device: Device for SD (mps/cuda/cpu)
            api_key: Anthropic API key for the agent
        """
        print("🚀 Initializing Universal Designer...")

        # Initialize components
        self.agent = UniversalDesignAgent(api_key=api_key)
        self.decoder = SDDecoder(device=device)
        self.renderer = TextRenderer()
        self.device = device

        # Load LoRA if available
        lora_path = Path('checkpoints/pytorch_lora_weights.safetensors')
        if lora_path.exists():
            self.decoder.load_lora(str(lora_path))
            print("✨ Using Fine-Tuned LoRA Model")

        print("✅ Universal Designer ready!")

    def create_design(self,
                     prompt: str,
                     variants: int = 1,
                     logo_path: Optional[str] = None) -> tuple[DesignSpec, List[Image.Image]]:
        """
        Create design(s) from a natural language prompt

        Args:
            prompt: Natural language design request
            variants: Number of design variations to generate
            logo_path: Override logo path (if not in prompt)

        Returns:
            (DesignSpec, List of PIL Images)
        """
        # 1. Interpret prompt
        spec = self.agent.interpret_prompt(prompt)

        # Override logo if provided
        if logo_path:
            spec.logo_path = logo_path

        # 2. Generate designs
        designs = self._generate_from_spec(spec, variants)

        return spec, designs

    def _generate_from_spec(self,
                           spec: DesignSpec,
                           variants: int = 1) -> List[Image.Image]:
        """
        Generate designs from a DesignSpec

        Args:
            spec: Design specification
            variants: Number of variants

        Returns:
            List of PIL Images
        """
        designs = []

        # Build SD prompt
        sd_prompt = f"{spec.style_prompt}, {spec.format_name.lower()} design, professional graphic design, high quality, 4k, trending on behance, vector art style"
        negative_prompt = "text, watermark, signature, blurry, low quality, distorted, people, faces"

        print(f"\n🎨 Generating {variants} design variant(s)...")

        for i in range(variants):
            # Generate background
            image = self.decoder.pipe(
                sd_prompt,
                num_inference_steps=30,
                negative_prompt=negative_prompt,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]

            # Add logo if specified
            if spec.logo_path and Path(spec.logo_path).exists():
                image = self._add_logo(image, spec.logo_path)

            # Render text
            metadata = {
                'v_Goal': spec.goal,
                'v_Tone': spec.tone,
                'v_Format': spec.format,
                '_custom_text': {
                    'headline': spec.content.headline,
                    'subhead': spec.content.subheading,
                    'details': spec.content.details
                }
            }

            image = self._render_custom_text(image, metadata)
            designs.append(image)

            print(f"  ✅ Variant {i+1} complete")

        return designs

    def _add_logo(self, image: Image.Image, logo_path: str) -> Image.Image:
        """Add logo to top-left corner"""
        logo = Image.open(logo_path).convert('RGBA')
        w, h = image.size
        logo_w = int(w * 0.15)
        aspect = logo.height / logo.width
        logo_h = int(logo_w * aspect)
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        padding = int(w * 0.05)

        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image.paste(logo, (padding, padding), logo)
        return image.convert('RGB')

    def _render_custom_text(self, image: Image.Image, metadata: dict) -> Image.Image:
        """Render custom text onto image using smart layout"""
        from PIL import ImageDraw, ImageFont, ImageStat

        draw = ImageDraw.Draw(image)
        w, h = image.size

        # Get custom text
        custom_text = metadata.get('_custom_text', {})
        headline = custom_text.get('headline', 'DESIGN')
        subhead = custom_text.get('subhead', '')
        details = custom_text.get('details', '')

        # Use smart layout analysis
        layout_scores = self.renderer._analyze_layout(image)
        best_x, best_y, align = self.renderer._find_best_position(layout_scores, w, h)

        # Get font
        tone = metadata.get('v_Tone', 0.5)
        font_name = self.renderer._get_font_for_tone(tone)
        font_path = self.renderer._get_font_path(font_name)

        # Load fonts
        font_size = int(h * 0.12)
        try:
            if font_path == "default":
                raise Exception("Use default")
            font = ImageFont.truetype(font_path, font_size)
            sub_font = ImageFont.truetype(font_path, int(font_size * 0.5))
            detail_font = ImageFont.truetype(font_path, int(font_size * 0.35))
        except:
            font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
            detail_font = ImageFont.load_default()

        # Determine anchor
        anchor = {"left": "la", "center": "ma", "right": "ra"}[align]

        # Check contrast
        sample_box = (max(0, best_x - 100), max(0, best_y - 50), min(w, best_x + 100), min(h, best_y + 100))
        crop = image.crop(sample_box)
        stat = ImageStat.Stat(crop)
        avg_brightness = sum(stat.mean) / 3

        text_color = "#FFFFFF" if avg_brightness < 180 else "#000000"
        shadow_color = "#000000" if text_color == "#FFFFFF" else "#FFFFFF"

        # Draw texts
        self.renderer._draw_text_with_shadow(draw, (best_x, best_y), headline, font, text_color, shadow_color, anchor=anchor)

        bbox = draw.textbbox((best_x, best_y), headline, font=font, anchor=anchor)
        text_h = bbox[3] - bbox[1]
        sub_y = best_y + text_h + 10

        if subhead:
            self.renderer._draw_text_with_shadow(draw, (best_x, sub_y), subhead, sub_font, text_color, shadow_color, anchor=anchor)
            if details:
                bbox2 = draw.textbbox((best_x, sub_y), subhead, font=sub_font, anchor=anchor)
                detail_y = sub_y + (bbox2[3] - bbox2[1]) + 15
                self.renderer._draw_text_with_shadow(draw, (best_x, detail_y), details, detail_font, text_color, shadow_color, anchor=anchor)
        elif details:
            self.renderer._draw_text_with_shadow(draw, (best_x, sub_y), details, detail_font, text_color, shadow_color, anchor=anchor)

        return image
