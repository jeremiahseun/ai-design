"""
Conference Designer (Temporary Wrapper)

A specialized wrapper for creating conference/event designs.
This will be replaced by the Universal Design Agent in the future.
"""

import torch
from pathlib import Path
from typing import Optional, Dict
from PIL import Image
from src.models.sd_decoder import SDDecoder
from src.generators.text_renderer import TextRenderer

class ConferenceDesigner:
    def __init__(self, device: str = "mps"):
        self.device = device
        self.decoder = SDDecoder(device=device)
        self.renderer = TextRenderer()

        # Load LoRA if available
        lora_path = Path('checkpoints/pytorch_lora_weights.safetensors')
        if lora_path.exists():
            self.decoder.load_lora(str(lora_path))
            print("✨ Using Fine-Tuned LoRA Model")

    def create_design(self,
                     event_name: str,
                     tagline: str = "",
                     details: str = "",
                     logo_path: Optional[str] = None,
                     style: str = "professional tech conference, modern, clean",
                     variants: int = 1) -> list:
        """
        Create conference design(s)

        Args:
            event_name: Main conference name
            tagline: Subtitle/tagline
            details: Date, location, etc.
            logo_path: Path to logo image (optional)
            style: Style description for SD
            variants: Number of design variations to generate

        Returns:
            List of PIL Images
        """
        designs = []

        # Build SD prompt
        sd_prompt = f"{style}, conference poster, event design, professional graphic design, high quality, 4k, trending on behance, vector art style"

        # Metadata (hardcoded for conference use case)
        metadata = {
            'v_Goal': 0,    # Inform
            'v_Tone': 0.5,  # Professional/Neutral
            'v_Format': 0   # Poster
        }

        for i in range(variants):
            print(f"\n🎨 Generating variant {i+1}/{variants}...")

            # 1. Generate background with SD
            negative_prompt = "text, watermark, signature, blurry, low quality, distorted, people, faces"

            # Add variety by varying the seed
            image = self.decoder.pipe(
                sd_prompt,
                num_inference_steps=30,
                negative_prompt=negative_prompt,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]

            # 2. Add logo if provided
            if logo_path and Path(logo_path).exists():
                image = self._add_logo(image, logo_path)

            # 3. Render text using custom content
            image = self._render_conference_text(
                image,
                event_name,
                tagline,
                details,
                metadata
            )

            designs.append(image)

        return designs

    def _add_logo(self, image: Image.Image, logo_path: str) -> Image.Image:
        """Add logo to top-left corner"""
        logo = Image.open(logo_path).convert('RGBA')

        # Resize logo to ~15% of image width
        w, h = image.size
        logo_w = int(w * 0.15)
        aspect = logo.height / logo.width
        logo_h = int(logo_w * aspect)
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        # Paste in top-left with padding
        padding = int(w * 0.05)

        # Create a copy to paste logo
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        image.paste(logo, (padding, padding), logo)
        return image.convert('RGB')

    def _render_conference_text(self,
                                image: Image.Image,
                                event_name: str,
                                tagline: str,
                                details: str,
                                metadata: Dict) -> Image.Image:
        """
        Custom text rendering for conference (overrides TextRenderer defaults)
        """
        # Create custom metadata with conference text
        custom_meta = metadata.copy()
        custom_meta['_custom_text'] = {
            'headline': event_name,
            'subhead': tagline,
            'details': details
        }

        # Use the smart renderer but inject custom text
        return self._custom_render(image, custom_meta)

    def _custom_render(self, image: Image.Image, metadata: Dict) -> Image.Image:
        """
        Temporarily override the renderer's text logic
        """
        from PIL import ImageDraw, ImageFont, ImageStat

        draw = ImageDraw.Draw(image)
        w, h = image.size

        # Get custom text
        custom_text = metadata.get('_custom_text', {})
        headline = custom_text.get('headline', 'EVENT')
        subhead = custom_text.get('subhead', '')
        details = custom_text.get('details', '')

        # Use the same smart layout analysis
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
        if align == "left":
            anchor = "la"
        elif align == "center":
            anchor = "ma"
        else:
            anchor = "ra"

        # Check contrast
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

        # Draw headline
        self.renderer._draw_text_with_shadow(draw, (best_x, best_y), headline, font, text_color, shadow_color, anchor=anchor)

        # Draw subhead
        bbox = draw.textbbox((best_x, best_y), headline, font=font, anchor=anchor)
        text_h = bbox[3] - bbox[1]
        sub_y = best_y + text_h + 10

        if subhead:
            self.renderer._draw_text_with_shadow(draw, (best_x, sub_y), subhead, sub_font, text_color, shadow_color, anchor=anchor)

            # Draw details below subhead
            if details:
                bbox2 = draw.textbbox((best_x, sub_y), subhead, font=sub_font, anchor=anchor)
                detail_y = sub_y + (bbox2[3] - bbox2[1]) + 15
                self.renderer._draw_text_with_shadow(draw, (best_x, detail_y), details, detail_font, text_color, shadow_color, anchor=anchor)
        elif details:
            # No subhead, draw details directly
            self.renderer._draw_text_with_shadow(draw, (best_x, sub_y), details, detail_font, text_color, shadow_color, anchor=anchor)

        return image
