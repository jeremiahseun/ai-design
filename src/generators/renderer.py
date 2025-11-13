"""
Module 2: Renderer - JSON to P_Image
Renders design briefs (JSON) into actual images using Pillow.
This creates the P_Image tensors for training.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Dict, Any, Tuple, Optional
import os


class DesignRenderer:
    """
    Renders design briefs into images (P_Image)
    Uses Pillow to draw elements onto a canvas
    """

    def __init__(self, canvas_size: int = 256):
        """
        Initialize renderer
        Args:
            canvas_size: Size of the canvas (square)
        """
        self.canvas_size = canvas_size
        self.background_color = (255, 255, 255)  # White background

        # Try to load a font, fallback to default
        self.fonts = self._load_fonts()

    def _load_fonts(self) -> Dict[str, Any]:
        """
        Load fonts for text rendering
        Attempts to use system fonts, falls back to PIL default
        """
        fonts = {}

        # Common font paths on macOS
        font_paths = [
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/SFNSDisplay.ttf',
            '/Library/Fonts/Arial.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
        ]

        # Try to load fonts
        for size in [14, 16, 18, 20, 28, 32]:
            try:
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        fonts[size] = ImageFont.truetype(font_path, size)
                        break
                else:
                    # Fallback to default
                    fonts[size] = ImageFont.load_default()
            except Exception:
                fonts[size] = ImageFont.load_default()

        return fonts

    def _get_font(self, size: int) -> Any:
        """
        Get font for given size
        """
        # Find closest available font size
        available_sizes = sorted(self.fonts.keys())
        closest_size = min(available_sizes, key=lambda x: abs(x - size))
        return self.fonts[closest_size]

    def render(self, design: Dict[str, Any]) -> np.ndarray:
        """
        Render a design brief to an image
        Args:
            design: Design dictionary from generator
        Returns:
            numpy array [H, W, 3] in range [0, 255]
        """
        # Create canvas
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), self.background_color)
        draw = ImageDraw.Draw(img)

        # Import color map from generator
        from generators.generator import COLOR_MAP

        # Render each element
        for element in design['elements']:
            self._render_element(draw, element, COLOR_MAP)

        # Convert to numpy array
        return np.array(img)

    def _render_element(self, draw: ImageDraw, element: Dict[str, Any], color_map: Dict):
        """
        Render a single element onto the canvas
        """
        elem_type = element['type']
        pos = element['pos']
        box = element['box']  # (width, height)
        color_id = element['color_id']
        content = element['content']

        # Get RGB color
        color = color_map.get(color_id, (128, 128, 128))

        if elem_type == 'image':
            # Render as a filled rectangle with border
            x1, y1 = pos
            x2, y2 = x1 + box[0], y1 + box[1]
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

        elif elem_type in ['headline', 'subheadline', 'body', 'cta']:
            # Render as text
            font_size = element.get('font_size', 14)
            font = self._get_font(font_size)

            x, y = pos

            # For CTA, draw a background box first
            if elem_type == 'cta':
                x1, y1 = x, y
                x2, y2 = x + box[0], y + box[1]
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

                # Draw text in contrasting color (white or black)
                text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
                # Center text in box
                text_bbox = draw.textbbox((0, 0), content, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x + (box[0] - text_width) // 2
                text_y = y + (box[1] - text_height) // 2
                draw.text((text_x, text_y), content, fill=text_color, font=font)
            else:
                # Regular text
                draw.text((x, y), content, fill=color, font=font)

    def render_batch(self, designs: list) -> np.ndarray:
        """
        Render multiple designs
        Args:
            designs: List of design dictionaries
        Returns:
            numpy array [B, H, W, 3]
        """
        batch = []
        for design in designs:
            img = self.render(design)
            batch.append(img)

        return np.stack(batch, axis=0)

    def save_image(self, img_array: np.ndarray, filepath: str):
        """
        Save rendered image to file
        Args:
            img_array: numpy array [H, W, 3]
            filepath: output path
        """
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(filepath)

    def render_to_tensor(self, design: Dict[str, Any]):
        """
        Render and convert directly to P_Image tensor
        Note: This requires torch, so we import locally
        """
        try:
            import sys
            sys.path.append('src')
            from core.schemas import P_Image

            # Render to numpy
            img_array = self.render(design)

            # Convert to P_Image tensor
            return P_Image.from_numpy(img_array)
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion. Use render() for numpy arrays.")


def visualize_design(design: Dict[str, Any], output_path: str = 'data/rendered_design.png'):
    """
    Helper function to render and save a design
    """
    renderer = DesignRenderer()
    img = renderer.render(design)
    renderer.save_image(img, output_path)
    print(f"Design rendered and saved to {output_path}")
    return img


if __name__ == '__main__':
    # Test the renderer
    import sys
    sys.path.append('src')
    from generators.generator import DesignGenerator

    print("=" * 60)
    print("Design Renderer Test")
    print("=" * 60)

    # Generate a design
    gen = DesignGenerator(seed=42)
    design = gen.generate()

    print(f"\nRendering design:")
    print(f"  Layout: {design['layout']}")
    print(f"  Format: {design['format']}")
    print(f"  Elements: {len(design['elements'])}")

    # Render it
    renderer = DesignRenderer()
    img = renderer.render(design)

    print(f"\nRendered image shape: {img.shape}")
    print(f"Pixel range: [{img.min()}, {img.max()}]")

    # Save it
    renderer.save_image(img, 'data/test_render.png')
    print(f"Saved to: data/test_render.png")

    # Test batch rendering
    designs = gen.generate_batch(3)
    batch = renderer.render_batch(designs)
    print(f"\nBatch rendering shape: {batch.shape}")

    print("\n" + "=" * 60)
