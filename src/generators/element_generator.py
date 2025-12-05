"""
Element Generator

Generates procedural visual elements (shapes, lines, grids) and loads icons.
Used to enrich layouts with non-text/image assets.
"""

from PIL import Image, ImageDraw
import random
from typing import Tuple, Optional, List

class ElementGenerator:
    def __init__(self):
        pass

    def generate_shape(self,
                      shape_type: str,
                      width: int,
                      height: int,
                      color: str,
                      opacity: float = 1.0) -> Image.Image:
        """
        Generate a procedural shape with transparency.
        """
        # Create RGBA image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Parse color to RGB
        c = self._hex_to_rgb(color)
        fill_color = (c[0], c[1], c[2], int(255 * opacity))

        if shape_type == "circle":
            # Draw circle/ellipse fitting the bounds
            draw.ellipse([0, 0, width, height], fill=fill_color)

        elif shape_type == "pill":
            # Draw rounded rectangle (pill shape)
            radius = min(width, height) // 2
            draw.rounded_rectangle([0, 0, width, height], radius=radius, fill=fill_color)

        elif shape_type == "rect":
            draw.rectangle([0, 0, width, height], fill=fill_color)

        elif shape_type == "line":
            # Draw a line (horizontal or vertical based on aspect ratio)
            if width > height:
                # Horizontal line centered vertically
                y = height // 2
                draw.line([0, y, width, y], fill=fill_color, width=height)
            else:
                # Vertical line centered horizontally
                x = width // 2
                draw.line([x, 0, x, height], fill=fill_color, width=width)

        elif shape_type == "grid":
            # Draw a grid pattern
            step = 20
            stroke_width = 1
            for x in range(0, width, step):
                draw.line([x, 0, x, height], fill=fill_color, width=stroke_width)
            for y in range(0, height, step):
                draw.line([0, y, width, y], fill=fill_color, width=stroke_width)

        return img

    def generate_icon(self, icon_name: str, size: int, color: str) -> Image.Image:
        """
        Generate or load an icon.
        """
        # Create RGBA image
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Parse color
        c = self._hex_to_rgb(color)
        fill_color = (c[0], c[1], c[2], 255)

        # Procedural fallback icons
        padding = size // 4
        if icon_name == "star":
            # Draw a simple star
            cx, cy = size // 2, size // 2
            r_outer = size // 2 - padding
            r_inner = r_outer // 2.5
            points = []
            import math
            for i in range(10):
                angle = i * math.pi / 5 - math.pi / 2
                r = r_outer if i % 2 == 0 else r_inner
                points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            draw.polygon(points, fill=fill_color)

        elif icon_name == "arrow":
            # Draw an arrow
            w, h = size - 2*padding, size - 2*padding
            x, y = padding, padding
            # Triangle part
            draw.polygon([(x+w, y+h/2), (x+w/2, y), (x+w/2, y+h)], fill=fill_color)
            # Rect part
            draw.rectangle([x, y+h/3, x+w/2, y+2*h/3], fill=fill_color)

        else:
            # Default circle
            draw.ellipse([padding, padding, size-padding, size-padding], fill=fill_color)

        return img

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])

        if len(hex_color) != 6:
            # Fallback for invalid color codes
            return (128, 128, 128)

        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (128, 128, 128)

    def generate_organic_blob(self, width: int, height: int, color: str) -> Image.Image:
        """
        Create organic, flowing shapes using bezier curves
        """
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c = self._hex_to_rgb(color)
        fill_color = (c[0], c[1], c[2], 200) # Slightly transparent

        # Simple blob implementation using a polygon with random variations
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        points = []
        import math
        num_points = 12
        for i in range(num_points):
            angle = i * 2 * math.pi / num_points
            # Randomize radius for organic feel
            r = radius * random.uniform(0.8, 1.2)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            points.append((x, y))

        # Smooth polygon (simplified as polygon for now, ideally would use bezier)
        draw.polygon(points, fill=fill_color)
        return img

    def generate_fragmented_shape(self, width: int, height: int, color: str) -> Image.Image:
        """
        Create broken/scattered geometric fragments
        """
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        c = self._hex_to_rgb(color)

        # Create multiple small shards
        num_shards = random.randint(3, 7)
        for _ in range(num_shards):
            # Random shard
            shard_w = random.randint(width // 10, width // 4)
            shard_h = random.randint(height // 10, height // 4)
            x = random.randint(0, width - shard_w)
            y = random.randint(0, height - shard_h)

            opacity = random.randint(100, 255)
            fill = (c[0], c[1], c[2], opacity)

            if random.random() > 0.5:
                draw.rectangle([x, y, x+shard_w, y+shard_h], fill=fill)
            else:
                draw.ellipse([x, y, x+shard_w, y+shard_h], fill=fill)

        return img
