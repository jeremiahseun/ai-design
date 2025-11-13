"""
Module 1: JSON Generator - Source of Truth
Procedurally generates design briefs as JSON/Dict structures.
These JSONs are the ground truth for all synthetic data.
"""

import random
from typing import Dict, List, Tuple, Any
import json


class DesignGenerator:
    """
    Generates procedural design briefs as structured dictionaries.
    This is the "Source of Truth" for synthetic data generation.
    """

    # Design parameters
    LAYOUTS = ['left_aligned', 'center_aligned', 'mixed']
    FORMATS = ['poster', 'social_post', 'flyer', 'banner']
    GOALS = [
        'promote_event', 'announce_product', 'share_info',
        'inspire', 'educate', 'advertise', 'celebrate',
        'inform', 'motivate', 'brand_awareness'
    ]

    # Color palettes (as color_ids for simplicity)
    PALETTES = [
        {'name': 'monochrome', 'colors': [0, 1, 2]},  # Black, white, gray
        {'name': 'vibrant', 'colors': [3, 4, 5]},     # Red, blue, yellow
        {'name': 'pastel', 'colors': [6, 7, 8]},      # Soft pink, blue, green
        {'name': 'corporate', 'colors': [9, 10, 11]}, # Navy, gray, white
        {'name': 'warm', 'colors': [12, 13, 14]},     # Orange, red, yellow
        {'name': 'cool', 'colors': [15, 16, 17]},     # Blue, cyan, purple
    ]

    # Content templates
    CONTENT_TYPES = ['headline', 'subheadline', 'body', 'cta', 'logo', 'image']

    def __init__(self, seed: int = None):
        """
        Initialize generator with optional seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate(self) -> Dict[str, Any]:
        """
        Generate a complete design brief
        Returns: Dictionary containing elements and metadata
        """
        # Select layout and format
        layout = random.choice(self.LAYOUTS)
        format_type = random.choice(self.FORMATS)
        palette = random.choice(self.PALETTES)
        goal = random.choice(self.GOALS)

        # Generate elements based on layout
        elements = self._generate_elements(layout, palette)

        # Create metadata
        meta = self._generate_meta(goal, format_type, palette)

        return {
            'layout': layout,
            'format': format_type,
            'palette': palette,
            'elements': elements,
            'meta': meta
        }

    def _generate_elements(self, layout: str, palette: Dict) -> List[Dict[str, Any]]:
        """
        Generate design elements based on layout type
        Each element has: type, content, pos, box, color_id, hierarchy
        """
        elements = []
        canvas_size = 256

        if layout == 'left_aligned':
            elements = self._generate_left_aligned(canvas_size, palette)
        elif layout == 'center_aligned':
            elements = self._generate_center_aligned(canvas_size, palette)
        else:  # mixed
            elements = self._generate_mixed(canvas_size, palette)

        return elements

    def _generate_left_aligned(self, size: int, palette: Dict) -> List[Dict[str, Any]]:
        """Generate left-aligned layout"""
        elements = []
        margin = 20
        x_start = margin
        y_pos = margin

        # Headline (top left)
        elements.append({
            'type': 'headline',
            'content': 'HEADLINE TEXT',
            'pos': (x_start, y_pos),
            'box': (size - 2 * margin, 40),  # (width, height)
            'color_id': palette['colors'][0],
            'hierarchy': 1.0,  # Highest priority
            'font_size': 32
        })
        y_pos += 60

        # Subheadline
        elements.append({
            'type': 'subheadline',
            'content': 'Subheadline text here',
            'pos': (x_start, y_pos),
            'box': (size - 2 * margin, 25),
            'color_id': palette['colors'][1],
            'hierarchy': 0.8,
            'font_size': 20
        })
        y_pos += 45

        # Body text
        elements.append({
            'type': 'body',
            'content': 'Body text content',
            'pos': (x_start, y_pos),
            'box': (size - 2 * margin - 40, 60),
            'color_id': palette['colors'][1],
            'hierarchy': 0.5,
            'font_size': 14
        })
        y_pos += 80

        # CTA (Call to action)
        elements.append({
            'type': 'cta',
            'content': 'LEARN MORE',
            'pos': (x_start, y_pos),
            'box': (120, 35),
            'color_id': palette['colors'][2],
            'hierarchy': 0.9,
            'font_size': 16
        })

        # Add an image placeholder
        elements.append({
            'type': 'image',
            'content': 'placeholder',
            'pos': (size - margin - 80, size // 2),
            'box': (60, 60),
            'color_id': palette['colors'][2],
            'hierarchy': 0.6,
            'font_size': 0
        })

        return elements

    def _generate_center_aligned(self, size: int, palette: Dict) -> List[Dict[str, Any]]:
        """Generate center-aligned layout"""
        elements = []
        margin = 30
        center_x = size // 2

        y_pos = margin + 20

        # Headline (centered)
        headline_width = 200
        elements.append({
            'type': 'headline',
            'content': 'CENTERED TITLE',
            'pos': (center_x - headline_width // 2, y_pos),
            'box': (headline_width, 40),
            'color_id': palette['colors'][0],
            'hierarchy': 1.0,
            'font_size': 32
        })
        y_pos += 60

        # Subheadline (centered)
        sub_width = 180
        elements.append({
            'type': 'subheadline',
            'content': 'Centered subtitle',
            'pos': (center_x - sub_width // 2, y_pos),
            'box': (sub_width, 25),
            'color_id': palette['colors'][1],
            'hierarchy': 0.8,
            'font_size': 20
        })
        y_pos += 50

        # Image (centered)
        img_size = 80
        elements.append({
            'type': 'image',
            'content': 'placeholder',
            'pos': (center_x - img_size // 2, y_pos),
            'box': (img_size, img_size),
            'color_id': palette['colors'][2],
            'hierarchy': 0.7,
            'font_size': 0
        })
        y_pos += 100

        # CTA (centered)
        cta_width = 140
        elements.append({
            'type': 'cta',
            'content': 'GET STARTED',
            'pos': (center_x - cta_width // 2, y_pos),
            'box': (cta_width, 35),
            'color_id': palette['colors'][0],
            'hierarchy': 0.9,
            'font_size': 18
        })

        return elements

    def _generate_mixed(self, size: int, palette: Dict) -> List[Dict[str, Any]]:
        """Generate mixed alignment layout"""
        elements = []
        margin = 25

        # Top left headline
        elements.append({
            'type': 'headline',
            'content': 'MIXED LAYOUT',
            'pos': (margin, margin),
            'box': (150, 35),
            'color_id': palette['colors'][0],
            'hierarchy': 1.0,
            'font_size': 28
        })

        # Right-aligned subheadline
        elements.append({
            'type': 'subheadline',
            'content': 'Right aligned',
            'pos': (size - margin - 140, margin + 50),
            'box': (140, 20),
            'color_id': palette['colors'][1],
            'hierarchy': 0.7,
            'font_size': 18
        })

        # Center image
        img_size = 70
        elements.append({
            'type': 'image',
            'content': 'placeholder',
            'pos': (size // 2 - img_size // 2, size // 2 - img_size // 2),
            'box': (img_size, img_size),
            'color_id': palette['colors'][2],
            'hierarchy': 0.6,
            'font_size': 0
        })

        # Bottom left body
        elements.append({
            'type': 'body',
            'content': 'Body text',
            'pos': (margin, size - margin - 60),
            'box': (100, 50),
            'color_id': palette['colors'][1],
            'hierarchy': 0.5,
            'font_size': 14
        })

        # Bottom right CTA
        elements.append({
            'type': 'cta',
            'content': 'CLICK HERE',
            'pos': (size - margin - 110, size - margin - 40),
            'box': (110, 30),
            'color_id': palette['colors'][0],
            'hierarchy': 0.85,
            'font_size': 16
        })

        return elements

    def _generate_meta(self, goal: str, format_type: str, palette: Dict) -> Dict[str, Any]:
        """
        Generate V_Meta semantic metadata
        """
        goal_id = self.GOALS.index(goal)
        format_id = self.FORMATS.index(format_type)
        tone = random.uniform(0.3, 0.9)  # Emotional tone

        return {
            'v_Goal': goal_id,
            'v_Tone': tone,
            'v_Content': f"{goal} {format_type} design",
            'v_Format': format_id
        }

    def generate_batch(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate multiple design briefs
        """
        return [self.generate() for _ in range(n)]

    def save_json(self, design: Dict[str, Any], filepath: str):
        """
        Save design brief to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(design, f, indent=2)

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """
        Load design brief from JSON file
        """
        with open(filepath, 'r') as f:
            return json.load(f)


# Color ID to RGB mapping (for visualization)
COLOR_MAP = {
    0: (0, 0, 0),         # Black
    1: (255, 255, 255),   # White
    2: (128, 128, 128),   # Gray
    3: (220, 50, 50),     # Red
    4: (50, 100, 220),    # Blue
    5: (230, 220, 50),    # Yellow
    6: (255, 182, 193),   # Pink
    7: (173, 216, 230),   # Light Blue
    8: (144, 238, 144),   # Light Green
    9: (25, 25, 112),     # Navy
    10: (169, 169, 169),  # Dark Gray
    11: (245, 245, 245),  # Off White
    12: (255, 140, 0),    # Orange
    13: (200, 50, 50),    # Dark Red
    14: (255, 215, 0),    # Gold
    15: (70, 130, 180),   # Steel Blue
    16: (0, 206, 209),    # Cyan
    17: (147, 112, 219),  # Purple
}


if __name__ == '__main__':
    # Test the generator
    gen = DesignGenerator(seed=42)

    print("=" * 60)
    print("Design Generator Test")
    print("=" * 60)

    # Generate 5 designs
    for i in range(5):
        design = gen.generate()
        print(f"\nDesign {i + 1}:")
        print(f"  Layout: {design['layout']}")
        print(f"  Format: {design['format']}")
        print(f"  Palette: {design['palette']['name']}")
        print(f"  Goal: {design['meta']['v_Content']}")
        print(f"  Elements: {len(design['elements'])} items")

        # Save first design as example
        if i == 0:
            gen.save_json(design, 'data/example_design.json')
            print(f"  -> Saved to data/example_design.json")

    print("\n" + "=" * 60)
