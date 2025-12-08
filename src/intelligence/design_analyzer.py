"""
Design Analyzer using Gemini 1.5 Flash (Free Tier).
Extracts deep design intelligence from images at zero cost.
"""

import os
import time
import json
import typing
from pathlib import Path
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from PIL import Image

class DesignAnalyzer:
    """
    Analyzes design images to extract structural patterns using Gemini 1.5 Flash.
    Optimized for the Free Tier (15 RPM limit).
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer.

        Args:
            api_key: Google API Key. If None, looks for GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        # Fallback: Try to load from .env file manually if not in environment
        if not self.api_key and os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in .env or pass it explicitly.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # Rate limiting stats
        self.last_request_time = 0
        self.min_interval = 4.0  # 4 seconds = 15 requests per minute max (safe buffer)

    def analyze_design(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single design image to extract structure.
        """
        self._rate_limit()

        try:
            image = Image.open(image_path)

            prompt = """
            Analyze this design image and extract its highly detailed structural DNA in JSON format.
            I need to "reverse engineer" this design to create a precise programmatic template.

            Extract the following granular details:

            1. "layout_structure": {
                "type": [centered, asymmetric, split, grid, bottom_anchor, top_anchor, diagonal, frame],
                "description": "Brief description of the overall arrangement",
                "margin_usage": "How is white space used? (tight, spacious, uneven)"
            }

            2. "visual_hierarchy_path": ["List elements in order of visual prominence, e.g., Headline -> Product Image -> CTA"]

            3. "background": {
                "type": [solid, gradient, photo, texture, pattern],
                "color_hex": "dominant background color if valid",
                "overlay": {
                    "present": boolean,
                    "type": [gradient_fade, solid_dim, blur],
                    "opacity_estimate": 0.0 to 1.0 (approximate)
                }
            }

            4. "typography_detailed": List of text blocks. For each block:
                - "role": [headline, subheading, body, cta, caption, price, tag]
                - "content_type": [static_text, dynamic_data]
                - "position_exact": {"x": 0.5, "y": 0.5} (normalized 0-1 coordinates of center)
                - "box_dimensions": {"width": 0.5, "height": 0.1} (normalized)
                - "alignment_inner": [left, center, right, justified]
                - "alignment_page": "How is it aligned relative to the page? (e.g., center_axis, left_margin)"
                - "style": {
                    "weight": [light, regular, bold, extra_bold],
                    "case": [uppercase, lowercase, title_case, sentence_case],
                    "letter_spacing": [tight, normal, wide],
                    "color_category": [light, dark, accent, brand],
                    "contrast_ratio": "high/medium/low vs background"
                }

            5. "images_and_assets": List of visual assets. For each:
                - "type": [photo_main, photo_secondary, background_texture, 3d_render]
                - "role": [hero, support, background, decoration]
                - "position": {"x": 0.5, "y": 0.5}
                - "size": {"width": 0.5, "height": 0.5}
                - "styling": {
                    "opacity": 0.0 to 1.0,
                    "corner_radius": [none, small, medium, full_round],
                    "shadow": boolean,
                    "border": boolean,
                    "mask_shape": [rect, circle, blob, none]
                }

            6. "icons_and_ui": List of UI elements/icons. For each:
                - "type": [icon_outline, icon_solid, arrow, line, button, container]
                - "purpose": [navigation, decoration, bullet_point, CTA_container]
                - "position": {"x": 0.5, "y": 0.5}
                - "consistency": "Is it part of a set/pattern?"

            7. "logo_analysis": {
                "present": boolean,
                "position_category": [top_left, top_center, top_right, bottom_center, floating],
                "position_exact": {"x": 0.0, "y": 0.0},
                "size_category": [discreet, prominent, dominant],
                "integration": [overlay, standalone, watermark]
            }

            Return ONLY valid JSON.
            """

            response = self.model.generate_content([prompt, image])

            # Extract JSON from response
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            return data

        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return {"error": str(e)}

    def analyze_batch(self, image_paths: List[str], output_file: str = "data/design_patterns_raw.json"):
        """
        Analyze a batch of images and save results incrementally.
        """
        results = []

        # Load existing if any to resume
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                analyzed_paths = {r.get('file_path') for r in results}
                image_paths = [p for p in image_paths if p not in analyzed_paths]
                print(f"Resuming analysis. {len(results)} already done, {len(image_paths)} remaining.")
            except:
                pass

        print(f"Starting analysis of {len(image_paths)} designs with Gemini 1.5 Flash...")

        for i, path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] Analyzing {os.path.basename(path)}...")

            analysis = self.analyze_design(path)

            if "error" not in analysis:
                result_record = {
                    "file_path": path,
                    "filename": os.path.basename(path),
                    "analysis": analysis,
                    "timestamp": time.time()
                }
                results.append(result_record)

                # specific checkpointing
                if i % 5 == 0:
                    self._save_results(results, output_file)
            else:
                 print(f"Skipping {os.path.basename(path)} due to error.")

        self._save_results(results, output_file)
        print(f"Analysis complete. Results saved to {output_file}")

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _save_results(self, data, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    # Test run
    import glob

    # Analyze a few images from the scrapers directory
    image_dir = "src/scrapers/images"
    all_images = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")

    if not all_images:
        print(f"No images found in {image_dir}")
        exit()

    # Take first 5 for test
    test_batch = all_images[:5]

    analyzer = DesignAnalyzer()
    analyzer.analyze_batch(test_batch, "data/design_analysis_test.json")
