"""
Style Analyzer

Learns from Pinterest samples and extracts design DNA using Gemini Vision.
"""

import json
import os
import time
import random
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from PIL import Image

class StyleAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.style_database = {}

    def analyze_reference_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple Pinterest images to extract patterns
        """
        results = []
        print(f"📸 Analyzing {len(image_paths)} reference images...")

        for img_path in image_paths:
            try:
                # Load image
                if isinstance(img_path, str):
                    image = Image.open(img_path)
                else:
                    image = img_path # Assume it's already a PIL Image

                analysis = self._analyze_single_image(image)
                results.append(analysis)
            except Exception as e:
                print(f"⚠️ Failed to analyze image {img_path}: {e}")

        if not results:
            return {"error": "No images could be analyzed"}

        return self.synthesize_patterns(results)

    def _analyze_single_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a single image using Gemini Vision
        """
        prompt = """
        Analyze this design and extract:

        1. COMPOSITION STRUCTURE:
           - Layout type (asymmetric, grid-based, free-form, centered)
           - Element placement patterns (overlapping, scattered, aligned)
           - Visual weight distribution (balanced, intentionally unbalanced)

        2. TYPOGRAPHY CHARACTERISTICS:
           - Hierarchy levels (how many distinct sizes/weights)
           - Type treatment (rotation, cropping, baseline shifts)
           - Font pairing style (contrast level, mood)

        3. COLOR STRATEGY:
           - Gradient complexity (stops, direction, blend modes)
           - Color relationships (analogous, complementary, monochrome+accent)
           - Contrast approach (high/low, where applied)

        4. DEPTH TECHNIQUES:
           - Layering approach (overlaps, transparency, shadows)
           - Texture application (noise, grain, patterns)
           - Dimensional effects (3D, depth illusion)

        5. DESIGN MOVEMENT:
           - Visual flow direction (eye path through design)
           - Energy level (static, dynamic, kinetic)
           - Focal point strategy (single hero, multiple anchors)

        6. CONTEMPORARY STYLE MARKERS:
           - Specific trend (glassmorphism, brutalism, neo-geo, etc.)
           - Era feel (2020-2022, 2023-2024, 2025+)
           - Uniqueness factors (what makes it memorable)

        Respond in structured JSON format.
        """

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [prompt, image],
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    print(f"Error analyzing image: {e}")
                    return {}
        return {}

    def synthesize_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract common patterns across multiple designs
        """
        synthesis_prompt = f"""
        Given these {len(analyses)} design analyses, identify:

        1. RECURRING PATTERNS (what shows up in 60%+ of designs)
        2. DISTINCTIVE TECHNIQUES (unique approaches worth copying)
        3. RULE VIOLATIONS (intentional breaks from traditional design rules)
        4. SUCCESS FACTORS (what makes these designs work)

        Create a design rulebook that could replicate this aesthetic.
        Format as actionable rules in JSON:
        {{
            "name": "Suggested Style Name",
            "composition_rules": ["rule 1", "rule 2"],
            "color_rules": ["rule 1", "rule 2"],
            "typography_rules": ["rule 1", "rule 2"],
            "depth_rules": ["rule 1", "rule 2"],
            "examples": ["description of example 1"]
        }}

        Analyses: {json.dumps(analyses)}
        """

        try:
            response = self.model.generate_content(
                synthesis_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error synthesizing patterns: {e}")
            return {
                "name": "Fallback Style",
                "composition_rules": ["Use balanced layout"],
                "color_rules": ["Use high contrast"],
                "typography_rules": ["Clear hierarchy"],
                "depth_rules": ["Flat"]
            }

    def get_profile(self, profile_name: str) -> Any:
        """
        Retrieve a pre-defined or cached profile
        """
        # For now, return a default if not found
        # In a real system, this would load from a database or file
        from src.analysis.style_profiles import StyleProfile

        if profile_name == "modern_bold":
            profile = StyleProfile("Modern Bold")
            profile.composition_rules = [
                "Use asymmetric balance with 60/40 weight distribution",
                "Place primary element in upper-left or lower-right power zones",
                "Allow at least one element to break canvas boundaries",
                "Create visual tension through unexpected proximity"
            ]
            profile.color_rules = [
                "Use 4-5 stop gradients with non-linear easing",
                "Apply multiply blend mode to overlapping elements",
                "Introduce vibrant accent (HSB: S>80, B>70) against muted base",
                "Add 3-5% noise to gradients for depth"
            ]
            profile.typography_rules = [
                "Use extreme scale contrast (Headline 3x larger than body)",
                "Tight leading (0.9-1.0) for display type",
                "Mix serif and sans-serif for tension"
            ]
            profile.depth_rules = [
                "Use soft, large drop shadows for float effect",
                "Layer elements with varying opacity"
            ]
            return profile

        return StyleProfile("Default")
