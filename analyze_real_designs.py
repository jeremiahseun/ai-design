"""
Real Design Analyzer

Analyzes actual professional designs to extract common patterns:
- Layout structures (text zones, visual zones)
- Typography placement patterns
- Element types and positioning
- Color distribution
- Aspect ratios

This creates data-driven templates based on real design principles.
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class DesignPatternAnalyzer:
    def __init__(self, designs_dir: str = "data/real_designs"):
        self.designs_dir = Path(designs_dir)
        self.images_dir = self.designs_dir / "images"
        self.metadata_dir = self.designs_dir / "metadata"

        self.patterns = defaultdict(list)

    def analyze_dataset(self, sample_size: int = 50):
        """Analyze a sample of real designs to extract patterns."""
        print(f"🔍 Analyzing {sample_size} real designs...")

        # Get all metadata files
        metadata_files = sorted(list(self.metadata_dir.glob("*.json")))[:sample_size]

        for i, metadata_file in enumerate(metadata_files):
            if i % 10 == 0:
                print(f"   Processing {i+1}/{len(metadata_files)}...")

            # Load metadata
            with open(metadata_file) as f:
                meta = json.load(f)

            # Load corresponding image
            img_path = self.images_dir / meta["filename"]
            if not img_path.exists():
                continue

            img = Image.open(img_path)

            # Analyze this design
            analysis = self.analyze_single_design(img, meta)

            # Aggregate patterns
            for key, value in analysis.items():
                self.patterns[key].append(value)

        return self.summarize_patterns()

    def analyze_single_design(self, img: Image.Image, meta: Dict) -> Dict:
        """Extract patterns from a single design."""
        width, height = img.size
        aspect_ratio = width / height

        # Convert to numpy for analysis
        img_array = np.array(img.convert('RGB'))

        # Analyze color distribution (simplified)
        dominant_colors = self._extract_dominant_colors(img_array)

        # Detect layout zones via color/brightness analysis
        layout_type = self._detect_layout_structure(img_array)

        # Text vs Visual ratio (heuristic: brightness variance)
        text_regions = self._detect_text_regions(img_array)

        return {
            "aspect_ratio": aspect_ratio,
            "format": meta["v_meta"]["v_Format"],  # 0=poster, 1=social, 2=flyer, 3=banner
            "goal": meta["v_meta"]["v_Goal"],  # 0=inform, 1=persuade, 2=promote, 3=inspire
            "tone": meta["v_meta"]["v_Tone"],
            "grammar": meta["v_grammar"],  # [alignment, contrast, whitespace, hierarchy]
            "dominant_colors": dominant_colors,
            "layout_type": layout_type,
            "text_coverage": text_regions["coverage"],
            "text_position": text_regions["primary_zone"]
        }

    def _extract_dominant_colors(self, img_array: np.ndarray, n_colors: int = 3) -> List[str]:
        """Extract dominant colors (simplified)."""
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)

        # Sample to speed up
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # Simple clustering by quantization
        quantized = (pixels // 64) * 64  # Reduce to 4 levels per channel
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        top_indices = np.argsort(-counts)[:n_colors]

        colors = []
        for idx in top_indices:
            rgb = unique[idx]
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            colors.append(hex_color)

        return colors

    def _detect_layout_structure(self, img_array: np.ndarray) -> str:
        """Detect if layout is split, centered, asymmetric, etc."""
        height, width = img_array.shape[:2]

        # Analyze horizontal splits (top vs bottom)
        top_half = img_array[:height//2, :]
        bottom_half = img_array[height//2:, :]

        top_variance = np.var(top_half)
        bottom_variance = np.var(bottom_half)

        # Analyze vertical splits (left vs right)
        left_half = img_array[:, :width//2]
        right_half = img_array[:, width//2:]

        left_variance = np.var(left_half)
        right_variance = np.var(right_half)

        # Simple heuristics
        if abs(top_variance - bottom_variance) > top_variance * 0.5:
            return "split_horizontal"
        elif abs(left_variance - right_variance) > left_variance * 0.5:
            return "split_vertical"
        elif top_variance < np.var(img_array) * 0.3:
            return "top_heavy"
        elif bottom_variance < np.var(img_array) * 0.3:
            return "bottom_heavy"
        else:
            return "balanced"

    def _detect_text_regions(self, img_array: np.ndarray) -> Dict:
        """Heuristically detect where text is likely placed."""
        height, width = img_array.shape[:2]

        # Convert to grayscale
        gray = np.mean(img_array, axis=2)

        # High-frequency areas often indicate text
        # Split image into zones
        zones = {
            "top": gray[:height//3, :],
            "middle": gray[height//3:2*height//3, :],
            "bottom": gray[2*height//3:, :],
            "left": gray[:, :width//3],
            "right": gray[:, 2*width//3:]
        }

        variances = {zone: np.var(region) for zone, region in zones.items()}
        primary_zone = max(variances, key=variances.get)

        # Text coverage (rough estimate)
        high_contrast_pixels = np.sum(np.abs(gray - np.mean(gray)) > 50)
        coverage = high_contrast_pixels / gray.size

        return {
            "primary_zone": primary_zone,
            "coverage": coverage
        }

    def summarize_patterns(self) -> Dict:
        """Summarize discovered patterns across all designs."""
        print("\n📊 Pattern Summary:")
        print("="*60)

        summary = {}

        # Aspect ratio clustering
        aspect_ratios = self.patterns["aspect_ratio"]
        summary["aspect_ratio_distribution"] = {
            "square (0.9-1.1)": sum(1 for ar in aspect_ratios if 0.9 <= ar <= 1.1),
            "vertical (0.5-0.8)": sum(1 for ar in aspect_ratios if 0.5 <= ar < 0.9),
            "horizontal (1.2-2.5)": sum(1 for ar in aspect_ratios if 1.2 <= ar <= 2.5),
        }

        # Layout types
        layout_counts = Counter(self.patterns["layout_type"])
        summary["layout_types"] = dict(layout_counts.most_common())

        # Text positioning
        text_pos_counts = Counter(self.patterns["text_position"])
        summary["text_positions"] = dict(text_pos_counts.most_common())

        # Grammar scores (average)
        grammar_scores = np.array(self.patterns["grammar"])
        summary["avg_grammar"] = {
            "alignment": float(np.mean(grammar_scores[:, 0])),
            "contrast": float(np.mean(grammar_scores[:, 1])),
            "whitespace": float(np.mean(grammar_scores[:, 2])),
            "hierarchy": float(np.mean(grammar_scores[:, 3]))
        }

        # Print summary
        print(f"\n🎨 Aspect Ratios:")
        for ratio_type, count in summary["aspect_ratio_distribution"].items():
            print(f"   {ratio_type}: {count}")

        print(f"\n📐 Layout Types:")
        for layout, count in summary["layout_types"].items():
            print(f"   {layout}: {count}")

        print(f"\n📝 Text Positions:")
        for pos, count in summary["text_positions"].items():
            print(f"   {pos}: {count}")

        print(f"\n✅ Average Grammar Scores:")
        for metric, score in summary["avg_grammar"].items():
            print(f"   {metric}: {score:.2f}")

        return summary

    def extract_template_from_pattern(self, pattern_name: str) -> Dict:
        """Convert discovered patterns into template definitions."""
        # This will be implemented to generate templates from patterns
        pass

if __name__ == "__main__":
    analyzer = DesignPatternAnalyzer()
    patterns = analyzer.analyze_dataset(sample_size=100)

    # Save patterns for template generation
    with open("data/design_patterns_learned.json", "w") as f:
        json.dump(patterns, f, indent=2)

    print("\n✅ Patterns saved to data/design_patterns_learned.json")
