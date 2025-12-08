
"""
Pattern Extractor V3.
Exemplar-based extraction. Selects the single best/most representative design from each cluster
to serve as the template, avoiding "mushy" averages.
"""

import json
import os
import numpy as np
import statistics
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

@dataclass
class LearnedTemplate:
    name: str
    layout_type: str
    description: str
    canvas_ratio: float
    elements: List[Dict]
    typography: List[Dict]
    background: Dict
    confidence_score: float
    source_count: int
    exemplar_source: str # Filename of the source design

class PatternExtractor:
    """
    Extracts design patterns by identifying the best 'Exemplar' from clusters.
    """

    def __init__(self, analysis_file: str = "data/design_patterns_raw.json"):
        self.analysis_file = analysis_file
        self.raw_data = []
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                self.raw_data = json.load(f)

    def extract_patterns(self, min_cluster_size: int = 4) -> List[LearnedTemplate]:
        if not self.raw_data:
            print("No data to analyze.")
            return []

        print(f"Extracting V3 patterns from {len(self.raw_data)} designs...")

        # 1. Cluster Designs
        clusters = self._cluster_designs()

        learned_templates = []

        for key, group in clusters.items():
            if len(group) < min_cluster_size:
                continue

            print(f"Processing Cluster: {key} (N={len(group)})")
            template = self._select_exemplar(key, group)
            if template:
                learned_templates.append(template)

        return learned_templates

    def _cluster_designs(self) -> Dict[str, List[Dict]]:
        """
        Groups designs by structural similarity.
        """
        clusters = defaultdict(list)

        for record in self.raw_data:
            analysis = record.get("analysis", {})
            # Safety check: insure analysis is a dict
            if not isinstance(analysis, dict):
                continue

            if "error" in analysis: continue

            # Feature 1: Layout Structure Safety Check
            layout_struct = analysis.get("layout_structure", {})
            if not isinstance(layout_struct, dict):
                 layout_struct = {}

            l_types = layout_struct.get("type", ["unknown"])
            if isinstance(l_types, str): l_types = [l_types]
            layout_type = l_types[0] if l_types else "unknown"

            # Feature 2: Image Dominance
            images = analysis.get("images_and_assets", [])
            img_count = len([i for i in images if i.get("type") == "photo_main"])
            img_mode = "text_only" if img_count == 0 else "single_hero" if img_count == 1 else "multi_image"

            # Feature 3: Primary Alignment
            typography = analysis.get("typography_detailed", [])
            headline = next((t for t in typography if t.get("role") == "headline"), None)
            align = headline.get("alignment_inner", "center") if headline else "center"

            # Composite Key
            key = f"{layout_type}_{img_mode}_{align}"
            clusters[key].append(analysis)

        return clusters

    def _select_exemplar(self, key: str, group: List[Dict]) -> LearnedTemplate:
        """
        Selects the single best design from the group to serve as the template.
        """
        best_score = -1
        exemplar = None

        for design in group:
            score = self._score_design(design)
            if score > best_score:
                best_score = score
                exemplar = design

        if not exemplar:
            return None

        # Convert the exemplar directly into a template
        description = exemplar.get("layout_structure", {}).get("description", "Learned layout.")

        # Elements
        final_elements = []
        for i, el in enumerate(exemplar.get("images_and_assets", [])):
            if el.get("type") in ["photo_main", "photo"]:
                pos = el.get("position", {})
                size = el.get("size", {})
                final_elements.append({
                    "type": "image",
                    "x_percent": pos.get("x", 0.5),
                    "y_percent": pos.get("y", 0.5),
                    "width_percent": size.get("width", 0.5),
                    "height_percent": size.get("height", 0.5),
                    "layer_index": 0
                })

        # Typography
        final_typography = []
        used_roles = set()

        for t in exemplar.get("typography_detailed", []):
            role = t.get("role", "body")
            # Uniquify role if multiple
            base_role = role
            counter = 1
            while role in used_roles:
                role = f"{base_role}_{counter}"
                counter += 1
            used_roles.add(role)

            pos = t.get("position_exact", {})
            dims = t.get("box_dimensions", {})

            final_typography.append({
                "role": base_role, # Keep base role for mapping
                "id": role,        # Unique ID
                "x_percent": pos.get("x", 0.5),
                "y_percent": pos.get("y", 0.5),
                "width_percent": dims.get("width", 0.5),
                "size_percent": dims.get("height", 0.05),
                "alignment": t.get("alignment_inner", "left"),
                "font_variable": "heading" if base_role == "headline" else "body",
                "font_size_variable": "xl" if base_role == "headline" else "md"
            })

        # Background
        bg_type = exemplar.get("background", {}).get("type", ["solid"])
        if isinstance(bg_type, list): bg_type = bg_type[0] if bg_type else "solid"

        return LearnedTemplate(
            name=f"Learned {key.replace('_', ' ').title()}",
            layout_type=key.split('_')[0],
            description=f"Exemplar extracted from {len(group)} designs. Original: {description}",
            canvas_ratio=1.0, # detailed analysis usually implies square for now
            elements=final_elements,
            typography=final_typography,
            background={"type": bg_type},
            confidence_score=len(group) / 200, # normalized
            source_count=len(group),
            exemplar_source="derived"
        )

    def _score_design(self, design: Dict) -> float:
        """
        Scores a design for 'validity' as a template.
        Higher score = better structure, more complete metadata.
        """
        score = 0

        # Reward: Has headline
        typography = design.get("typography_detailed", [])
        if any(t.get("role") == "headline" for t in typography):
            score += 10

        # Reward: Has clear alignment
        # (Already filtered by cluster, but good to check)

        # Reward: Has images if it's an image layout
        images = design.get("images_and_assets", [])
        if images:
            score += 5

        # Penalty: Too chaotic (too many text blocks)
        if len(typography) > 6:
            score -= 5

        # Penalty: Overlapping elements (naive check not strictly needed if we trust source)

        return score

    def save_templates(self, templates: List[LearnedTemplate], output_file: str):
        output_data = {
            "category": "Learned Patterns V3 (Exemplar)",
            "templates": []
        }
        for t in templates:
            system_template = {
                "id": t.name.lower().replace(" ", "_"),
                "name": t.name,
                "description": t.description,
                "layout": {
                    "canvas_ratio": t.canvas_ratio,
                    "elements": [
                        {
                            "type": e["type"],
                            "x_percent": e["x_percent"],
                            "y_percent": e["y_percent"],
                            "width_percent": e["width_percent"],
                            "height_percent": e["height_percent"],
                            "layer_index": 0,
                            "is_placeholder": True
                        } for e in t.elements
                    ],
                    "typography": [
                        {
                            "id": tp['id'],
                            "role": tp['role'],
                            "text_content_variable": tp['role'],
                            "font_variable": tp['font_variable'],
                            "size_percent": tp['size_percent'],
                            "x_percent": tp['x_percent'],
                            "y_percent": tp['y_percent'],
                            "width_percent": tp['width_percent'],
                            "alignment": tp['alignment'],
                            "font_size_variable": tp['font_size_variable']
                        } for tp in t.typography
                    ],
                    "background": t.background
                },
                "recommended_styles": ["Modern", "Learned"],
                "tags": ["learned", "v3", t.layout_type]
            }
            output_data["templates"].append(system_template)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(templates)} exemplar templates to {output_file}")

if __name__ == "__main__":
    extractor = PatternExtractor()
    patterns = extractor.extract_patterns(min_cluster_size=5) # High threshold
    extractor.save_templates(patterns, "src/templates/presets/learned/auto_generated_v3.json")
