"""
Layout Selector

A deterministic reasoning engine that selects the best layout
based on the DesignSpec and content metrics.
"""

from typing import Dict, List
from src.agents.design_spec import DesignSpec
from src.layouts.layout_engine import LayoutEngine

class LayoutSelector:
    def __init__(self):
        self.layouts = [
            "split_horizontal",
            "split_vertical",
            "central_hero",
            "typographic_bold",
            "modern_clean"
        ]

    def select_layout(self, spec: DesignSpec) -> str:
        """
        Select the best layout name based on the spec.
        """
        scores = {layout: 0 for layout in self.layouts}

        # 1. Format Constraints (Hard Filters)
        if spec.format == 3: # Banner (Horizontal)
            scores["split_horizontal"] += 10
            scores["modern_clean"] += 5
            scores["split_vertical"] -= 100 # Impossible for banners
            scores["typographic_bold"] -= 5

        elif spec.format == 0: # Poster (Vertical)
            scores["split_vertical"] += 5
            scores["typographic_bold"] += 5
            scores["central_hero"] += 5
            scores["modern_clean"] += 5
            scores["split_horizontal"] -= 5 # Less common for posters

        elif spec.format == 1: # Social (Square)
            scores["central_hero"] += 8
            scores["modern_clean"] += 5
            scores["typographic_bold"] += 5

        # 2. Goal-Based Scoring
        if spec.goal == 0: # Inform (Text heavy)
            scores["typographic_bold"] += 5
            scores["split_vertical"] += 3
            scores["central_hero"] -= 2 # Image focus might hide text

        elif spec.goal == 3: # Inspire (Image heavy)
            scores["central_hero"] += 8
            scores["modern_clean"] += 5
            scores["typographic_bold"] -= 5

        # 3. Content Metrics (The "Reasoning" Part)
        total_text_len = len(spec.content.headline or "") + \
                         len(spec.content.subheading or "") + \
                         len(spec.content.details or "")

        if total_text_len > 150:
            # Very text heavy -> Needs dedicated text zone
            scores["split_vertical"] += 5
            scores["split_horizontal"] += 5
            scores["central_hero"] -= 10 # Overlay will be messy

        elif total_text_len < 50:
            # Minimal text -> Can be artistic
            scores["central_hero"] += 5
            scores["modern_clean"] += 5

        # 4. Tone Adjustments
        if spec.tone > 0.7: # Energetic/Bold
            scores["typographic_bold"] += 3
        elif spec.tone < 0.3: # Calm/Minimal
            scores["modern_clean"] += 5

        # Select winner
        best_layout = max(scores, key=scores.get)
        print(f"🧠 Layout Reasoning Scores: {scores}")
        print(f"✅ Selected: {best_layout}")

        return best_layout
