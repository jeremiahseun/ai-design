"""
Template Matcher

Intelligent agent that selects the most appropriate template
based on a design brief or spec.
"""

from typing import List, Dict, Optional, Tuple
from src.templates.template_library import TemplateLibrary
from src.templates.template_schema import Template

class TemplateMatcher:
    def __init__(self, library: TemplateLibrary):
        self.library = library

    def match_template(self, brief: Dict, design_spec: Optional[any] = None) -> Tuple[Optional[Template], float]:
        """
        Analyze brief and return the best template + confidence score (0.0 - 1.0).
        """
        best_template = None
        best_score = -1.0

        # Keywords to look for in brief
        keywords = brief.get("keywords", [])
        description = brief.get("description", "").lower()
        format_type = brief.get("format", "").lower() # e.g. "instagram", "story", "poster"

        for template in self.library.templates.values():
            score = 0.0

            # 1. Category Matching (High Priority)
            if format_type:
                if format_type in template.category.lower() or \
                   any(tag in format_type for tag in template.tags):
                    score += 5.0

            # 2. Tag Matching
            for tag in template.tags:
                if tag in description or tag in keywords:
                    score += 2.0

            # 3. Description Matching (Semantic - simple keyword overlap)
            desc_words = set(description.split())
            template_desc_words = set(template.description.lower().split())
            overlap = len(desc_words.intersection(template_desc_words))
            score += overlap * 0.5

            # Normalize score (rough heuristic)
            # Max expected score ~10-15
            normalized_score = min(score / 10.0, 1.0)

            if normalized_score > best_score:
                best_score = normalized_score
                best_template = template

        if best_score < 0.2:
            return None, 0.0

        return best_template, best_score

    def suggest_alternatives(self, category: str) -> List[Template]:
        """Return random templates from a category for user choice."""
        return self.library.find_templates(category=category)[:3]
