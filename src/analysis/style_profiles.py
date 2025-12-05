"""
Style Profile

Represents a learned design aesthetic with actionable rules for composition,
color, typography, and depth.
"""

from typing import List, Dict, Any

class StyleProfile:
    """
    Represents a learned design aesthetic
    """
    def __init__(self, name: str):
        self.name = name
        self.composition_rules: List[str] = []
        self.color_rules: List[str] = []
        self.typography_rules: List[str] = []
        self.depth_rules: List[str] = []
        self.examples: List[str] = []

    def to_prompt_context(self) -> str:
        """
        Convert style profile into prompt context for generation
        """
        return f"""
        STYLE PROFILE: {self.name}

        COMPOSITION RULES:
        {chr(10).join(f"- {rule}" for rule in self.composition_rules)}

        COLOR STRATEGY:
        {chr(10).join(f"- {rule}" for rule in self.color_rules)}

        TYPOGRAPHY APPROACH:
        {chr(10).join(f"- {rule}" for rule in self.typography_rules)}

        DEPTH TECHNIQUES:
        {chr(10).join(f"- {rule}" for rule in self.depth_rules)}
        """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleProfile':
        """
        Create a StyleProfile from a dictionary (e.g. from JSON)
        """
        profile = cls(data.get("name", "Unknown Style"))
        profile.composition_rules = data.get("composition_rules", [])
        profile.color_rules = data.get("color_rules", [])
        profile.typography_rules = data.get("typography_rules", [])
        profile.depth_rules = data.get("depth_rules", [])
        profile.examples = data.get("examples", [])
        return profile
