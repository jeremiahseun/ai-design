"""
Enhanced Design Critic

Critiques designs based on a specific StyleProfile using Gemini Vision.
"""

import json
import os
import time
import random
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from PIL import Image
from src.analysis.style_profiles import StyleProfile

class EnhancedDesignCritic:
    def __init__(self, api_key: Optional[str] = None, style_profile: Optional[StyleProfile] = None):
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.style_profile = style_profile
        self.critique_rubric = self.build_rubric()

    def build_rubric(self) -> Dict[str, Any]:
        """
        Create scoring rubric from style profile
        """
        return {
            "visual_impact": {
                "weight": 0.30,
                "criteria": [
                    "Does this make you pause? (1-10)",
                    "Is there a clear focal point in <0.5s? (1-10)",
                    "Would this stop a scroll? (1-10)"
                ]
            },
            "compositional_excellence": {
                "weight": 0.25,
                "criteria": [
                    "Are style profile composition rules followed? (1-10)",
                    "Is hierarchy clear through size AND position? (1-10)",
                    "Is negative space intentional? (1-10)"
                ]
            },
            "aesthetic_modernity": {
                "weight": 0.20,
                "criteria": [
                    "Does this feel current (2024-2025)? (1-10)",
                    "Are depth techniques applied? (1-10)",
                    "Is it distinctive vs template-like? (1-10)"
                ]
            },
            "technical_execution": {
                "weight": 0.15,
                "criteria": [
                    "Are colors harmonious? (1-10)",
                    "Is typography readable yet interesting? (1-10)",
                    "Are blend modes/effects purposeful? (1-10)"
                ]
            },
            "brand_alignment": {
                "weight": 0.10,
                "criteria": [
                    "Does it match the brief's mood? (1-10)",
                    "Is hierarchy serving content goals? (1-10)"
                ]
            }
        }

    def format_rubric(self) -> str:
        """
        Format rubric for prompt
        """
        return json.dumps(self.critique_rubric, indent=2)

    def critique_with_scoring(self, design_image: Image.Image, brief: str = "") -> Dict[str, Any]:
        """
        Provide scored critique with specific improvements
        """
        if not self.style_profile:
            style_context = "General Modern Design Standards"
        else:
            style_context = self.style_profile.to_prompt_context()

        critique_prompt = f"""
        Evaluate this design against the following rubric:

        You are a senior design director critiquing a junior designer's work.

        DESIGN CONTEXT:
        - Brief: {brief}
        - Style Profile: {self.style_profile.name if self.style_profile else "General Modern"}

        EVALUATE THE IMAGE:
        1. Visual Impact (30%): Is it "scroll-stopping"? Does it pop?
        2. Composition (25%): Is the layout balanced? Is the hierarchy clear?
        3. Modernity (20%): Does it look current (2025+) or dated?
        4. Technical (15%): Is text legible? Are colors harmonious?
        5. Brand/Brief (10%): Does it match the "Christmas/He is Alive" theme?

        CRITICAL INSTRUCTION:
        - Be SPECIFIC. Do not say "improve composition". Say "Move the title up 100px" or "Make the circle larger".
        - If the design is empty or boring, give a low score (< 4.0).
        - If it's good but needs tweaks, give a medium score (6.0-8.0).
        - Only give > 9.0 for perfection.

        RETURN JSON (Strictly follow this):
        {{
            "scores": {{ "visual_impact": 7, "composition": 6, "modernity": 8, "technical": 9, "brand": 7 }},
            "total_score": 7.4,
            "action": "REFINE",  // Options: APPROVE (>8.5), REFINE (6-8.5), REGENERATE (<6), ERROR
            "priority_fixes": [
                "Increase hero text size by 50%",
                "Change background gradient to use warmer colors",
                "Add a glow effect to the central blob"
            ],
            "detailed_feedback": "The design is clean but lacks festive energy..."
        }}
        """

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [critique_prompt, design_image],
                    generation_config={"response_mime_type": "application/json"}
                )
                text = response.text.strip()
                print(f"   🧠 Raw Critique Response: {text[:200]}...") # Debug print

                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text.rsplit("\n", 1)[0]

                critique_json = json.loads(text.strip())
                return self.parse_critique(critique_json)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    print(f"Error critiquing design: {e}")
                    return self._get_fallback_critique()


        return {
            "total_score": 0,
            "action": "ERROR",
            "detailed_feedback": "Failed to generate critique after retries"
        }

    def parse_critique(self, critique_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure critique for action
        """
        # Ensure total_score is present or calculate it
        if "total_score" not in critique_json:
            critique_json["total_score"] = self.calculate_weighted_score(critique_json)

        total_score = critique_json["total_score"]

        if total_score < 7.0:
            action = "REGENERATE"
            priority_fixes = self.extract_top_issues(critique_json, n=3)
        elif total_score < 8.5:
            action = "REFINE"
            priority_fixes = self.extract_top_issues(critique_json, n=2)
        else:
            action = "APPROVE"
            priority_fixes = self.extract_polish_notes(critique_json)

        return {
            "total_score": total_score,
            "action": action,
            "category_scores": critique_json.get("scores", {}),
            "priority_fixes": priority_fixes,
            "detailed_feedback": critique_json.get("feedback", critique_json)
        }

    def calculate_weighted_score(self, critique_json: Dict[str, Any]) -> float:
        # Placeholder implementation if model doesn't return total
        # In a real scenario, we'd parse the category scores
        return critique_json.get("overall_score", 5.0)

    def extract_top_issues(self, critique_json: Dict[str, Any], n: int = 3) -> List[str]:
        # Placeholder extraction logic
        # Ideally, we'd parse the 'Specific fix' fields from the JSON
        fixes = []
        if "actionable_feedback" in critique_json:
             for item in critique_json["actionable_feedback"][:n]:
                 if isinstance(item, dict):
                     fixes.append(item.get("fix", str(item)))
                 else:
                     fixes.append(str(item))
        return fixes if fixes else ["Improve overall composition", "Enhance contrast"]

    def extract_polish_notes(self, critique_json: Dict[str, Any]) -> List[str]:
        return ["Check final alignment", "Verify color consistency"]
