"""
Design Critic

Uses Gemini Vision to critique generated designs based on design principles.
Returns structured feedback for refinement.
"""

import json
import os
from typing import Dict, List, Any, Optional
from PIL import Image
import google.generativeai as genai

class DesignCritic:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def critique_design(self, image: Image.Image, goal: str, tone: str) -> Dict[str, Any]:
        """
        Critique a design image and return structured feedback.
        """
        import time
        import random

        prompt = f"""
        Act as a Senior Art Director. Critique this design based on the goal: "{goal}" and tone: "{tone}".

        Evaluate on 0-10 scale:
        1. Hierarchy (Is the most important info most visible?)
        2. Contrast (Is text readable? Do colors pop?)
        3. Balance (Is the layout stable?)
        4. Alignment (Are elements aligned properly?)

        Output JSON only:
        {{
            "scores": {{
                "hierarchy": float,
                "contrast": float,
                "balance": float,
                "alignment": float
            }},
            "overall_score": float,
            "critique": "One sentence summary",
            "actionable_feedback": [
                {{
                    "issue": "headline_contrast_low",
                    "fix": "increase_tone",
                    "reason": "Headline blends into background"
                }},
                {{
                    "issue": "cluttered_layout",
                    "fix": "reduce_elements",
                    "reason": "Too many text blocks"
                }}
            ]
        }}
        """

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [prompt, image],
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                print(f"⚠️ Error critiquing design (Attempt {attempt+1}/{max_retries}): {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Rate limited. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    break

        return {
            "scores": {"hierarchy": 5, "contrast": 5, "balance": 5, "alignment": 5},
            "overall_score": 5.0,
            "critique": "Error generating critique (Rate limit or API error).",
            "actionable_feedback": []
        }
