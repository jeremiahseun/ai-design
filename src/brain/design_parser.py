"""
Enhanced Design Parser

Parses design briefs and recommends appropriate style profiles using Gemini.
"""

import json
import os
import time
import random
from typing import Dict, Any, Optional
import google.generativeai as genai
from src.analysis.style_analyzer import StyleAnalyzer

class EnhancedDesignParser:
    def __init__(self, api_key: Optional[str] = None, style_analyzer: Optional[StyleAnalyzer] = None):
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.style_analyzer = style_analyzer or StyleAnalyzer(api_key=api_key)

    def parse_with_style_recommendation(self, brief: str) -> Dict[str, Any]:
        """
        Parse brief and recommend appropriate style profile
        """
        analysis_prompt = f"""
        Design Brief: {brief}

        TASK 1 - EXTRACT REQUIREMENTS:
        - Content elements (text, images, data)
        - Hierarchy (what's most important)
        - Constraints (dimensions, brand rules)
        - Mood keywords (professional, playful, bold, minimal)

        TASK 2 - RECOMMEND STYLE APPROACH:
        Available styles: ["Modern Bold", "Minimalist Clean", "Tech Futuristic", "Organic Nature"]

        Based on the brief's mood and context, which style profile fits best?
        Why? What adaptations should we make?

        TASK 3 - DEFINE SUCCESS METRICS:
        For this specific brief, the design succeeds if:
        - [Visual impact criterion]
        - [Functional criterion]
        - [Brand alignment criterion]

        Return structured JSON with: requirements, recommended_style, success_criteria
        """

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    analysis_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                text = response.text.strip()
                # Clean markdown code blocks if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text.rsplit("\n", 1)[0]
                text = text.strip()

                return json.loads(text)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    print(f"Error parsing brief: {e}")
                    # Fallback
                    return {
                        "requirements": {"mood": "neutral"},
                        "recommended_style": "Modern Bold",
                        "success_criteria": []
                    }

        return {
            "requirements": {"mood": "neutral"},
            "recommended_style": "Modern Bold",
            "success_criteria": []
        }
