"""
Design Parser

Uses Gemini Flash to parse natural language design requests into
structured DesignSpec objects with dynamic content elements.
"""

import json
import os
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from src.agents.design_spec import DesignSpec, DesignContent, DesignElement

class DesignParser:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def parse_request(self, prompt: str) -> DesignSpec:
        """
        Parse a natural language prompt into a DesignSpec.
        """
        import time
        import random

        system_prompt = """
        You are an expert Design Director. Your job is to parse a user's design request into a structured JSON specification.

        Output Format (JSON):
        {
            "goal": "inform" | "persuade" | "entertain" | "inspire",
            "format": "poster" | "social" | "flyer" | "banner",
            "tone": float (0.0 = calm/minimal, 1.0 = energetic/bold),
            "style_keywords": "comma, separated, keywords, for, visual, style",
            "elements": [
                {"text": "Text content", "role": "primary" | "secondary" | "tertiary", "style_hint": "optional style note"}
            ]
        }

        Rules:
        1. Extract all text content mentioned by the user.
        2. Assign roles based on hierarchy (Primary = Headline, Secondary = Subhead, Tertiary = Details).
        3. Infer tone and goal from the context if not explicit.
        4. Generate relevant style keywords for a vector search (e.g. "luxury, serif, gold" or "tech, neon, dark").
        """

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    f"{system_prompt}\n\nUser Request: {prompt}",
                    generation_config={"response_mime_type": "application/json"}
                )

                data = json.loads(response.text)

                # Map strings to enums/ints
                goal_map = {"inform": 0, "persuade": 1, "entertain": 2, "inspire": 3}
                format_map = {"poster": 0, "social": 1, "flyer": 2, "banner": 3}

                elements = [
                    DesignElement(e["text"], e["role"], e.get("style_hint"))
                    for e in data.get("elements", [])
                ]

                return DesignSpec(
                    goal=goal_map.get(data.get("goal", "inform"), 0),
                    format=format_map.get(data.get("format", "poster"), 0),
                    tone=float(data.get("tone", 0.5)),
                    style_prompt=data.get("style_keywords", ""),
                    content=DesignContent(elements=elements)
                )

            except Exception as e:
                print(f"⚠️ Error parsing request (Attempt {attempt+1}/{max_retries}): {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Rate limited. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    break # Don't retry other errors

        print("❌ Failed to parse request after retries.")
        # Fallback spec
        return DesignSpec(
            goal=0, format=0, tone=0.5, style_prompt="clean",
            content=DesignContent(elements=[
                DesignElement("Design Generation Failed", "primary"),
                DesignElement("Please check API quota or try again later.", "secondary")
            ])
        )
