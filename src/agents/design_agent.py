"""
Universal Design Agent

Uses an LLM to interpret natural language design requests and convert them
into structured DesignSpec objects.
"""

import os
import json
from typing import Optional
import google.generativeai as genai
from .design_spec import DesignSpec, DesignContent

class UniversalDesignAgent:
    """
    LLM-powered agent that interprets design requests
    """

    SYSTEM_PROMPT = """You are an expert graphic design consultant. Your job is to analyze design requests and extract structured information.

Given a user's design request, extract:
1. **Design Goal** (choose ONE):
   - 0 = Inform (educational, infographics, data presentation)
   - 1 = Persuade (marketing, sales, call-to-action)
   - 2 = Entertain (fun, playful, engaging)
   - 3 = Inspire (motivational, emotional, artistic)

2. **Format** (choose ONE):
   - 0 = Poster (vertical, print quality)
   - 1 = Social Media (square, digital)
   - 2 = Flyer (promotional, handout)
   - 3 = Banner (horizontal, header)

3. **Tone** (0.0 to 1.0):
   - 0.0-0.3 = Calm/Minimal (soft colors, whitespace, elegant)
   - 0.4-0.6 = Professional/Balanced (corporate, trustworthy)
   - 0.7-1.0 = Energetic/Bold (vibrant, high contrast, loud)

4. **Visual Style**: Detailed description of colors, mood, themes for image generation

5. **Text Content**:
   - Headline (main message)
   - Subheading (supporting text)
   - Details (date, location, specifics)

6. **Logo Path**: If user mentions a logo file path, include it

Return ONLY a valid JSON object with this exact structure:
{
  "goal": <int 0-3>,
  "format": <int 0-3>,
  "tone": <float 0.0-1.0>,
  "style_prompt": "<detailed visual style>",
  "content": {
    "headline": "<main text>",
    "subheading": "<secondary text>",
    "details": "<additional info>"
  },
  "logo_path": "<path or null>",
  "constraints": {}
}

Be intelligent about defaults. If format isn't specified, infer from context. If tone isn't clear, use 0.5.
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the agent

        Args:
            api_key: Google API key (or use GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var or pass api_key parameter.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent JSON
                "response_mime_type": "application/json"
            }
        )

    def interpret_prompt(self, user_prompt: str) -> DesignSpec:
        """
        Interpret a natural language design request

        Args:
            user_prompt: User's design request in plain English

        Returns:
            DesignSpec object
        """
        print(f"\n🤖 Agent analyzing: \"{user_prompt[:80]}...\"")

        try:
            # Build full prompt with system instructions
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nUser Request: {user_prompt}"

            # Call Gemini
            response = self.model.generate_content(full_prompt)

            # Parse JSON response
            spec_dict = json.loads(response.text)

            # Convert to DesignSpec
            spec = DesignSpec.from_dict(spec_dict)

            print(f"✅ Agent interpretation complete!")
            print(f"   Goal: {spec.goal_name} | Format: {spec.format_name} | Tone: {spec.tone_description}")

            return spec

        except Exception as e:
            print(f"❌ Agent error: {e}")
            raise RuntimeError(f"Failed to interpret prompt: {e}")

    def refine_spec(self, current_spec: DesignSpec, refinement: str) -> DesignSpec:
        """
        Refine an existing spec based on user feedback

        Args:
            current_spec: Current DesignSpec
            refinement: User's refinement request (e.g., "make it more energetic")

        Returns:
            Updated DesignSpec
        """
        print(f"\n🔄 Refining design: \"{refinement}\"")

        refine_prompt = f"""Given this current design specification:
{current_spec.to_json()}

The user wants this change: "{refinement}"

Return the UPDATED specification as JSON. Only change what's needed based on the refinement request."""

        return self.interpret_prompt(refine_prompt)
