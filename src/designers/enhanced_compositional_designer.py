"""
Enhanced Compositional Designer

Orchestrates the design process using Style Intelligence, Agents, and Upgraded Generators.
"""

import json
import os
from typing import Dict, Any, Optional
from PIL import Image
import google.generativeai as genai

from src.brain.design_parser import EnhancedDesignParser
from src.brain.design_critic import EnhancedDesignCritic
from src.analysis.style_profiles import StyleProfile
from src.generators.asset_generator import EnhancedAssetGenerator
from src.generators.element_generator import ElementGenerator
from src.generators.text_renderer import EnhancedTextRenderer
from src.generators.text_renderer import EnhancedTextRenderer
from src.layouts.layout_engine import LayoutEngine
from src.templates.template_library import TemplateLibrary
from src.templates.template_matcher import TemplateMatcher

class EnhancedCompositionalDesigner:
    def __init__(self, api_key: Optional[str] = None):
        self.parser = EnhancedDesignParser(api_key=api_key)
        self.critic = EnhancedDesignCritic(api_key=api_key)
        self.layout_engine = LayoutEngine()

        # Generators
        self.asset_generator = EnhancedAssetGenerator()
        self.element_generator = ElementGenerator() # Enhanced methods added to base class
        self.text_renderer = EnhancedTextRenderer()

        # Template System
        self.template_library = TemplateLibrary()
        self.template_matcher = TemplateMatcher(self.template_library)

        self.max_iterations = 3

        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def design_with_iteration(self, brief: str, style_profile: Optional[StyleProfile] = None) -> Image.Image:
        """
        Main design loop with layered generation and critic feedback
        """
        print(f"🎨 Starting design process for: {brief}")

        # Phase 1: Understand & Plan
        parsed = self.parser.parse_with_style_recommendation(brief)

        if not style_profile:
            style_name = parsed.get("recommended_style", "Modern Bold")
            style_profile = StyleProfile(style_name)

        self.critic.style_profile = style_profile

        # LAYER 1: Background Planning & Generation
        print("   🖼️  Layer 1: Generating Background...")
        bg_plan = self.create_background_plan(parsed, style_profile)

        # Generate the actual background image
        width = bg_plan.get("canvas", {}).get("width", 1080)
        height = bg_plan.get("canvas", {}).get("height", 1350)
        bg_spec = bg_plan.get("background", {})
        background_image = self.asset_generator.generate_advanced_background(bg_spec, width, height)

        # LAYER 2: Contextual Layout Planning (Vision)
        print("   👀 Layer 2: Analyzing Background for Layout...")
        layout_plan = self.create_contextual_layout_plan(parsed, style_profile, background_image)

        # Merge plans
        full_plan = {**bg_plan, **layout_plan}

        # Phase 2: Iterative Generation
        current_design = None

        for iteration in range(self.max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")

            # Generate design (using pre-generated background if available to save time, or regenerate)
            # For this workflow, we keep the background constant and iterate on layout
            current_design = self.generate_design(full_plan, background_image)

            # Critique
            critique = self.critic.critique_with_scoring(current_design, brief)
            print(f"   Score: {critique['total_score']:.1f} - Action: {critique['action']}")

            if critique["action"] == "APPROVE":
                print(f"✅ Design approved!")
                return current_design

            elif critique["action"] == "REFINE":
                print(f"   Refining: {critique.get('priority_fixes', [])}")
                full_plan = self.update_plan_for_refinement(
                    full_plan,
                    critique.get("priority_fixes", ["Improve overall design"])
                )

            elif critique["action"] == "ERROR":
                print(f"❌ Critique failed: {critique.get('detailed_feedback', 'Unknown error')}")
                full_plan = self.update_plan_for_regeneration(
                    full_plan,
                    ["Improve visual impact", "Enhance composition", "Increase contrast"]
                )

            else:  # REGENERATE
                print(f"   Regenerating: {critique.get('priority_fixes', [])}")
                full_plan = self.update_plan_for_regeneration(
                    full_plan,
                    critique.get("priority_fixes", ["Improve design quality"])
                )

        print("⚠️ Max iterations reached, returning best attempt")
        return current_design

    def design_from_template(self, brief: str, template_id: Optional[str] = None) -> Image.Image:
        """
        Generate design starting from a template.
        """
        print(f"📄 Starting template-based design for: {brief}")

        parsed = self.parser.parse_with_style_recommendation(brief)

        # Select Template
        template = None
        if template_id:
            template = self.template_library.get_template(template_id)
            if not template:
                print(f"⚠️ Template {template_id} not found, falling back to auto-match.")

        if not template:
            template, score = self.template_matcher.match_template(parsed)
            if template:
                print(f"✅ Auto-matched template: {template.name} (Score: {score:.2f})")
            else:
                print("⚠️ No suitable template found. Falling back to standard generation.")
                return self.design_with_iteration(brief)

        # Convert Template to Plan
        plan = self._convert_template_to_plan(template, parsed)

        # Generate
        print("   🎨 Rendering template...")
        return self.generate_design(plan)

    def _convert_template_to_plan(self, template, parsed_brief) -> Dict[str, Any]:
        """Convert a static template into a dynamic design plan using brief content and design intelligence."""

        # Get or create style profile
        from src.analysis.style_profiles import StyleProfile
        from src.intelligence.design_knowledge import DesignKnowledge

        style_name = parsed_brief.get("recommended_style", template.recommended_styles[0] if template.recommended_styles else "Modern Bold")
        style_profile = StyleProfile(style_name)

        # Initialize design knowledge for intelligent font/color selection
        design_knowledge = DesignKnowledge()

        # Determine canvas dimensions
        width = int(1080 * template.layout.canvas_ratio)
        height = 1080
        if template.layout.canvas_ratio < 1:  # Vertical
            width = 1080
            height = int(1080 / template.layout.canvas_ratio)

        # Generate color palette from brief
        brief_text = parsed_brief.get("description", "") + " " + parsed_brief.get("headline", "")
        color_palette = self._generate_color_palette(brief_text)

        # Create sophisticated background
        background_spec = self._create_template_background(template, color_palette, width, height)

        plan = {
            "canvas": {"width": width, "height": height},
            "background": background_spec,
            "elements": [],
            "typography": {}
        }

        # Map Elements with intelligent color mapping
        for el in template.layout.elements:
            color = self._resolve_color_variable(el.color_variable, color_palette)

            plan["elements"].append({
                "type": el.type,
                "x": int(el.x_percent * width),
                "y": int(el.y_percent * height),
                "width": int(el.width_percent * width),
                "height": int(el.height_percent * height),
                "color": color,
                "opacity": 0.8 if el.is_placeholder else 1.0,
                "layer_index": el.layer_index
            })

        # Map Typography with intelligent font selection
        text_mapping = {
            "title": parsed_brief.get("headline", "HEADLINE"),
            "headline": parsed_brief.get("headline", "HEADLINE"),
            "subtitle": parsed_brief.get("subheading", "Subtitle"),
            "date": "Coming Soon",
            "quote": parsed_brief.get("headline", "Insert Quote Here"),
            "author": parsed_brief.get("subheading", "— Author"),
            "cta": parsed_brief.get("cta", "Learn More"),
            "details": parsed_brief.get("details", "Details here."),
            "category": "NEWS"
        }

        # Select fonts using design knowledge
        hero_font = design_knowledge.find_nearest(brief_text, "font", 1)[0][0] if design_knowledge.model else "Montserrat"

        hero_text = None
        secondary_texts = []

        for type_el in template.layout.typography:
            content = text_mapping.get(type_el.text_content_variable, "Text")
            text_color = self._resolve_color_variable(type_el.color_variable, color_palette)

            text_def = {
                "text": content,
                "content": content,
                "color": text_color,
                "size": int(type_el.size_percent * height),
                "x": int(type_el.x_percent * width),
                "y": int(type_el.y_percent * height),
                "font": hero_font,
                "treatment": "standard",
                "y_position": "custom"
            }

            if type_el.role == "hero":
                hero_text = text_def
            else:
                secondary_texts.append(text_def)

        plan["typography"]["hero"] = hero_text
        plan["typography"]["secondary"] = secondary_texts

        return plan

    def _generate_color_palette(self, brief_text: str) -> Dict[str, str]:
        """Generate a cohesive color palette based on brief."""
        import random

        # Color schemes based on common keywords
        palettes = {
            "tech": {"primary": "#1E3A8A", "secondary": "#3B82F6", "accent": "#06B6D4", "text": "#FFFFFF", "bg": "#0F172A"},
            "luxury": {"primary": "#000000", "secondary": "#D4AF37", "accent": "#FFD700", "text": "#FFFFFF", "bg": "#1A1A1A"},
            "minimal": {"primary": "#000000", "secondary": "#6B7280", "accent": "#F59E0B", "text": "#000000", "bg": "#FFFFFF"},
            "vibrant": {"primary": "#EC4899", "secondary": "#8B5CF6", "accent": "#F59E0B", "text": "#FFFFFF", "bg": "#1F2937"},
            "corporate": {"primary": "#1E40AF", "secondary": "#64748B", "accent": "#0EA5E9", "text": "#FFFFFF", "bg": "#F8FAFC"},
            "creative": {"primary": "#7C3AED", "secondary": "#EC4899", "accent": "#F59E0B", "text": "#FFFFFF", "bg": "#1E1B4B"}
        }

        # Match keywords to palette
        brief_lower = brief_text.lower()
        for theme, palette in palettes.items():
            if theme in brief_lower:
                return palette

        # Default: modern palette
        return {"primary": "#2563EB", "secondary": "#64748B", "accent": "#F59E0B", "text": "#FFFFFF", "bg": "#F8FAFC"}

    def _resolve_color_variable(self, color_var: Optional[str], palette: Dict[str, str]) -> str:
        """Map color variables to actual colors from palette."""
        if not color_var:
            return palette.get("primary", "#000000")

        mapping = {
            "primary": palette.get("primary", "#000000"),
            "secondary": palette.get("secondary", "#666666"),
            "accent": palette.get("accent", "#FF6600"),
            "white": "#FFFFFF",
            "black": "#000000",
            "grey": "#9CA3AF",
            "white_dim": "#E5E7EB",
            "secondary_light": palette.get("secondary", "#E5E7EB") + "40"  # Add transparency
        }

        return mapping.get(color_var, palette.get("primary", "#000000"))

    def _create_template_background(self, template, color_palette: Dict[str, str], width: int, height: int) -> Dict[str, Any]:
        """Create sophisticated backgrounds for templates."""
        template_bg = template.layout.background_style or {}
        bg_type = template_bg.get("type", "solid")

        if bg_type == "gradient":
            return {
                "type": "gradient",
                "gradient": {
                    "stops": [
                        [0, color_palette.get("bg", "#FFFFFF")],
                        [1, color_palette.get("primary", "#000000") + "20"]  # Transparent overlay
                    ],
                    "angle": template_bg.get("angle", 135)
                },
                "noise": {"opacity": 0.03}
            }
        elif bg_type == "dominant_color":
            return {
                "type": "gradient",
                "gradient": {
                    "stops": [
                        [0, color_palette.get("primary", "#2563EB")],
                        [0.5, color_palette.get("secondary", "#64748B")],
                        [1, color_palette.get("accent", "#F59E0B") + "30"]
                    ],
                    "angle": 45
                },
                "noise": {"opacity": 0.05}
            }
        elif bg_type == "image_fill":
            # For image-based templates, use a rich gradient as placeholder
            return {
                "type": "gradient",
                "gradient": {
                    "stops": [
                        [0, color_palette.get("primary", "#1E40AF")],
                        [1, color_palette.get("secondary", "#64748B")]
                    ],
                    "angle": 180
                },
                "noise": {"opacity": 0.08}
            }
        else:
            # Solid with subtle gradient
            return {
                "type": "gradient",
                "gradient": {
                    "stops": [
                        [0, color_palette.get("bg", "#FFFFFF")],
                        [1, color_palette.get("bg", "#F8FAFC")]
                    ],
                    "angle": 135
                },
                "noise": {"opacity": 0.02}
            }


    def create_background_plan(self, parsed: Dict[str, Any], style_profile: StyleProfile) -> Dict[str, Any]:
        """
        Step 1: Plan just the background based on brief and style
        """
        prompt = f"""
        You are an expert background artist.

        BRIEF: {json.dumps(parsed)}
        STYLE: {style_profile.to_prompt_context()}

        TASK: Design a stunning, high-end background.

        JSON RESPONSE:
        {{
            "canvas": {{ "width": 1080, "height": 1350 }},
            "background": {{
                "type": "gradient",
                "gradient": {{
                    "stops": [[0, "#DarkColor"], [1, "#LightColor"]],
                    "angle": 45
                }},
                "noise": {{ "opacity": 0.05 }}
            }}
        }}
        """
        return self._generate_json(prompt, "background plan")

    def create_contextual_layout_plan(self, parsed: Dict[str, Any], style_profile: StyleProfile, bg_image: Image.Image) -> Dict[str, Any]:
        """
        Step 2: Plan layout by LOOKING at the generated background
        """
        prompt = f"""
        You are a layout expert. LOOK at this background image.

        BRIEF: {json.dumps(parsed)}
        STYLE: {style_profile.to_prompt_context()}

        TASK: Place text and elements optimally on THIS background.

        CRITICAL RULES:
        1. Find empty/clean areas for text to ensure readability.
        2. Place geometric elements to balance the background's visual weight.
        3. Ensure high contrast between text color and the specific background area it sits on.
        4. USE ONLY THESE ELEMENT TYPES: "circle", "rect", "pill", "blob", "fragment". Do NOT use "text" or "image" as element types.

        JSON RESPONSE (Strictly follow this):
        {{
            "elements": [
                {{ "type": "blob", "x": 500, "y": 600, "width": 800, "height": 800, "color": "#Color3", "opacity": 0.8 }}
            ],
            "typography": {{
                "hero": {{
                    "text": "MAIN TITLE FROM BRIEF",
                    "treatment": "standard",
                    "color": "#ContrastColor",
                    "font": "Montserrat",
                    "size": 120,
                    "y_position": "top"
                }},
                "secondary": [
                    {{ "content": "Subtitle or Date", "x": 100, "y": 1200, "size": 40, "color": "#ContrastColor" }}
                ]
            }}
        }}
        """
        try:
            # Add retry logic for layout planning
            max_retries = 3
            import time
            import random

            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        [prompt, bg_image],
                        generation_config={"response_mime_type": "application/json"}
                    )
                    return self._clean_json_response(response.text)
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        time.sleep(2 * (2 ** attempt) + random.uniform(0, 1))
                    else:
                        raise e
            return {"elements": [], "typography": {}}
        except Exception as e:
            print(f"Error creating layout plan: {e}")
            return {"elements": [], "typography": {}}

    def _generate_json(self, prompt: str, context: str) -> Dict[str, Any]:
        max_retries = 3
        import time
        import random

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return self._clean_json_response(response.text)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"   ⚠️ Quota hit for {context}, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(2 * (2 ** attempt) + random.uniform(0, 1))
                else:
                    print(f"Error generating {context}: {e}")
                    return {}
        return {}

    def _clean_json_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("\n", 1)[0]
        return json.loads(text.strip())

    # Keep existing update methods but refactor to use _generate_json helper if desired
    # For now, keeping them as is to minimize diff, but we need create_design_plan removed/replaced

    def create_design_plan(self, parsed: Dict[str, Any], style_profile: StyleProfile) -> Dict[str, Any]:
        # Legacy method kept for compatibility if needed, but design_with_iteration now uses layered approach
        return self.create_background_plan(parsed, style_profile)

    def update_plan_for_refinement(self, plan: Dict[str, Any], fixes: list) -> Dict[str, Any]:
        """
        Adjust plan based on critic feedback
        """
        update_prompt = f"""
        CURRENT PLAN:
        {json.dumps(plan)}

        REQUIRED FIXES:
        {chr(10).join(f"- {fix}" for fix in fixes)}

        Update the plan to address these fixes while keeping successful elements.
        Return updated JSON plan.
        """
        return self._generate_json(update_prompt, "refinement plan")

    def update_plan_for_regeneration(self, plan: Dict[str, Any], fixes: list) -> Dict[str, Any]:
        """
        Major plan revision for failed designs
        """
        regen_prompt = f"""
        FAILED PLAN:
        {json.dumps(plan)}

        CRITICAL ISSUES:
        {chr(10).join(f"- {fix}" for fix in fixes)}

        Create a NEW plan that addresses these issues and takes a different approach.
        Return completely new JSON plan.
        """
        return self._generate_json(regen_prompt, "regeneration plan")

    def generate_design(self, plan: Dict[str, Any], pre_generated_bg: Optional[Image.Image] = None) -> Image.Image:
        """
        Execute the design plan using generators
        """
        # 1. Canvas Dimensions
        width = plan.get("canvas", {}).get("width", 1080)
        height = plan.get("canvas", {}).get("height", 1350)

        # 2. Background
        if pre_generated_bg:
            canvas = pre_generated_bg.copy()
        else:
            bg_spec = plan.get("background", {})
            canvas = self.asset_generator.generate_advanced_background(bg_spec, width, height)

        # 3. Elements
        elements = plan.get("elements", [])
        print(f"   🎨 Generating {len(elements)} elements...")
        for i, el in enumerate(elements):
            shape_type = el.get("type", "circle")
            w = el.get("width", 100)
            h = el.get("height", 100)
            color = el.get("color", "#CCCCCC")
            x = el.get("x", 0)
            y = el.get("y", 0)

            print(f"      - Element {i+1}: {shape_type} at ({x}, {y}) size {w}x{h} color {color}")

            if shape_type == "blob":
                img = self.element_generator.generate_organic_blob(w, h, color)
            elif shape_type == "fragment":
                img = self.element_generator.generate_fragmented_shape(w, h, color)
            else:
                img = self.element_generator.generate_shape(shape_type, w, h, color, opacity=el.get("opacity", 1.0))

            # Ensure we're pasting with mask if image has alpha
            if img.mode == 'RGBA':
                canvas.paste(img, (x, y), img)
            else:
                canvas.paste(img, (x, y))

        # 4. Typography
        typography = plan.get("typography", {})

        # Hero
        hero = typography.get("hero", {})
        if hero:
            print(f"   📝 Rendering Hero Text: '{hero.get('text', '')}'")
            canvas = self.text_renderer.render_hero_typography(canvas, hero)
        else:
            print("   ⚠️ No Hero text in plan")

        # Secondary
        secondary = typography.get("secondary", [])
        if isinstance(secondary, dict): secondary = [secondary]

        if secondary:
            print(f"   📝 Rendering {len(secondary)} secondary text elements")

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)

        # Track occupied areas to prevent overlap
        occupied_areas = []
        # Add hero text area (approximate)
        if hero:
            # Assume hero is at top
            occupied_areas.append((0, 0, width, height * 0.3))

        for text_el in secondary:
            content = text_el.get("content", "")
            if not content: continue

            x = text_el.get("x", 50)
            y = text_el.get("y", 50)
            size = text_el.get("size", 24)
            color = text_el.get("color", "#000000")

            # Simple rendering for secondary text
            try:
                # Try to use a standard font
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
            except:
                font = ImageFont.load_default()

            # Check for collision and adjust Y if needed
            text_bbox = draw.textbbox((x, y), content, font=font)
            text_h = text_bbox[3] - text_bbox[1]

            # Simple collision avoidance: if overlaps with previous, move down
            for area in occupied_areas:
                if (x < area[2] and x + (text_bbox[2]-text_bbox[0]) > area[0] and
                    y < area[3] and y + text_h > area[1]):
                    # Collision! Move down
                    y = area[3] + 20 # 20px padding

            # Update occupied areas
            occupied_areas.append((x, y, x + (text_bbox[2]-text_bbox[0]), y + text_h))

            draw.text((x, y), content, font=font, fill=color)

        return canvas

    def _get_fallback_plan(self) -> Dict[str, Any]:
        return {
            "canvas": {"width": 1080, "height": 1350},
            "background": {"gradient": {"stops": [[0, "#FFFFFF"], [1, "#EEEEEE"]]}},
            "elements": [],
            "typography": {"hero": {"text": "DESIGN FAILED", "color": "#000000"}}
        }
