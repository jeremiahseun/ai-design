"""
Template Library

Manages loading, validating, and retrieving design templates from JSON presets.
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path

from src.templates.template_schema import Template, TemplateLayout, TemplateElement, TemplateTypography

class TemplateLibrary:
    def __init__(self, presets_dir: Optional[str] = None):
        self.templates: Dict[str, Template] = {}

        if presets_dir is None:
            # Default to src/templates/presets relative to this file
            current_file = Path(__file__)
            self.presets_dir = current_file.parent / "presets"
        else:
            self.presets_dir = Path(presets_dir)

        self._load_presets()

    def _load_presets(self):
        """Load all JSON files from the presets directory recursively."""
        if not self.presets_dir.exists():
            print(f"⚠️ Presets directory not found: {self.presets_dir}")
            return

        for path in self.presets_dir.rglob("*.json"):
            self._load_file(path)

    def _load_file(self, path: Path):
        """Parse a single JSON preset file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for t_data in data.get("templates", []):
                template = self._parse_template(t_data)
                self.templates[template.id] = template

        except Exception as e:
            print(f"❌ Error loading template file {path.name}: {e}")

    def _parse_template(self, data: Dict) -> Template:
        """Convert JSON dictionary to Template object."""
        layout_data = data.get("layout", {})

        # Parse Elements
        elements = []
        for el in layout_data.get("elements", []):
            elements.append(TemplateElement(
                type=el["type"],
                name=el.get("name", "element"),
                x_percent=el["x_percent"],
                y_percent=el["y_percent"],
                width_percent=el["width_percent"],
                height_percent=el["height_percent"],
                layer_index=el.get("layer_index", 0),
                color_variable=el.get("color_variable"),
                shape_style=el.get("shape_style"),
                is_placeholder=el.get("is_placeholder", False)
            ))

        # Parse Typography
        typography = []
        for type_el in layout_data.get("typography", []):
            typography.append(TemplateTypography(
                role=type_el["role"],
                text_content_variable=type_el["text_content_variable"],
                font_variable=type_el.get("font_variable", "primary"),
                size_percent=type_el["size_percent"],
                x_percent=type_el["x_percent"],
                y_percent=type_el["y_percent"],
                color_variable=type_el.get("color_variable", "primary"),
                alignment=type_el.get("alignment", "left"),
                max_width_percent=type_el.get("max_width_percent", 0.8)
            ))

        layout = TemplateLayout(
            canvas_ratio=layout_data.get("canvas_ratio", 1.0),
            elements=elements,
            typography=typography,
            background_style=layout_data.get("background_style")
        )

        return Template(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            layout=layout,
            recommended_styles=data.get("recommended_styles", []),
            is_fixed=data.get("is_fixed", False)
        )

    def get_template(self, template_id: str) -> Optional[Template]:
        return self.templates.get(template_id)

    def find_templates(self, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Template]:
        """Find templates matching criteria."""
        matches = []
        for t in self.templates.values():
            if category and t.category != category:
                continue

            if tags:
                # Check if all requested tags are present
                if not all(tag in t.tags for tag in tags):
                    continue

            matches.append(t)
        return matches

    def get_all_categories(self) -> List[str]:
        return list(set(t.category for t in self.templates.values()))
