"""
Template Schema

Defines the data structures for the design template system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TemplateElement:
    type: str  # "circle", "rect", "pill", "blob", "fragment", "image"
    name: str # Descriptive name e.g., "hero_bg_shape"
    x_percent: float # 0.0 to 1.0 relative to canvas width
    y_percent: float # 0.0 to 1.0 relative to canvas height
    width_percent: float # 0.0 to 1.0
    height_percent: float # 0.0 to 1.0
    layer_index: int = 0
    color_variable: Optional[str] = None # e.g., "primary", "secondary", "accent"
    shape_style: Optional[Dict[str, Any]] = None # specific style props (radius, etc)
    is_placeholder: bool = False # If true, AI should replace/fill this

@dataclass
class TemplateTypography:
    role: str # "hero", "subheading", "body", "caption"
    text_content_variable: str # "headline", "date", "cta" - maps to brief
    font_variable: str # "primary_font", "display_font"
    size_percent: float # Approx height relative to canvas
    x_percent: float
    y_percent: float
    color_variable: str
    alignment: str = "left" # "left", "center", "right"
    max_width_percent: float = 0.8

@dataclass
class TemplateLayout:
    canvas_ratio: float # width / height
    elements: List[TemplateElement] = field(default_factory=list)
    typography: List[TemplateTypography] = field(default_factory=list)
    background_style: Optional[Dict[str, Any]] = None # Gradient or solid definition

@dataclass
class Template:
    id: str
    name: str
    description: str
    category: str # "social_media", "corporate", "event", etc.
    tags: List[str]
    layout: TemplateLayout
    recommended_styles: List[str] = field(default_factory=list) # e.g. ["Modern", "Minimal"]
    is_fixed: bool = False # If true, layout structure is rigid. If false, AI can shift things.
