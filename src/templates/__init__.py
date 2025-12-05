"""
Template System

Provides structural and stylistic starting points for AI design generation.
"""

from src.templates.template_schema import Template, TemplateLayout, TemplateElement, TemplateTypography
from src.templates.template_library import TemplateLibrary
from src.templates.template_matcher import TemplateMatcher

__all__ = [
    'Template',
    'TemplateLayout',
    'TemplateElement',
    'TemplateTypography',
    'TemplateLibrary',
    'TemplateMatcher'
]
