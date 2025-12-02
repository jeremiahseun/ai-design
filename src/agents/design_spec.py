"""
Design Specification Data Class

Represents the structured output from the Universal Design Agent.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json

@dataclass
class DesignElement:
    """A single element in the design content."""
    text: str
    role: str  # "primary", "secondary", "tertiary"
    style_hint: Optional[str] = None

@dataclass
class DesignContent:
    """
    Content for the design.
    Now supports dynamic list of elements.
    """
    elements: List[DesignElement]

    # Backward compatibility properties
    @property
    def headline(self) -> str:
        primary = [e.text for e in self.elements if e.role == "primary"]
        return primary[0] if primary else ""

    @property
    def subheading(self) -> str:
        secondary = [e.text for e in self.elements if e.role == "secondary"]
        return secondary[0] if secondary else ""

    @property
    def details(self) -> str:
        tertiary = [e.text for e in self.elements if e.role == "tertiary"]
        return "\n".join(tertiary) if tertiary else ""

@dataclass
class DesignSpec:
    """
    Complete specification for a design

    Attributes:
        goal: Design goal (0=Inform, 1=Persuade, 2=Entertain, 3=Inspire)
        format: Design format (0=Poster, 1=Social, 2=Flyer, 3=Banner)
        tone: Emotional tone (0.0=Calm/Minimal, 1.0=Energetic/Bold)
        style_prompt: Visual style description for SD
        content: Text content
        logo_path: Path to logo file (optional)
        constraints: Additional requirements (optional)
    """
    goal: int
    format: int
    tone: float
    style_prompt: str
    content: DesignContent
    logo_path: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None

    @property
    def goal_name(self) -> str:
        """Human-readable goal name"""
        return ["Inform", "Persuade", "Entertain", "Inspire"][self.goal]

    @property
    def format_name(self) -> str:
        """Human-readable format name"""
        return ["Poster", "Social Media", "Flyer", "Banner"][self.format]

    @property
    def tone_description(self) -> str:
        """Human-readable tone description"""
        if self.tone < 0.4: return "Calm & Minimal"
        if self.tone > 0.6: return "Energetic & Bold"
        return "Balanced & Professional"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DesignSpec':
        """Create from dictionary"""
        # Handle nested DesignContent
        if isinstance(data.get('content'), dict):
            content_data = data['content']
            # Handle elements list
            if 'elements' in content_data:
                elements = [DesignElement(**e) for e in content_data['elements']]
                data['content'] = DesignContent(elements=elements)
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'DesignSpec':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"""Design Specification:
  Goal: {self.goal_name}
  Format: {self.format_name}
  Tone: {self.tone_description} ({self.tone:.2f})
  Elements: {len(self.content.elements)}
  Style: {self.style_prompt}
"""
