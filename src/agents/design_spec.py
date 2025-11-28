"""
Design Specification Data Class

Represents the structured output from the Universal Design Agent.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json

@dataclass
class DesignContent:
    """Text content for the design"""
    headline: str
    subheading: str = ""
    details: str = ""

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
        goals = {0: "Inform", 1: "Persuade", 2: "Entertain", 3: "Inspire"}
        return goals.get(self.goal, "Unknown")

    @property
    def format_name(self) -> str:
        """Human-readable format name"""
        formats = {0: "Poster", 1: "Social Media", 2: "Flyer", 3: "Banner"}
        return formats.get(self.format, "Unknown")

    @property
    def tone_description(self) -> str:
        """Human-readable tone description"""
        if self.tone < 0.33:
            return "Calm & Minimal"
        elif self.tone < 0.67:
            return "Professional & Balanced"
        else:
            return "Energetic & Bold"

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
            data['content'] = DesignContent(**data['content'])
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
  Headline: "{self.content.headline}"
  Subheading: "{self.content.subheading}"
  Details: "{self.content.details}"
  Style: {self.style_prompt}
  Logo: {self.logo_path if self.logo_path else "None"}
"""
