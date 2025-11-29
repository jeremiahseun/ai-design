"""
Universal Designer

Combines the Design Agent with the Compositional AI pipeline to create designs from
natural language prompts.
"""

from pathlib import Path
from typing import Optional, List
from PIL import Image

from src.agents.design_agent import UniversalDesignAgent
from src.agents.design_spec import DesignSpec
from src.designers.compositional_designer import CompositionalDesigner

class UniversalDesigner:
    """
    End-to-end designer that takes natural language and outputs designs
    """

    def __init__(self,
                 device: str = "mps",
                 api_key: Optional[str] = None):
        """
        Initialize the universal designer

        Args:
            device: Device for SD (mps/cuda/cpu)
            api_key: Google API key for the agent
        """
        print("🚀 Initializing Universal Designer...")

        # Initialize components
        self.agent = UniversalDesignAgent(api_key=api_key)
        self.designer = CompositionalDesigner(device=device)
        self.device = device

        print("✅ Universal Designer ready!")

    def create_design(self,
                     prompt: str,
                     variants: int = 1,
                     logo_path: Optional[str] = None) -> tuple[DesignSpec, List[Image.Image]]:
        """
        Create design(s) from a natural language prompt

        Args:
            prompt: Natural language design request
            variants: Number of design variations to generate
            logo_path: Override logo path (if not in prompt)

        Returns:
            (DesignSpec, List of PIL Images)
        """
        # 1. Interpret prompt
        spec = self.agent.interpret_prompt(prompt)

        # Override logo if provided
        if logo_path:
            spec.logo_path = logo_path

        # 2. Generate designs using the compositional pipeline
        designs = []
        print(f"\n🎨 Generating {variants} design variant(s)...")

        for i in range(variants):
            design = self.designer.create_design(spec)
            designs.append(design)
            print(f"  ✅ Variant {i+1} complete")

        return spec, designs
