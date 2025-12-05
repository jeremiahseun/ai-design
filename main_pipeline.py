"""
Advanced Design Pipeline

Integrates all layers:
1. Style Intelligence (Analyzer)
2. Enhanced Brain (Parser + Critic)
3. Refined Heart (Designer)
4. Upgraded Hands (Generators)
"""

import os
import sys
from typing import Optional, List
from PIL import Image

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.style_analyzer import StyleAnalyzer
from src.brain.design_parser import EnhancedDesignParser
from src.brain.design_critic import EnhancedDesignCritic
from src.designers.enhanced_compositional_designer import EnhancedCompositionalDesigner
from src.analysis.style_profiles import StyleProfile

class AdvancedDesignPipeline:
    def __init__(self, api_key: Optional[str] = None):
        # Initialize all components
        self.style_analyzer = StyleAnalyzer(api_key=api_key)
        self.designer = EnhancedCompositionalDesigner(api_key=api_key)
        # Note: Parser and Critic are initialized inside Designer,
        # but we can also access them if needed.

    def run(self, brief: str, pinterest_refs: List[str] = None, output_path: str = "output/final_design.png") -> Image.Image:
        """
        Complete design generation pipeline
        """
        print("🚀 Starting Advanced Design Pipeline")
        print(f"📝 Brief: {brief}")

        # Step 1: Analyze Pinterest references if provided
        style_profile = None
        if pinterest_refs:
            print("📸 Analyzing reference images...")
            style_data = self.style_analyzer.analyze_reference_batch(pinterest_refs)
            if style_data:
                style_profile = StyleProfile.from_dict(style_data)
                print(f"✨ Extracted Style: {style_profile.name}")
        else:
            # Use default style profile or let parser recommend
            print("ℹ️  No references provided. Using AI-recommended style.")

        # Step 2: Generate design with iteration
        # The designer handles parsing, planning, generation, and critique loop
        final_design = self.designer.design_with_iteration(brief, style_profile)

        # Step 3: Export
        if final_design:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_design.save(output_path)
            print(f"✅ Design saved to: {output_path}")
        else:
            print("❌ Design generation failed.")

        return final_design

if __name__ == "__main__":
    # Example Usage
    import argparse

    parser = argparse.ArgumentParser(description="Run Advanced Design Pipeline")
    parser.add_argument("--brief", type=str, required=True, help="Design brief")
    parser.add_argument("--refs", nargs="+", help="List of Pinterest image paths")
    parser.add_argument("--output", type=str, default="output/final_design.png", help="Output path")

    args = parser.parse_args()

    pipeline = AdvancedDesignPipeline()
    pipeline.run(args.brief, args.refs, args.output)
