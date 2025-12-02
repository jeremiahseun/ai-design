"""
Iterative Designer

Orchestrates the loop: Generate -> Critique -> Refine -> Repeat.
"""

from PIL import Image
from src.agents.design_spec import DesignSpec
from src.designers.compositional_designer import CompositionalDesigner
from src.intelligence.design_critic import DesignCritic
from src.intelligence.design_refiner import DesignRefiner

class IterativeDesigner:
    def __init__(self, device: str = "cpu"):
        self.generator = CompositionalDesigner(device=device)
        self.critic = DesignCritic()
        self.refiner = DesignRefiner()

    def create_design_with_refinement(self, spec: DesignSpec, max_iterations: int = 3, output_prefix: str = "iter") -> Image.Image:
        """
        Generate a design and iteratively refine it based on AI critique.
        """
        current_spec = spec
        best_image = None
        best_score = -1.0

        for i in range(max_iterations):
            print(f"\n🔄 Iteration {i+1}/{max_iterations}")
            print("-" * 30)

            # 1. Generate
            image = self.generator.create_design(current_spec)
            if image:
                image.save(f"{output_prefix}_{i+1}.png")
            else:
                print("❌ Generator returned None. Stopping iteration.")
                break

            # 2. Critique
            print("🧐 Critiquing design...")
            feedback = self.critic.critique_design(
                image,
                current_spec.goal_name,
                current_spec.tone_description
            )

            score = feedback.get("overall_score", 0)
            print(f"⭐️ Score: {score}/10")
            print(f"📝 Critique: {feedback.get('critique')}")

            # Save best
            if score > best_score:
                best_score = score
                best_image = image

            # Stop if good enough
            if score >= 8.5:
                print("✅ Design meets quality threshold!")
                break

            # 3. Refine (if not last iteration)
            if i < max_iterations - 1:
                current_spec = self.refiner.refine_spec(current_spec, feedback)

        print(f"\n🏆 Best Design Score: {best_score}/10")
        return best_image
