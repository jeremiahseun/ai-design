"""
Design Refiner

Adjusts DesignSpec parameters based on structured feedback from the Critic.
"""

from src.agents.design_spec import DesignSpec

class DesignRefiner:
    @staticmethod
    def refine_spec(spec: DesignSpec, feedback: dict) -> DesignSpec:
        """
        Apply refinements to the spec based on feedback.
        Returns a NEW modified spec (does not mutate original).
        """
        import copy
        new_spec = copy.deepcopy(spec)

        print("\n🔧 Refiner Actions:")

        for item in feedback.get("actionable_feedback", []):
            issue = item.get("issue", "")
            fix = item.get("fix", "")

            if "contrast" in issue or "contrast" in fix:
                # Adjust tone to increase contrast
                if new_spec.tone < 0.5:
                    new_spec.tone = max(0.1, new_spec.tone - 0.2) # Make it lighter/calmer
                    print(f"  -> Decreased tone to {new_spec.tone:.2f} (Lighten)")
                else:
                    new_spec.tone = min(1.0, new_spec.tone + 0.2) # Make it bolder/darker
                    print(f"  -> Increased tone to {new_spec.tone:.2f} (Darken)")

            if "hierarchy" in issue:
                # Maybe change layout format or just boost tone for impact
                new_spec.tone = min(1.0, new_spec.tone + 0.1)
                print(f"  -> Boosted tone for hierarchy")

            if "cluttered" in issue:
                # Remove tertiary elements if too cluttered
                original_count = len(new_spec.content.elements)
                new_spec.content.elements = [e for e in new_spec.content.elements if e.role != "tertiary"]
                if len(new_spec.content.elements) < original_count:
                    print(f"  -> Removed {original_count - len(new_spec.content.elements)} tertiary elements to reduce clutter")

        return new_spec
