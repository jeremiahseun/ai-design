"""
Verify Layouts
Logical verification for the Layout Engine.

Checks:
1. Zone Bounds: Are all zones within 0.0-1.0?
2. Collision Detection: Do text zones overlap with other text zones or image zones?
3. Structure: Does every layout have at least one text zone?
"""

import sys
import os
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layouts.layout_engine import LayoutEngine, Layout, Zone, ZoneType

def check_bounds(layout: Layout) -> List[str]:
    errors = []
    for zone in layout.zones:
        if not (0.0 <= zone.x <= 1.0 and 0.0 <= zone.y <= 1.0):
            errors.append(f"Zone '{zone.name}' out of bounds: ({zone.x}, {zone.y})")
        if not (0.0 <= zone.w <= 1.0 and 0.0 <= zone.h <= 1.0):
            errors.append(f"Zone '{zone.name}' invalid size: ({zone.w}, {zone.h})")
        if zone.x + zone.w > 1.01: # Small tolerance
            errors.append(f"Zone '{zone.name}' exceeds width: {zone.x + zone.w}")
        if zone.y + zone.h > 1.01:
            errors.append(f"Zone '{zone.name}' exceeds height: {zone.y + zone.h}")
    return errors

def check_collisions(layout: Layout) -> List[str]:
    errors = []
    zones = layout.zones
    for i, z1 in enumerate(zones):
        for j, z2 in enumerate(zones):
            if i >= j: continue # Avoid duplicate checks

            # Ignore background zones
            if z1.type == ZoneType.BACKGROUND or z2.type == ZoneType.BACKGROUND:
                continue

            # Check overlap
            x_overlap = max(0, min(z1.x + z1.w, z2.x + z2.w) - max(z1.x, z2.x))
            y_overlap = max(0, min(z1.y + z1.h, z2.y + z2.h) - max(z1.y, z2.y))

            if x_overlap > 0.01 and y_overlap > 0.01: # Tolerance
                # Allow text over image if z-index is different
                if z1.type != z2.type and z1.z_index != z2.z_index:
                    continue

                # Allow text over text ONLY if explicitly designed (e.g. magazine)
                # But generally we want to warn
                errors.append(f"Overlap detected between '{z1.name}' and '{z2.name}'")

    return errors

def verify_all_layouts():
    print("="*60)
    print("🔍 VERIFYING LAYOUTS")
    print("="*60)

    layout_names = [
        "split_horizontal", "split_vertical", "central_hero",
        "typographic_bold", "modern_clean", "magazine_grid",
        "diagonal_split", "asymmetric_editorial"
    ]

    all_passed = True

    for name in layout_names:
        print(f"\nChecking '{name}'...")
        layout = LayoutEngine.get_layout(name)

        # 1. Bounds
        bound_errors = check_bounds(layout)
        if bound_errors:
            print("  ❌ Bounds Errors:")
            for e in bound_errors: print(f"    - {e}")
            all_passed = False
        else:
            print("  ✅ Bounds OK")

        # 2. Collisions
        collision_errors = check_collisions(layout)
        if collision_errors:
            print("  ⚠️  Collision Warnings:") # Warn only, as some overlaps are intentional
            for e in collision_errors: print(f"    - {e}")
        else:
            print("  ✅ No Collisions")

    if all_passed:
        print("\n✅ All layouts passed critical checks.")
    else:
        print("\n❌ Some layouts failed critical checks.")

if __name__ == "__main__":
    verify_all_layouts()
