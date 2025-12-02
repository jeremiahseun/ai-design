"""
Layout Adapter

Dynamically adjusts layout zones based on actual rendered content
to prevent overlaps and maintain visual hierarchy.
"""

from typing import List, Tuple, Dict
from src.layouts.layout_engine import Zone, Layout, ZoneType
from dataclasses import dataclass

@dataclass
class ContentMetrics:
    """Metrics for rendered content in a zone."""
    zone_name: str
    actual_height: float  # Actual height needed (normalized 0-1)
    actual_width: float   # Actual width needed (normalized 0-1)
    line_count: int

class LayoutAdapter:
    """Adapts layout zones based on actual content."""

    @staticmethod
    def estimate_text_height(text: str, font_size: float, zone_width: float,
                            canvas_width: int, canvas_height: int) -> Tuple[float, int]:
        """
        Estimate the height needed for text in normalized coordinates.

        Returns:
            (normalized_height, line_count)
        """
        if not text:
            return 0.0, 0

        # Estimate characters per line based on zone width
        # Average character width is ~0.6em for proportional fonts
        char_width_px = font_size * 0.6
        zone_width_px = zone_width * canvas_width
        chars_per_line = int(zone_width_px / char_width_px)

        if chars_per_line <= 0:
            chars_per_line = 10  # Minimum

        # Calculate line count based on text length
        text_length = len(text)
        line_count = max(1, (text_length + chars_per_line - 1) // chars_per_line)  # Ceiling division

        # Calculate height (line_height is typically 1.5x font size)
        line_height_px = font_size * 1.5
        total_height_px = line_count * line_height_px

        # Add padding (10% of height)
        total_height_px *= 1.1

        # Normalize
        normalized_height = total_height_px / canvas_height

        return normalized_height, line_count

    @staticmethod
    def adapt_zones(layout: Layout, spec, canvas_width: int, canvas_height: int) -> Layout:
        """
        Adapt zones based on actual content from DesignSpec.

        Args:
            layout: Original layout
            spec: DesignSpec with content
            canvas_width, canvas_height: Canvas dimensions

        Returns:
            Adapted Layout with adjusted zones
        """
        from src.utils.typography_engine import ModularScale

        # Create modular scale based on tone
        if spec.tone > 0.7:
            ratio = ModularScale.GOLDEN_RATIO
        elif spec.tone < 0.3:
            ratio = ModularScale.MAJOR_SECOND
        else:
            ratio = ModularScale.MAJOR_THIRD

        scale = ModularScale(base_size=16, ratio=ratio)

        # Font sizes for different elements
        font_sizes = {
            "headline": scale.get_size(3),      # ~40px
            "subheading": scale.get_size(1),    # ~20px
            "details": scale.get_size(0),       # ~16px
            "body_text": scale.get_size(0)      # ~16px
        }

        # Calculate content metrics for each zone
        metrics = []
        for zone in layout.zones:
            if zone.type == ZoneType.TEXT:
                # Get text content based on dynamic name format: text_{index}_{role}
                text = ""
                font_size = 16

                try:
                    parts = zone.name.split('_')
                    if len(parts) >= 3 and parts[0] == "text":
                        idx = int(parts[1])
                        role = parts[2]

                        if 0 <= idx < len(spec.content.elements):
                            text = spec.content.elements[idx].text

                            # Determine font size
                            if role == "primary": font_size = font_sizes["headline"]
                            elif role == "secondary": font_size = font_sizes["subheading"]
                            else: font_size = font_sizes["body_text"]
                except (ValueError, IndexError):
                    pass

                # Estimate height
                if text:
                    height, lines = LayoutAdapter.estimate_text_height(
                        text, font_size, zone.w, canvas_width, canvas_height
                    )

                    metrics.append(ContentMetrics(
                        zone_name=zone.name,
                        actual_height=height,
                        actual_width=zone.w,  # Keep original width
                        line_count=lines
                    ))

        # Adapt zones
        adapted_zones = []
        y_offset = 0.0

        for zone in sorted(layout.zones, key=lambda z: z.y):  # Sort by Y position
            if zone.type == ZoneType.TEXT:
                # Find metrics for this zone
                zone_metrics = next((m for m in metrics if m.zone_name == zone.name), None)

                if zone_metrics:
                    # Check if we need more space
                    needed_height = zone_metrics.actual_height
                    original_height = zone.h

                    # Use whichever is larger
                    new_height = max(needed_height, original_height)

                    # Create adapted zone with adjusted Y and height
                    adapted_zone = Zone(
                        x=zone.x,
                        y=zone.y + y_offset,
                        w=zone.w,
                        h=new_height,
                        type=zone.type,
                        name=zone.name,
                        z_index=zone.z_index
                    )

                    # Accumulate offset if we expanded
                    if new_height > original_height:
                        y_offset += (new_height - original_height)

                    adapted_zones.append(adapted_zone)
                else:
                    # No metrics, keep original but apply offset
                    adapted_zones.append(Zone(
                        x=zone.x,
                        y=zone.y + y_offset,
                        w=zone.w,
                        h=zone.h,
                        type=zone.type,
                        name=zone.name,
                        z_index=zone.z_index
                    ))
            else:
                # Non-text zones (images, logos) - apply offset but don't resize
                adapted_zones.append(Zone(
                    x=zone.x,
                    y=zone.y + y_offset,
                    w=zone.w,
                    h=zone.h,
                    type=zone.type,
                    name=zone.name,
                    z_index=zone.z_index
                ))

        return Layout(
            name=layout.name + "_adapted",
            zones=adapted_zones,
            description=layout.description + " (Adapted)"
        )

    @staticmethod
    def detect_overlaps(zones: List[Zone]) -> List[Tuple[Zone, Zone]]:
        """
        Detect overlapping zones.

        Returns:
            List of (zone1, zone2) tuples that overlap
        """
        overlaps = []

        for i, z1 in enumerate(zones):
            for z2 in zones[i+1:]:
                # Check if rectangles overlap
                # Overlap if: z1.right > z2.left AND z1.left < z2.right AND z1.bottom > z2.top AND z1.top < z2.bottom
                z1_right = z1.x + z1.w
                z1_bottom = z1.y + z1.h
                z2_right = z2.x + z2.w
                z2_bottom = z2.y + z2.h

                if (z1_right > z2.x and z1.x < z2_right and
                    z1_bottom > z2.y and z1.y < z2_bottom):
                    overlaps.append((z1, z2))

        return overlaps
