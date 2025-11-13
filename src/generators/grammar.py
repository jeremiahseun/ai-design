"""
Module 4: Grammar Engine - F_Tensor to V_Grammar
Calculates 4 design grammar scores from F_Tensor:
    1. Alignment: Histogram variance of element positions
    2. Contrast: WCAG luminance ratio between text and background
    3. Whitespace: Gini coefficient of distance transform
    4. Hierarchy: Cosine similarity between target and visual weight

Dependencies: OpenCV (cv2), NumPy
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


class GrammarEngine:
    """
    Evaluates design quality using 4 grammar principles.
    All calculations are done on F_Tensor, not the rendered image.
    """

    def __init__(self, canvas_size: int = 256):
        """
        Initialize grammar engine
        Args:
            canvas_size: Size of the canvas
        """
        self.canvas_size = canvas_size

    def calculate_all(self, f_tensor: np.ndarray, rendered_img: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all 4 grammar scores

        Args:
            f_tensor: F_Tensor [4, H, W]
            rendered_img: Optional rendered image [H, W, 3] for contrast calculation

        Returns:
            Dictionary with scores [0, 1] for each grammar dimension
        """
        alignment = self.calculate_alignment(f_tensor)
        contrast = self.calculate_contrast(f_tensor, rendered_img)
        whitespace = self.calculate_whitespace(f_tensor)
        hierarchy = self.calculate_hierarchy(f_tensor, rendered_img)

        return {
            'Alignment': alignment,
            'Contrast': contrast,
            'Whitespace': whitespace,
            'Hierarchy': hierarchy
        }

    def calculate_alignment(self, f_tensor: np.ndarray) -> float:
        """
        Calculate alignment score using histogram variance of x-coordinates

        Logic:
        - Well-aligned designs have elements clustered at specific x-positions
        - Calculate histogram of element x-coordinates
        - Lower variance = better alignment (more clustered)
        - Score = 1 - normalized_variance

        Args:
            f_tensor: F_Tensor [4, H, W]

        Returns:
            Alignment score [0, 1], higher is better
        """
        # Combine text and image masks to get all content
        content_mask = np.maximum(f_tensor[0], f_tensor[1])

        if content_mask.sum() == 0:
            return 0.0

        # Get x-coordinates of all content pixels
        y_coords, x_coords = np.where(content_mask > 0)

        # Create histogram of x-coordinates (bin by columns)
        hist, _ = np.histogram(x_coords, bins=self.canvas_size)

        # Normalize histogram
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)

        # Calculate variance
        variance = np.var(hist)

        # Normalize variance to [0, 1] range
        # Maximum variance occurs when distribution is most spread out
        max_variance = 1.0 / self.canvas_size  # Uniform distribution
        normalized_variance = min(variance / (max_variance + 1e-8), 1.0)

        # Higher score = lower variance = better alignment
        alignment_score = 1.0 - normalized_variance

        # Apply sigmoid to make score more discriminative
        alignment_score = self._sigmoid(alignment_score * 2 - 1)

        return float(np.clip(alignment_score, 0, 1))

    def calculate_contrast(self, f_tensor: np.ndarray, rendered_img: np.ndarray = None) -> float:
        """
        Calculate contrast score using WCAG luminance ratio

        Logic:
        - Sample pixels at text boundaries (using mask dilation)
        - Calculate luminance for text pixels vs background pixels
        - Use WCAG formula: (L_lighter + 0.05) / (L_darker + 0.05)
        - Good contrast ratio is >= 4.5 (WCAG AA standard)

        Args:
            f_tensor: F_Tensor [4, H, W]
            rendered_img: Rendered image [H, W, 3] in range [0, 255]

        Returns:
            Contrast score [0, 1], higher is better
        """
        if rendered_img is None:
            # Fallback: estimate from color IDs
            return self._estimate_contrast_from_colors(f_tensor)

        # Get text mask
        text_mask = f_tensor[0].astype(np.uint8)

        if text_mask.sum() == 0:
            return 1.0  # No text = perfect contrast (nothing to evaluate)

        # Dilate text mask to get boundary regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(text_mask, kernel, iterations=1)
        boundary = dilated - text_mask  # Boundary region

        # Sample text pixels
        text_pixels = rendered_img[text_mask == 1]
        if len(text_pixels) == 0:
            return 0.5

        # Sample background pixels (boundary region)
        bg_pixels = rendered_img[boundary == 1]
        if len(bg_pixels) == 0:
            # Use inverse mask as background
            bg_mask = 1 - text_mask
            bg_pixels = rendered_img[bg_mask == 1]

        if len(bg_pixels) == 0:
            return 0.5

        # Calculate average luminance
        text_lum = self._calculate_luminance(text_pixels.mean(axis=0))
        bg_lum = self._calculate_luminance(bg_pixels.mean(axis=0))

        # WCAG contrast ratio
        lighter = max(text_lum, bg_lum)
        darker = min(text_lum, bg_lum)
        contrast_ratio = (lighter + 0.05) / (darker + 0.05)

        # Normalize to [0, 1]
        # WCAG AA standard is 4.5, AAA is 7
        # Map 1 -> 0, 4.5 -> 0.5, 7+ -> 1
        score = np.clip((contrast_ratio - 1) / 6, 0, 1)

        return float(score)

    def _calculate_luminance(self, rgb: np.ndarray) -> float:
        """
        Calculate relative luminance using WCAG formula
        RGB should be in range [0, 255]
        """
        # Normalize to [0, 1]
        rgb = rgb / 255.0

        # Apply gamma correction
        def gamma_correct(c):
            if c <= 0.03928:
                return c / 12.92
            else:
                return ((c + 0.055) / 1.055) ** 2.4

        r, g, b = [gamma_correct(c) for c in rgb]

        # Calculate luminance
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _estimate_contrast_from_colors(self, f_tensor: np.ndarray) -> float:
        """
        Fallback: Estimate contrast from color IDs in F_Tensor
        """
        text_mask = f_tensor[0]
        color_map = f_tensor[2]

        if text_mask.sum() == 0:
            return 1.0

        # Get color IDs of text regions
        text_color_ids = color_map[text_mask == 1]

        # Get color IDs of background
        bg_mask = (text_mask == 0) & (f_tensor[1] == 0)  # Neither text nor image
        bg_color_ids = color_map[bg_mask]

        if len(text_color_ids) == 0 or len(bg_color_ids) == 0:
            return 0.5

        # Different color IDs suggest contrast
        text_color = np.median(text_color_ids)
        bg_color = np.median(bg_color_ids)

        # Simple heuristic: different colors = good contrast
        if abs(text_color - bg_color) > 2:
            return 0.8
        elif abs(text_color - bg_color) > 1:
            return 0.6
        else:
            return 0.4

    def calculate_whitespace(self, f_tensor: np.ndarray) -> float:
        """
        Calculate whitespace distribution using Gini coefficient

        Logic:
        - Apply distance transform on inverted content mask
        - Distance transform gives distance to nearest content pixel
        - Calculate Gini coefficient of distance values
        - Lower Gini = more uniform whitespace distribution = better

        Args:
            f_tensor: F_Tensor [4, H, W]

        Returns:
            Whitespace score [0, 1], higher is better
        """
        # Combine text and image masks
        content_mask = np.maximum(f_tensor[0], f_tensor[1])

        if content_mask.sum() == 0:
            return 0.0  # No content = no whitespace to evaluate

        # Invert mask (1 where empty, 0 where content)
        empty_mask = (1 - content_mask).astype(np.uint8)

        # Distance transform
        dist_transform = cv2.distanceTransform(empty_mask, cv2.DIST_L2, 5)

        # Get distance values for empty regions
        distances = dist_transform[empty_mask == 1]

        if len(distances) == 0:
            return 0.0

        # Calculate Gini coefficient
        gini = self._gini_coefficient(distances)

        # Lower Gini = better whitespace distribution
        # Normalize and invert
        score = 1.0 - np.clip(gini, 0, 1)

        return float(score)

    def _gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient
        0 = perfect equality, 1 = perfect inequality
        """
        if len(values) == 0:
            return 0.0

        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)

        # Calculate Gini
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

        return float(np.clip(gini, 0, 1))

    def calculate_hierarchy(self, f_tensor: np.ndarray, rendered_img: np.ndarray = None) -> float:
        """
        Calculate hierarchy score using cosine similarity

        Logic:
        - Target hierarchy is in F_Tensor[3] (designed hierarchy)
        - Calculate visual weight from size and contrast
        - Visual weight = element_size * contrast_factor
        - Compare using cosine similarity

        Args:
            f_tensor: F_Tensor [4, H, W]
            rendered_img: Optional rendered image for better weight calculation

        Returns:
            Hierarchy score [0, 1], higher is better
        """
        # Get target hierarchy from F_Tensor[3]
        target_hierarchy = f_tensor[3]

        if target_hierarchy.sum() == 0:
            return 0.0

        # Calculate visual weight
        # Weight = function of size (from masks) and position (from F_Tensor)
        content_mask = np.maximum(f_tensor[0], f_tensor[1])

        # Size weight: larger elements have more weight
        size_weight = cv2.GaussianBlur(content_mask, (15, 15), 0)

        # Position weight: elements higher in the canvas have more weight
        y_coords = np.arange(self.canvas_size).reshape(-1, 1)
        y_coords = 1 - (y_coords / self.canvas_size)  # Top = 1, Bottom = 0
        position_weight = np.repeat(y_coords, self.canvas_size, axis=1)

        # Combine weights
        visual_weight = size_weight * 0.7 + position_weight * 0.3

        # Normalize both maps to [0, 1]
        target_norm = target_hierarchy / (target_hierarchy.max() + 1e-8)
        visual_norm = visual_weight / (visual_weight.max() + 1e-8)

        # Flatten and compute cosine similarity
        target_flat = target_norm.flatten()
        visual_flat = visual_norm.flatten()

        # Cosine similarity
        dot_product = np.dot(target_flat, visual_flat)
        norm_target = np.linalg.norm(target_flat)
        norm_visual = np.linalg.norm(visual_flat)

        if norm_target == 0 or norm_visual == 0:
            return 0.0

        cosine_sim = dot_product / (norm_target * norm_visual)

        # Map from [-1, 1] to [0, 1]
        score = (cosine_sim + 1) / 2

        return float(np.clip(score, 0, 1))

    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid function for score normalization
        """
        return 1 / (1 + np.exp(-x))

    def calculate_batch(self, f_tensors: np.ndarray, rendered_imgs: np.ndarray = None) -> np.ndarray:
        """
        Calculate grammar scores for a batch

        Args:
            f_tensors: Batch of F_Tensors [B, 4, H, W]
            rendered_imgs: Optional batch of rendered images [B, H, W, 3]

        Returns:
            Grammar scores [B, 4] in order [Alignment, Contrast, Whitespace, Hierarchy]
        """
        batch_size = f_tensors.shape[0]
        scores = np.zeros((batch_size, 4), dtype=np.float32)

        for i in range(batch_size):
            f_tensor = f_tensors[i]
            rendered_img = rendered_imgs[i] if rendered_imgs is not None else None

            result = self.calculate_all(f_tensor, rendered_img)
            scores[i] = [
                result['Alignment'],
                result['Contrast'],
                result['Whitespace'],
                result['Hierarchy']
            ]

        return scores


if __name__ == '__main__':
    # Test the grammar engine
    import sys
    sys.path.append('src')
    from generators.generator import DesignGenerator
    from generators.renderer import DesignRenderer
    from generators.extractor import FeatureExtractor

    print("=" * 60)
    print("Grammar Engine Test")
    print("=" * 60)

    # Generate designs with different layouts
    gen = DesignGenerator(seed=42)
    renderer = DesignRenderer()
    extractor = FeatureExtractor()
    grammar = GrammarEngine()

    for layout_type in ['left_aligned', 'center_aligned', 'mixed']:
        print(f"\n{layout_type.upper()}:")
        print("-" * 60)

        # Generate 3 designs of this type
        designs = []
        for _ in range(3):
            design = gen.generate()
            if design['layout'] == layout_type:
                designs.append(design)
                if len(designs) == 3:
                    break

        if not designs:
            designs = [gen.generate()]

        for i, design in enumerate(designs[:1]):  # Test first one
            # Render and extract
            rendered = renderer.render(design)
            f_tensor = extractor.extract(design)

            # Calculate grammar scores
            scores = grammar.calculate_all(f_tensor, rendered)

            print(f"  Design {i+1}:")
            for dimension, score in scores.items():
                bar = "█" * int(score * 20)
                print(f"    {dimension:12s}: {score:.3f} {bar}")

    # Test batch processing
    print("\n" + "=" * 60)
    print("Batch Processing Test")
    print("=" * 60)

    designs = gen.generate_batch(5)
    f_tensors = extractor.extract_batch(designs)
    rendered_batch = renderer.render_batch(designs)

    batch_scores = grammar.calculate_batch(f_tensors, rendered_batch)
    print(f"\nBatch scores shape: {batch_scores.shape}")
    print(f"Mean scores: {batch_scores.mean(axis=0)}")
    print(f"  [Alignment, Contrast, Whitespace, Hierarchy]")

    print("\n" + "=" * 60)
