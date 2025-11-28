"""
Metadata-based labeling using heuristics and keyword matching.

This is the fallback labeler that doesn't require AI API calls.
Fast, free, and ~70-80% accurate.
"""

import re
from typing import Dict, List, Tuple
from PIL import Image
try:
    from .utils import (
        classify_format_by_aspect_ratio,
        estimate_tone_from_colors,
        validate_label
    )
except ImportError:
    from utils import (
        classify_format_by_aspect_ratio,
        estimate_tone_from_colors,
        validate_label
    )


class MetadataLabeler:
    """Heuristic-based labeler using keywords and image analysis."""

    def __init__(self):
        """Initialize with keyword dictionaries."""
        # Goal classification keywords
        self.goal_keywords = {
            'promotion': [
                'sale', 'discount', 'offer', 'deal', 'promo', 'promotion',
                'off', 'save', 'special', 'limited', 'exclusive', 'now'
            ],
            'education': [
                'course', 'learn', 'tutorial', 'guide', 'lesson', 'class',
                'workshop', 'training', 'education', 'teach', 'study',
                'webinar', 'seminar', 'certification'
            ],
            'branding': [
                'brand', 'identity', 'logo', 'visual', 'style', 'corporate',
                'business', 'company', 'agency', 'studio', 'about'
            ],
            'event': [
                'event', 'concert', 'festival', 'conference', 'party',
                'celebration', 'meetup', 'gathering', 'show', 'exhibition',
                'performance', 'live', 'tickets', 'rsvp', 'join us'
            ],
            'product': [
                'product', 'launch', 'new', 'introducing', 'shop', 'buy',
                'available', 'collection', 'release', 'features', 'specs'
            ],
            'service': [
                'service', 'services', 'consulting', 'hire', 'professional',
                'expert', 'solutions', 'support', 'help', 'assistance'
            ],
            'announcement': [
                'announce', 'announcement', 'news', 'update', 'coming soon',
                'soon', 'alert', 'notice', 'important', 'breaking'
            ],
            'portfolio': [
                'portfolio', 'work', 'projects', 'case study', 'showcase',
                'gallery', 'design', 'creative', 'art', 'artist'
            ],
            'social': [
                'follow', 'like', 'share', 'subscribe', 'instagram',
                'facebook', 'twitter', 'social media', 'tiktok', 'youtube'
            ],
        }

        # Additional context keywords that strengthen classification
        self.context_keywords = {
            'promotion': ['%', '$', 'free', 'get', 'hurry'],
            'education': ['how to', 'why', 'what', 'tips', 'secrets'],
            'event': ['date', 'time', 'venue', 'location', 'when', 'where'],
            'product': ['price', 'order', 'purchase', 'add to cart'],
        }

    def label_from_metadata(
        self,
        title: str = "",
        description: str = "",
        tags: List[str] = None,
        width: int = None,
        height: int = None,
        image: Image.Image = None
    ) -> Dict:
        """
        Generate labels from metadata and image.

        Args:
            title: Design title
            description: Design description
            tags: List of tags/keywords
            width: Image width (original size)
            height: Image height (original size)
            image: PIL Image (optional, for tone estimation)

        Returns:
            {
                'v_Goal': str,
                'v_Format': str,
                'v_Tone': float,
                'confidence': float  # 0-1, how confident we are
            }
        """
        tags = tags or []

        # Classify goal
        v_goal, goal_confidence = self._classify_goal(title, description, tags)

        # Classify format
        if width and height:
            v_format = classify_format_by_aspect_ratio(width, height)
            format_confidence = 0.95  # High confidence from aspect ratio
        else:
            v_format = 'poster'  # Default
            format_confidence = 0.3

        # Estimate tone
        if image:
            v_tone = estimate_tone_from_colors(image)
            tone_confidence = 0.6  # Medium confidence from color analysis
        else:
            v_tone = 0.5  # Neutral default
            tone_confidence = 0.2

        # Overall confidence (weighted average)
        overall_confidence = (
            0.5 * goal_confidence +
            0.3 * format_confidence +
            0.2 * tone_confidence
        )

        return {
            'v_Goal': v_goal,
            'v_Format': v_format,
            'v_Tone': float(v_tone),
            'confidence': float(overall_confidence),
            'method': 'metadata_heuristic'
        }

    def _classify_goal(
        self,
        title: str,
        description: str,
        tags: List[str]
    ) -> Tuple[str, float]:
        """
        Classify goal using keyword matching.

        Returns:
            (goal_string, confidence)
        """
        # Combine all text
        text = f"{title} {description} {' '.join(tags)}".lower()

        # Score each goal
        scores = {}
        for goal, keywords in self.goal_keywords.items():
            score = 0

            # Count primary keyword matches
            for keyword in keywords:
                if keyword in text:
                    score += 1

            # Bonus points for context keywords
            if goal in self.context_keywords:
                for keyword in self.context_keywords[goal]:
                    if keyword in text:
                        score += 0.5

            scores[goal] = score

        # Find best match
        if max(scores.values()) == 0:
            # No matches, default to 'other'
            return 'other', 0.2

        best_goal = max(scores, key=scores.get)
        best_score = scores[best_goal]

        # Calculate confidence
        # High score = high confidence, but cap at 0.9
        confidence = min(0.9, 0.3 + 0.1 * best_score)

        return best_goal, confidence

    def label_batch(
        self,
        metadata_list: List[Dict]
    ) -> List[Dict]:
        """
        Label a batch of designs.

        Args:
            metadata_list: List of metadata dicts with keys:
                - title (str)
                - description (str)
                - tags (List[str])
                - width (int)
                - height (int)
                - image_path (str, optional)

        Returns:
            List of label dicts
        """
        results = []

        for metadata in metadata_list:
            # Load image if path provided
            image = None
            if 'image_path' in metadata:
                try:
                    image = Image.open(metadata['image_path']).convert('RGB')
                except Exception as e:
                    print(f"Warning: Could not load image {metadata.get('image_path')}: {e}")

            # Generate labels
            labels = self.label_from_metadata(
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                tags=metadata.get('tags', []),
                width=metadata.get('width'),
                height=metadata.get('height'),
                image=image
            )

            # Add source metadata
            labels['source'] = metadata.get('source', 'unknown')
            labels['original_url'] = metadata.get('url', '')

            results.append(labels)

        return results

    def refine_with_image_analysis(
        self,
        labels: Dict,
        image: Image.Image
    ) -> Dict:
        """
        Refine labels using additional image analysis.

        Args:
            labels: Existing label dict from label_from_metadata
            image: PIL Image

        Returns:
            Updated label dict with refined values
        """
        # Re-estimate tone with better analysis
        tone = estimate_tone_from_colors(image)

        # Adjust tone based on visual complexity
        img_array = np.array(image.resize((256, 256)))
        edges = cv2.Canny(img_array, 100, 200)
        edge_density = edges.sum() / edges.size

        # High edge density = more busy/energetic
        tone_adjustment = 0.2 * (edge_density / 255.0)
        tone = min(1.0, tone + tone_adjustment)

        # Update labels
        labels['v_Tone'] = float(tone)

        # Increase confidence if image analysis is available
        labels['confidence'] = min(0.95, labels['confidence'] + 0.1)

        return labels


def quick_label(
    title: str = "",
    description: str = "",
    tags: List[str] = None,
    width: int = None,
    height: int = None,
    image_path: str = None
) -> Dict:
    """
    Quick convenience function for single image labeling.

    Args:
        title: Design title
        description: Design description
        tags: List of tags
        width: Image width
        height: Image height
        image_path: Path to image file

    Returns:
        Label dict
    """
    labeler = MetadataLabeler()

    image = None
    if image_path:
        try:
            image = Image.open(image_path).convert('RGB')
            if width is None or height is None:
                width, height = image.size
        except Exception as e:
            print(f"Warning: Could not load image: {e}")

    return labeler.label_from_metadata(
        title=title,
        description=description,
        tags=tags,
        width=width,
        height=height,
        image=image
    )


# Import numpy and cv2 at module level for image analysis
try:
    import numpy as np
    import cv2
except ImportError:
    print("Warning: numpy and opencv not available. Some features disabled.")
    np = None
    cv2 = None


if __name__ == "__main__":
    # Test the labeler
    print("Testing MetadataLabeler...\n")

    labeler = MetadataLabeler()

    # Test cases
    test_cases = [
        {
            'title': "Summer Sale - 50% Off All Items",
            'description': "Limited time offer! Get huge discounts on everything.",
            'tags': ['sale', 'discount', 'shopping'],
            'width': 1080,
            'height': 1080,
        },
        {
            'title': "Learn Python Programming",
            'description': "Complete course for beginners. Master Python in 30 days.",
            'tags': ['education', 'course', 'tutorial'],
            'width': 1920,
            'height': 1080,
        },
        {
            'title': "Music Festival 2024",
            'description': "Join us for the biggest music event of the year!",
            'tags': ['event', 'concert', 'festival'],
            'width': 800,
            'height': 1200,
        },
        {
            'title': "New Product Launch",
            'description': "Introducing our latest innovation. Available now!",
            'tags': ['product', 'launch', 'new'],
            'width': 2400,
            'height': 800,
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Title: {test['title']}")

        labels = labeler.label_from_metadata(**test)

        print(f"  Goal: {labels['v_Goal']} (confidence: {labels['confidence']:.2f})")
        print(f"  Format: {labels['v_Format']}")
        print(f"  Tone: {labels['v_Tone']:.2f}")
        print()
