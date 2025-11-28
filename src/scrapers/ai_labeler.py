"""
AI-based labeling using Claude Vision API.

High accuracy (~90-95%) but costs ~$0.003 per image.
Falls back to metadata labeling if API fails.
"""

import base64
import time
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import anthropic

try:
    from .config import Config
    from .metadata_labeler import MetadataLabeler
    from .utils import validate_label
except ImportError:
    from config import Config
    from metadata_labeler import MetadataLabeler
    from utils import validate_label


class AILabeler:
    """Claude Vision-based labeler for high-accuracy classification."""

    def __init__(self, config: Config):
        """
        Initialize AI labeler.

        Args:
            config: Config object with Claude API key
        """
        self.config = config
        self.api_key = config.get('claude', 'api_key')
        self.model = config.get('claude', 'model')
        self.max_tokens = config.get('claude', 'max_tokens')

        if not self.api_key:
            raise ValueError("Claude API key not configured. Check config.json")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.metadata_labeler = MetadataLabeler()  # Fallback

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_requests = 0

    def label_image(
        self,
        image_path: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Label a single image using Claude Vision.

        Args:
            image_path: Path to image file
            metadata: Optional metadata dict with title, description, tags

        Returns:
            Label dict with v_Goal, v_Format, v_Tone, confidence, method
        """
        metadata = metadata or {}

        try:
            # Load and encode image
            image = Image.open(image_path).convert('RGB')

            # Resize if too large (Claude has size limits)
            if max(image.size) > 1568:
                ratio = 1568 / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to base64
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()

            # Build prompt
            prompt = self._build_prompt(metadata)

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            # Parse response
            labels = self._parse_response(response.content[0].text)

            # Validate
            if not validate_label(labels['v_Goal'], labels['v_Format'], labels['v_Tone']):
                raise ValueError(f"Invalid labels from AI: {labels}")

            # Add metadata
            labels['confidence'] = 0.9  # High confidence for AI
            labels['method'] = 'claude_vision'
            labels['source'] = metadata.get('source', 'unknown')
            labels['original_url'] = metadata.get('url', '')

            self.total_requests += 1
            self.successful_requests += 1

            return labels

        except Exception as e:
            print(f"AI labeling failed: {e}")
            print("Falling back to metadata labeling...")

            self.total_requests += 1
            self.failed_requests += 1
            self.fallback_requests += 1

            # Fallback to metadata labeling
            image = Image.open(image_path).convert('RGB')
            width, height = image.size

            labels = self.metadata_labeler.label_from_metadata(
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                tags=metadata.get('tags', []),
                width=width,
                height=height,
                image=image
            )

            labels['method'] = 'metadata_fallback'
            labels['source'] = metadata.get('source', 'unknown')
            labels['original_url'] = metadata.get('url', '')

            return labels

    def label_batch(
        self,
        image_paths: List[str],
        metadata_list: Optional[List[Dict]] = None,
        rate_limit_delay: float = 1.0
    ) -> List[Dict]:
        """
        Label a batch of images.

        Args:
            image_paths: List of image file paths
            metadata_list: Optional list of metadata dicts
            rate_limit_delay: Seconds to wait between API calls

        Returns:
            List of label dicts
        """
        if metadata_list is None:
            metadata_list = [{}] * len(image_paths)

        results = []
        total = len(image_paths)

        print(f"Labeling {total} images with Claude Vision...")

        for i, (image_path, metadata) in enumerate(zip(image_paths, metadata_list), 1):
            # Label image
            labels = self.label_image(image_path, metadata)
            results.append(labels)

            # Progress
            if i % 10 == 0:
                print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")
                print(f"  Success: {self.successful_requests}, Failed: {self.failed_requests}")

            # Rate limiting
            if i < total:
                time.sleep(rate_limit_delay)

        print(f"Batch complete: {self.successful_requests} successful, {self.failed_requests} failed")

        return results

    def _build_prompt(self, metadata: Dict) -> str:
        """Build prompt for Claude Vision."""
        prompt = """Analyze this graphic design and classify it with the following labels:

1. **Goal** (the purpose/intent of the design):
   - promotion (sales, discounts, offers, deals)
   - education (courses, tutorials, guides, learning)
   - branding (brand identity, corporate, company)
   - event (concerts, festivals, conferences, parties)
   - product (product launches, features, shopping)
   - service (services, consulting, professional help)
   - announcement (news, updates, alerts, notices)
   - portfolio (showcasing work, creative projects)
   - social (social media content, engagement)
   - other (anything else)

2. **Format** (the design medium):
   - poster (square-ish or vertical, like movie posters)
   - social (vertical, like Instagram stories)
   - flyer (moderate horizontal, like A4 handouts)
   - banner (wide horizontal, like web banners)

3. **Tone** (the energy/mood of the design):
   - A float from 0.0 (calm, peaceful, minimal) to 1.0 (energetic, bold, vibrant)
   - Consider: color saturation, contrast, visual density, typography boldness

"""

        # Add context if available
        if metadata.get('title'):
            prompt += f"\nTitle: {metadata['title']}"
        if metadata.get('description'):
            prompt += f"\nDescription: {metadata['description']}"
        if metadata.get('tags'):
            prompt += f"\nTags: {', '.join(metadata['tags'])}"

        prompt += """

Return ONLY the three values in this exact format (no explanations):
goal,format,tone

Example: promotion,poster,0.8
"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse Claude's response into label dict.

        Expected format: "goal,format,tone"
        Example: "promotion,poster,0.8"
        """
        # Clean up response
        response_text = response_text.strip()

        # Handle potential extra text
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if ',' in line and len(line.split(',')) == 3:
                response_text = line
                break

        # Parse
        try:
            parts = response_text.split(',')
            if len(parts) != 3:
                raise ValueError(f"Expected 3 parts, got {len(parts)}")

            v_goal = parts[0].strip().lower()
            v_format = parts[1].strip().lower()
            v_tone = float(parts[2].strip())

            return {
                'v_Goal': v_goal,
                'v_Format': v_format,
                'v_Tone': v_tone
            }

        except Exception as e:
            raise ValueError(f"Failed to parse response '{response_text}': {e}")

    def get_stats(self) -> Dict:
        """Get labeling statistics."""
        return {
            'total_requests': self.total_requests,
            'successful': self.successful_requests,
            'failed': self.failed_requests,
            'fallback': self.fallback_requests,
            'success_rate': self.successful_requests / max(1, self.total_requests)
        }

    def estimate_cost(self, num_images: int) -> float:
        """
        Estimate cost for labeling N images.

        Claude Vision costs approximately $0.003 per image.
        """
        cost_per_image = 0.003
        return num_images * cost_per_image


def quick_label_with_ai(
    image_path: str,
    config: Config,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Quick convenience function for single image AI labeling.

    Args:
        image_path: Path to image file
        config: Config object with API key
        metadata: Optional metadata dict

    Returns:
        Label dict
    """
    labeler = AILabeler(config)
    return labeler.label_image(image_path, metadata)


if __name__ == "__main__":
    # Test the AI labeler
    print("Testing AILabeler...\n")

    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    # Load config
    config = load_config()

    # Validate
    is_valid, errors = config.validate()
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease configure API keys in config.json")
        exit(1)

    # Create labeler
    labeler = AILabeler(config)

    # Test with a sample image
    print("Ready to test AI labeling!")
    print("Provide an image path to test:")
    print("  Example: python -m src.scrapers.ai_labeler path/to/image.png")
