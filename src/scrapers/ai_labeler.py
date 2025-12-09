"""
AI-based labeling using Claude Vision API.

High accuracy (~90-95%) but costs ~$0.003 per image.
Falls back to metadata labeling if API fails.
"""

import base64
import time
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai
import google.api_core.exceptions
import random
from PIL import Image


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
        self.api_key = config.get('gemini', 'api_key')
        self.model_name = config.get('gemini', 'model')

        if not self.api_key:
            raise ValueError("Gemini API key not configured. Set GOOGLE_API_KEY env var or in config.json.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
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

        max_retries = 5
        base_delay = 5  # seconds
        jitter_factor = 0.5

        for attempt in range(max_retries):
            try:
                # Load and encode image
                image = Image.open(image_path).convert('RGB')

                # Build prompt
                prompt_text = self._build_prompt(metadata)

                # Call Gemini API
                response = self.model.generate_content([prompt_text, image])

                # Parse response
                labels = self._parse_response(response.text)

                # Validate
                if not validate_label(labels['v_Goal'], labels['v_Format'], labels['v_Tone']):
                    raise ValueError(f"Invalid labels from AI: {labels}")

                # Add metadata
                labels['confidence'] = 0.9  # High confidence for AI
                labels['method'] = 'gemini_vision' # Updated method name
                labels['source'] = metadata.get('source', 'unknown')
                labels['original_url'] = metadata.get('url', '')

                self.total_requests += 1
                self.successful_requests += 1

                return labels

            except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.GoogleAPIError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay * jitter_factor)
                    print(f"⚠️  Gemini API rate limit hit or transient error (Attempt {attempt+1}/{max_retries}). Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    print(f"❌ Gemini API failed after {max_retries} attempts: {e}")
                    self.total_requests += 1
                    self.failed_requests += 1
                    # Fall through to fallback logic

            except Exception as e:
                print(f"AI labeling failed: {e}")
                self.total_requests += 1
                self.failed_requests += 1
                # Fall through to fallback logic

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
        Estimate cost for labeling N images using Gemini Flash.

        Gemini 1.5 Flash costs approximately $0.00025 per image (dominated by image input).
        """
        cost_per_image = 0.00025
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
