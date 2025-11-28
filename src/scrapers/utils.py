"""
Utility functions for scraping and labeling.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(image: Image.Image, output_path: str) -> None:
    """Save PIL Image to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def load_image(image_path: str) -> Image.Image:
    """Load image from file."""
    return Image.open(image_path).convert('RGB')


def resize_image(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Resize image to target size with smart cropping.

    For designs, we use center crop to preserve the main content.
    """
    # Calculate aspect ratios
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # Image is wider, crop width
        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    else:
        # Image is taller, crop height
        new_height = int(image.width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))

    # Resize to target
    return image.resize(target_size, Image.Resampling.LANCZOS)


def compute_image_hash(image: Image.Image) -> str:
    """Compute hash of image for deduplication."""
    img_bytes = image.tobytes()
    return hashlib.md5(img_bytes).hexdigest()


def save_metadata(metadata: Dict, output_path: str) -> None:
    """Save metadata to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(metadata_path: str) -> Dict:
    """Load metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_aspect_ratio(width: int, height: int) -> float:
    """Calculate aspect ratio."""
    return width / height


def classify_format_by_aspect_ratio(width: int, height: int) -> str:
    """
    Classify design format based on aspect ratio.

    Returns:
        'poster', 'social', 'flyer', or 'banner'
    """
    ratio = get_aspect_ratio(width, height)

    if ratio > 2.0:
        return 'banner'  # Wide horizontal
    elif ratio > 1.3:
        return 'flyer'  # Moderate horizontal (A4-like)
    elif ratio < 0.6:
        return 'social'  # Vertical (Instagram story, etc.)
    else:
        return 'poster'  # Square-ish or moderate vertical


def extract_dominant_colors(image: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from image using k-means clustering.

    Args:
        image: PIL Image
        k: Number of dominant colors to extract

    Returns:
        List of RGB tuples
    """
    # Convert to numpy array
    img_array = np.array(image.resize((100, 100)))  # Downsample for speed
    pixels = img_array.reshape(-1, 3).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Convert centers to RGB tuples
    colors = [tuple(map(int, center)) for center in centers]
    return colors


def estimate_tone_from_colors(image: Image.Image) -> float:
    """
    Estimate design tone (energy level) from colors.

    Higher saturation and brightness = more energetic
    Returns float in [0, 1]
    """
    # Convert to HSV
    img_array = np.array(image.resize((100, 100)))
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Calculate mean saturation and value
    saturation = hsv[:, :, 1].mean() / 255.0
    value = hsv[:, :, 2].mean() / 255.0

    # Tone is weighted average (saturation matters more for energy)
    tone = 0.6 * saturation + 0.4 * value

    return float(np.clip(tone, 0.0, 1.0))


def rate_limit_sleep(delay: float = 1.0) -> None:
    """Sleep to respect rate limits."""
    time.sleep(delay)


def progress_bar(current: int, total: int, prefix: str = '', length: int = 50) -> None:
    """Print a progress bar to console."""
    percent = 100 * (current / float(total))
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # New line when complete


class DeduplicationCache:
    """Cache for deduplicating images by content hash."""

    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, image: Image.Image) -> bool:
        """Check if image is a duplicate."""
        img_hash = compute_image_hash(image)
        if img_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(img_hash)
        return False

    def add_hash(self, img_hash: str) -> None:
        """Manually add a hash to the cache."""
        self.seen_hashes.add(img_hash)

    def size(self) -> int:
        """Return number of unique images seen."""
        return len(self.seen_hashes)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def encode_v_goal(goal: str) -> int:
    """
    Encode goal string to integer class.

    Goal classes (10 total):
    0: promotion
    1: education
    2: branding
    3: event
    4: product
    5: service
    6: announcement
    7: portfolio
    8: social
    9: other
    """
    goal_mapping = {
        'promotion': 0,
        'education': 1,
        'branding': 2,
        'event': 3,
        'product': 4,
        'service': 5,
        'announcement': 6,
        'portfolio': 7,
        'social': 8,
        'other': 9,
    }
    return goal_mapping.get(goal.lower(), 9)  # Default to 'other'


def encode_v_format(format_str: str) -> int:
    """
    Encode format string to integer class.

    Format classes (4 total):
    0: poster
    1: social
    2: flyer
    3: banner
    """
    format_mapping = {
        'poster': 0,
        'social': 1,
        'flyer': 2,
        'banner': 3,
    }
    return format_mapping.get(format_str.lower(), 0)  # Default to 'poster'


def validate_label(v_goal: str, v_format: str, v_tone: float) -> bool:
    """
    Validate that labels are within expected ranges.

    Returns:
        True if valid, False otherwise
    """
    valid_goals = ['promotion', 'education', 'branding', 'event', 'product',
                   'service', 'announcement', 'portfolio', 'social', 'other']
    valid_formats = ['poster', 'social', 'flyer', 'banner']

    if v_goal not in valid_goals:
        return False
    if v_format not in valid_formats:
        return False
    if not (0.0 <= v_tone <= 1.0):
        return False

    return True


def format_dataset_stats(metadata_list: List[Dict]) -> str:
    """
    Generate statistics summary for dataset.

    Args:
        metadata_list: List of metadata dicts with v_Goal, v_Format, v_Tone

    Returns:
        Formatted string with statistics
    """
    from collections import Counter

    total = len(metadata_list)

    # Count goals and formats
    goal_counts = Counter(m['v_Goal'] for m in metadata_list)
    format_counts = Counter(m['v_Format'] for m in metadata_list)

    # Tone statistics
    tones = [m['v_Tone'] for m in metadata_list]
    tone_mean = np.mean(tones)
    tone_std = np.std(tones)

    stats = f"""
Dataset Statistics
==================
Total Images: {total}

Goal Distribution:
{'-' * 40}
"""
    for goal, count in sorted(goal_counts.items()):
        pct = 100 * count / total
        stats += f"  {goal:<15} {count:>5} ({pct:>5.1f}%)\n"

    stats += f"""
Format Distribution:
{'-' * 40}
"""
    for fmt, count in sorted(format_counts.items()):
        pct = 100 * count / total
        stats += f"  {fmt:<15} {count:>5} ({pct:>5.1f}%)\n"

    stats += f"""
Tone Statistics:
{'-' * 40}
  Mean:   {tone_mean:.3f}
  Std:    {tone_std:.3f}
  Min:    {min(tones):.3f}
  Max:    {max(tones):.3f}
"""

    return stats
