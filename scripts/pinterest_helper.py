"""
Pinterest Integration Helper

Provides easy access to Pinterest reference images for the Advanced Design Pipeline.
"""

import os
import glob
from pathlib import Path
from typing import List

def get_pinterest_images(max_images: int = 5) -> List[str]:
    """
    Get paths to Pinterest reference images.

    Args:
        max_images: Maximum number of images to return

    Returns:
        List of absolute paths to Pinterest images
    """
    # Check for downloaded Pinterest images
    pinterest_dir = Path("src/scraped_data/raw/pinterest")

    if not pinterest_dir.exists():
        print(f"⚠️  Pinterest directory not found: {pinterest_dir}")
        return []

    # Get all PNG/JPG images
    image_patterns = [
        str(pinterest_dir / "*.png"),
        str(pinterest_dir / "*.jpg"),
        str(pinterest_dir / "*.jpeg"),
    ]

    images = []
    for pattern in image_patterns:
        images.extend(glob.glob(pattern))

    # Return absolute paths
    images = [os.path.abspath(img) for img in images[:max_images]]

    print(f"📸 Found {len(images)} Pinterest reference images")
    return images

def get_pinterest_images_by_query(query: str = "graphic design poster", max_images: int = 10) -> List[str]:
    """
    Get Pinterest images matching a query (requires running scraper first).

    This function is a placeholder - you would need to:
    1. Run the Pinterest scraper with the query
    2. Download images
    3. Return the paths

    For now, it just returns existing images.
    """
    print(f"📌 Note: Using existing Pinterest images (query-based scraping requires separate workflow)")
    return get_pinterest_images(max_images)

if __name__ == "__main__":
    # Test
    images = get_pinterest_images(3)
    for img in images:
        print(f"  - {img}")
