"""
Pinterest scraper for design images.

Scrapes design images from Pinterest search results.
Uses requests + BeautifulSoup for web scraping.
"""

import requests
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin, urlparse, parse_qs

try:
    from .config import Config
    from .utils import (
        ensure_dir,
        save_image,
        save_metadata,
        sanitize_filename,
        DeduplicationCache,
        progress_bar
    )
except ImportError:
    from config import Config
    from utils import (
        ensure_dir,
        save_image,
        save_metadata,
        sanitize_filename,
        DeduplicationCache,
        progress_bar
    )


class PinterestScraper:
    """Scraper for Pinterest design images."""

    def __init__(self, config: Config):
        """
        Initialize Pinterest scraper.

        Args:
            config: Config object
        """
        self.config = config
        self.rate_limit_delay = config.get('figma', 'rate_limit_delay') or 2.0

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        self.dedup_cache = DeduplicationCache()

    def search(
        self,
        query: str,
        max_images: int = 100
    ) -> List[Dict]:
        """
        Search Pinterest for images.

        Note: This is a simplified scraper that extracts image URLs from Pinterest's HTML.
        Pinterest has anti-scraping measures, so this may not work reliably.

        For reliable scraping, consider:
        1. Using Pinterest's official API (requires developer account)
        2. Using a service like Apify or ScraperAPI
        3. Manual collection (save URLs to a text file)

        Args:
            query: Search query (e.g., "graphic design poster")
            max_images: Maximum number of images to scrape

        Returns:
            List of metadata dicts with image URLs
        """
        print(f"Searching Pinterest for: {query}")
        print()
        print("⚠️  Note: Pinterest has anti-scraping measures.")
        print("   For best results, manually collect URLs and use scrape_from_urls()")
        print()

        # Pinterest search URL
        search_url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"

        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()

            # Try to extract image URLs from the page
            # Pinterest uses JavaScript rendering, so this is limited
            image_urls = self._extract_image_urls(response.text)

            print(f"Found {len(image_urls)} image URLs")

            # Build metadata list
            results = []
            for i, url in enumerate(image_urls[:max_images]):
                results.append({
                    'url': url,
                    'title': f"{query} {i+1}",
                    'description': query,
                    'tags': query.split(),
                    'source': 'pinterest',
                })

            return results

        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def _extract_image_urls(self, html: str) -> List[str]:
        """
        Extract image URLs from Pinterest HTML.

        This is a best-effort extraction - Pinterest uses JS rendering.
        """
        urls = []

        # Look for image URLs in various formats
        patterns = [
            r'https://i\.pinimg\.com/originals/[^"\']+\.jpg',
            r'https://i\.pinimg\.com/originals/[^"\']+\.png',
            r'https://i\.pinimg\.com/736x/[^"\']+\.jpg',
            r'https://i\.pinimg\.com/736x/[^"\']+\.png',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html)
            urls.extend(matches)

        # Deduplicate
        urls = list(dict.fromkeys(urls))

        return urls

    def download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download image from URL.

        Args:
            url: Image URL

        Returns:
            PIL Image or None if failed
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Load as PIL Image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image

        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    def scrape_from_urls(
        self,
        urls: List[str],
        output_dir: str,
        metadata_hints: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Scrape images from a list of URLs.

        This is the recommended approach - manually collect Pinterest URLs
        and pass them here.

        Args:
            urls: List of Pinterest image URLs or pin URLs
            output_dir: Directory to save images and metadata
            metadata_hints: Optional list of metadata dicts (title, description, tags)

        Returns:
            List of metadata dicts
        """
        output_dir = ensure_dir(output_dir)

        if metadata_hints is None:
            metadata_hints = [{}] * len(urls)

        results = []
        total = len(urls)

        print(f"Downloading {total} images from Pinterest...")

        for i, (url, hints) in enumerate(zip(urls, metadata_hints), 1):
            print(f"\n[{i}/{total}] Downloading from {url[:60]}...")

            # If it's a pin URL, try to get the image URL
            if 'pinterest.com/pin/' in url:
                image_url = self._get_image_url_from_pin(url)
                if not image_url:
                    print(f"  ✗ Could not extract image URL from pin")
                    continue
            else:
                image_url = url

            # Download image
            image = self.download_image(image_url)
            if image is None:
                print(f"  ✗ Failed to download")
                continue

            # Check for duplicates
            if self.dedup_cache.is_duplicate(image):
                print(f"  ⊘ Duplicate image, skipping")
                continue

            # Build metadata
            metadata = {
                'title': hints.get('title', f'Pinterest Design {i}'),
                'description': hints.get('description', ''),
                'tags': hints.get('tags', []),
                'url': url,
                'image_url': image_url,
                'source': 'pinterest',
                'width': image.width,
                'height': image.height,
            }

            # Save image
            safe_title = sanitize_filename(metadata['title'])
            image_filename = f"pinterest_{i:04d}_{safe_title}.png"
            image_path = output_dir / image_filename
            save_image(image, str(image_path))

            metadata['image_path'] = str(image_path)
            results.append(metadata)

            print(f"  ✓ Saved: {metadata['title']}")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        print(f"\n✓ Download complete: {len(results)}/{total} images saved")

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        save_metadata(results, str(manifest_path))
        print(f"Manifest saved to: {manifest_path}")

        return results

    def _get_image_url_from_pin(self, pin_url: str) -> Optional[str]:
        """
        Get image URL from Pinterest pin URL.

        Args:
            pin_url: Pinterest pin URL (e.g., https://pinterest.com/pin/123456/)

        Returns:
            Direct image URL or None
        """
        try:
            response = self.session.get(pin_url, timeout=10)
            response.raise_for_status()

            # Try to find image URL in HTML
            urls = self._extract_image_urls(response.text)
            if urls:
                return urls[0]

            return None

        except Exception as e:
            print(f"Failed to get image from pin: {e}")
            return None

    def scrape_board(
        self,
        board_url: str,
        output_dir: str,
        max_images: Optional[int] = None
    ) -> List[Dict]:
        """
        Scrape images from a Pinterest board.

        Note: This requires JavaScript rendering, which is beyond simple requests.
        Recommended: Use browser automation (Selenium/Playwright) or manual collection.

        Args:
            board_url: Pinterest board URL
            output_dir: Directory to save images
            max_images: Maximum number of images

        Returns:
            List of metadata dicts
        """
        print(f"Board scraping requires browser automation (Selenium/Playwright)")
        print(f"Please manually collect pin URLs from the board and use scrape_from_urls()")
        return []


def load_urls_from_txt(file_path: str) -> List[str]:
    """
    Load Pinterest URLs from a text file (one per line).

    Args:
        file_path: Path to text file with URLs

    Returns:
        List of URLs
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    urls = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if 'pinterest' in line or 'pinimg.com' in line:
            urls.append(line)

    return urls


def load_urls_with_metadata(file_path: str) -> tuple[List[str], List[Dict]]:
    """
    Load URLs with metadata from a JSON file.

    Expected format:
    [
        {
            "url": "https://...",
            "title": "Design Title",
            "description": "Description",
            "tags": ["tag1", "tag2"]
        },
        ...
    ]

    Args:
        file_path: Path to JSON file

    Returns:
        (list_of_urls, list_of_metadata_hints)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    urls = [item['url'] for item in data]
    metadata = [
        {
            'title': item.get('title', ''),
            'description': item.get('description', ''),
            'tags': item.get('tags', [])
        }
        for item in data
    ]

    return urls, metadata


if __name__ == "__main__":
    # Test the scraper
    print("Pinterest Scraper")
    print("=" * 50)
    print()

    from .config import load_config

    # Load config
    config = load_config()

    # Create scraper
    scraper = PinterestScraper(config)

    print("Pinterest scraper ready!")
    print()
    print("⚠️  Important: Pinterest has anti-scraping measures.")
    print("   For best results, manually collect URLs.")
    print()
    print("Recommended workflow:")
    print("  1. Search Pinterest for 'graphic design poster', etc.")
    print("  2. Manually save image URLs to 'pinterest_urls.txt' (one per line)")
    print("  3. Run: python -m src.scrapers.pinterest_scraper")
    print()
    print("URL formats accepted:")
    print("  - Pin URLs: https://www.pinterest.com/pin/123456/")
    print("  - Direct image URLs: https://i.pinimg.com/originals/.../image.jpg")
    print()
