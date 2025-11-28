"""
Figma Community scraper.

Scrapes designs from Figma Community using the Figma REST API.
Requires Figma access token (get from figma.com/developers).
"""

import requests
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from io import BytesIO

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


class FigmaScraper:
    """Scraper for Figma Community designs."""

    def __init__(self, config: Config):
        """
        Initialize Figma scraper.

        Args:
            config: Config object with Figma access token
        """
        self.config = config
        self.access_token = config.get('figma', 'access_token')
        self.rate_limit_delay = config.get('figma', 'rate_limit_delay') or 1.0

        if not self.access_token:
            raise ValueError("Figma access token not configured. Check config.json")

        self.base_url = "https://api.figma.com/v1"
        self.headers = {
            'X-Figma-Token': self.access_token
        }

        self.dedup_cache = DeduplicationCache()

    def search_community(
        self,
        query: str,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search Figma Community for designs.

        Note: Figma doesn't have a public search API for community files.
        This is a placeholder that shows the structure for when you have file keys.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of file metadata dicts
        """
        print(f"Note: Figma Community doesn't have a public search API.")
        print(f"You'll need to manually collect file keys from the community.")
        print(f"File keys look like: 'abcd1234efgh5678' (from URLs)")
        print()

        # Return empty list - users need to provide file keys
        return []

    def get_file_metadata(self, file_key: str) -> Optional[Dict]:
        """
        Get metadata for a Figma file.

        Args:
            file_key: Figma file key (from URL)

        Returns:
            Metadata dict or None if failed
        """
        url = f"{self.base_url}/files/{file_key}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            # Extract metadata
            metadata = {
                'file_key': file_key,
                'title': data.get('name', 'Untitled'),
                'description': '',  # Not in API response
                'tags': [],  # Not in API response
                'url': f"https://www.figma.com/file/{file_key}",
                'source': 'figma',
            }

            # Get dimensions from first page/frame
            if 'document' in data and 'children' in data['document']:
                first_page = data['document']['children'][0]
                if 'children' in first_page and first_page['children']:
                    first_frame = first_page['children'][0]
                    if 'absoluteBoundingBox' in first_frame:
                        bbox = first_frame['absoluteBoundingBox']
                        metadata['width'] = int(bbox.get('width', 1920))
                        metadata['height'] = int(bbox.get('height', 1080))

            return metadata

        except Exception as e:
            print(f"Failed to get file metadata for {file_key}: {e}")
            return None

    def export_image(
        self,
        file_key: str,
        node_id: Optional[str] = None,
        scale: float = 2.0,
        format: str = 'png'
    ) -> Optional[Image.Image]:
        """
        Export an image from a Figma file.

        Args:
            file_key: Figma file key
            node_id: Specific node to export (if None, exports first frame)
            scale: Export scale (1.0 = original size, 2.0 = 2x)
            format: Export format ('png', 'jpg', 'svg')

        Returns:
            PIL Image or None if failed
        """
        # If no node_id, get the first frame
        if node_id is None:
            node_id = self._get_first_frame_id(file_key)
            if node_id is None:
                return None

        # Request export
        url = f"{self.base_url}/images/{file_key}"
        params = {
            'ids': node_id,
            'scale': scale,
            'format': format,
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Get image URL
            if 'images' not in data or node_id not in data['images']:
                print(f"No image URL in response for {file_key}")
                return None

            image_url = data['images'][node_id]

            # Download image
            time.sleep(0.5)  # Small delay before downloading
            img_response = requests.get(image_url)
            img_response.raise_for_status()

            # Load as PIL Image
            image = Image.open(BytesIO(img_response.content))
            return image

        except Exception as e:
            print(f"Failed to export image from {file_key}: {e}")
            return None

    def _get_first_frame_id(self, file_key: str) -> Optional[str]:
        """Get the ID of the first frame in the file."""
        url = f"{self.base_url}/files/{file_key}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            # Navigate to first frame
            if 'document' in data and 'children' in data['document']:
                first_page = data['document']['children'][0]
                if 'children' in first_page and first_page['children']:
                    first_frame = first_page['children'][0]
                    return first_frame.get('id')

            return None

        except Exception as e:
            print(f"Failed to get frame ID from {file_key}: {e}")
            return None

    def scrape_file(
        self,
        file_key: str,
        output_dir: str
    ) -> Optional[Dict]:
        """
        Scrape a single Figma file.

        Args:
            file_key: Figma file key
            output_dir: Directory to save images and metadata

        Returns:
            Metadata dict with image_path added, or None if failed
        """
        output_dir = ensure_dir(output_dir)

        # Get metadata
        metadata = self.get_file_metadata(file_key)
        if metadata is None:
            return None

        # Export image
        image = self.export_image(file_key)
        if image is None:
            return None

        # Check for duplicates
        if self.dedup_cache.is_duplicate(image):
            print(f"Duplicate image: {file_key}")
            return None

        # Save image
        safe_title = sanitize_filename(metadata['title'])
        image_filename = f"{file_key}_{safe_title}.png"
        image_path = output_dir / image_filename
        save_image(image, str(image_path))

        # Add image path to metadata
        metadata['image_path'] = str(image_path)
        metadata['width'], metadata['height'] = image.size

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        return metadata

    def scrape_files(
        self,
        file_keys: List[str],
        output_dir: str,
        max_files: Optional[int] = None
    ) -> List[Dict]:
        """
        Scrape multiple Figma files.

        Args:
            file_keys: List of Figma file keys
            output_dir: Directory to save images and metadata
            max_files: Maximum number of files to scrape

        Returns:
            List of metadata dicts
        """
        output_dir = ensure_dir(output_dir)

        if max_files:
            file_keys = file_keys[:max_files]

        results = []
        total = len(file_keys)

        print(f"Scraping {total} Figma files...")

        for i, file_key in enumerate(file_keys, 1):
            print(f"\n[{i}/{total}] Processing {file_key}...")

            metadata = self.scrape_file(file_key, output_dir)
            if metadata:
                results.append(metadata)
                print(f"  ✓ Saved: {metadata['title']}")
            else:
                print(f"  ✗ Failed")

            # Progress
            if i % 10 == 0:
                print(f"\nProgress: {len(results)}/{i} successful")

        print(f"\n✓ Scraping complete: {len(results)}/{total} files saved")

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        save_metadata(results, str(manifest_path))
        print(f"Manifest saved to: {manifest_path}")

        return results

    def scrape_from_urls(
        self,
        urls: List[str],
        output_dir: str,
        max_files: Optional[int] = None
    ) -> List[Dict]:
        """
        Scrape Figma files from URLs.

        Args:
            urls: List of Figma URLs (e.g., 'https://www.figma.com/file/abcd1234/...')
            output_dir: Directory to save images and metadata
            max_files: Maximum number of files to scrape

        Returns:
            List of metadata dicts
        """
        # Extract file keys from URLs
        file_keys = []
        for url in urls:
            file_key = self._extract_file_key(url)
            if file_key:
                file_keys.append(file_key)
            else:
                print(f"Warning: Could not extract file key from URL: {url}")

        return self.scrape_files(file_keys, output_dir, max_files)

    def _extract_file_key(self, url: str) -> Optional[str]:
        """Extract file key from Figma URL."""
        # URL format: https://www.figma.com/file/{file_key}/...
        # Or: https://www.figma.com/community/file/{file_key}/...
        parts = url.split('/')
        try:
            if 'file' in parts:
                idx = parts.index('file')
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        except Exception:
            pass
        return None


def load_file_keys_from_txt(file_path: str) -> List[str]:
    """
    Load Figma file keys from a text file (one per line).

    Args:
        file_path: Path to text file with file keys or URLs

    Returns:
        List of file keys
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    file_keys = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # If it's a URL, extract the key
        if 'figma.com' in line:
            scraper = FigmaScraper(Config())  # Dummy config for extraction
            file_key = scraper._extract_file_key(line)
            if file_key:
                file_keys.append(file_key)
        else:
            # Assume it's already a file key
            file_keys.append(line)

    return file_keys


if __name__ == "__main__":
    # Test the scraper
    print("Figma Community Scraper")
    print("=" * 50)
    print()

    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    # Load config
    config = load_config()

    # Validate
    if not config.get('figma', 'access_token'):
        print("Error: Figma access token not configured")
        print()
        print("To get a Figma access token:")
        print("1. Go to https://www.figma.com/developers")
        print("2. Log in to your Figma account")
        print("3. Create a new personal access token")
        print("4. Add it to config.json under figma.access_token")
        print()
        exit(1)

    # Create scraper
    scraper = FigmaScraper(config)

    print("Figma scraper ready!")
    print()
    print("Usage:")
    print("  1. Create a file 'figma_urls.txt' with Figma URLs (one per line)")
    print("  2. Run: python -m src.scrapers.figma_scraper")
    print()
    print("Example figma_urls.txt:")
    print("  https://www.figma.com/community/file/1234567890/Design-Name")
    print("  https://www.figma.com/file/abcdef123456/Another-Design")
    print()
