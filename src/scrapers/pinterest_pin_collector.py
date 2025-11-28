"""
Pinterest pin link collector using browser automation.

Automates Pinterest search to collect pin URLs.
Uses Playwright for reliable browser automation.
"""

import time
import random
import asyncio
from pathlib import Path
from typing import List, Optional
from playwright.async_api import async_playwright, Page, Browser

try:
    from .config import Config
except ImportError:
    from config import Config


class PinterestPinCollector:
    """Automated Pinterest pin URL collector using browser automation."""

    def __init__(self, config: Config):
        """
        Initialize Pinterest pin collector.

        Args:
            config: Config object with Pinterest settings
        """
        self.config = config
        self.headless = config.get('pinterest', 'headless') or True
        self.scroll_delay = config.get('pinterest', 'scroll_delay') or 2.5
        self.max_scrolls = config.get('pinterest', 'max_scrolls') or 10
        self.timeout = config.get('pinterest', 'timeout') or 30000

        self.collected_urls = []

    async def search(
        self,
        query: str,
        max_pins: int = 50,
        output_file: Optional[str] = None
    ) -> List[str]:
        """
        Search Pinterest and collect pin URLs.

        Args:
            query: Search query (e.g., "graphic design poster")
            max_pins: Maximum number of pins to collect
            output_file: Path to save collected URLs (optional)

        Returns:
            List of collected pin URLs
        """
        print(f"\n🔍 Searching Pinterest for: '{query}'")
        print(f"📌 Target: {max_pins} pins\n")

        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=self.headless)

            # Create context with realistic user agent
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )

            page = await context.new_page()

            try:
                # Navigate to Pinterest search
                search_url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"
                print(f"🌐 Navigating to: {search_url}")

                # Pinterest is slow to load, use 'domcontentloaded' instead of 'networkidle'
                # and increase timeout to 60 seconds
                await page.goto(search_url, wait_until='domcontentloaded', timeout=60000)

                # Wait for initial content to load
                print("⏳ Waiting for content to load...")
                await asyncio.sleep(5)

                # Check for cookie banner and close it
                await self._handle_cookie_banner(page)

                # Scroll and collect pin URLs
                collected = await self._scroll_and_collect(page, max_pins)

                self.collected_urls = collected
                print(f"\n✅ Collection complete: {len(collected)} pins collected")

                # Save to file if specified
                if output_file:
                    self._save_to_file(collected, output_file)
                    print(f"💾 Saved to: {output_file}")

            except Exception as e:
                print(f"❌ Error during collection: {e}")
                raise

            finally:
                await browser.close()

        return self.collected_urls

    async def _handle_cookie_banner(self, page: Page) -> None:
        """Handle cookie consent banner if present."""
        try:
            # Common Pinterest cookie banner selectors
            cookie_selectors = [
                'button[data-test-id="cookie-accept-button"]',
                'button:has-text("Accept")',
                'button:has-text("Got it")',
            ]

            for selector in cookie_selectors:
                try:
                    button = await page.wait_for_selector(selector, timeout=3000)
                    if button:
                        await button.click()
                        print("🍪 Closed cookie banner")
                        await asyncio.sleep(1)
                        return
                except:
                    continue
        except:
            pass  # No cookie banner found, continue

    async def _scroll_and_collect(self, page: Page, max_pins: int) -> List[str]:
        """
        Scroll page and collect pin URLs.

        Args:
            page: Playwright page object
            max_pins: Maximum number of pins to collect

        Returns:
            List of unique pin URLs
        """
        collected = set()
        scroll_count = 0
        no_new_pins_count = 0

        print("📜 Scrolling and collecting pin URLs...")

        while len(collected) < max_pins and scroll_count < self.max_scrolls:
            # Extract current pin URLs
            new_urls = await self._extract_pin_urls(page)

            previous_count = len(collected)
            collected.update(new_urls)
            new_count = len(collected) - previous_count

            scroll_count += 1
            print(f"  Scroll {scroll_count}/{self.max_scrolls}: "
                  f"Found {new_count} new pins | Total: {len(collected)}/{max_pins}")

            # Check if we found new pins
            if new_count == 0:
                no_new_pins_count += 1
                if no_new_pins_count >= 3:
                    print("  ⚠️  No new pins found after 3 scrolls, stopping...")
                    break
            else:
                no_new_pins_count = 0

            # Stop if we have enough
            if len(collected) >= max_pins:
                break

            # Scroll down with random human-like behavior
            await self._human_like_scroll(page)

            # Random delay to avoid detection
            delay = self.scroll_delay + random.uniform(-0.5, 0.5)
            await asyncio.sleep(delay)

        return list(collected)[:max_pins]

    async def _extract_pin_urls(self, page: Page) -> List[str]:
        """
        Extract pin URLs from current page.

        Args:
            page: Playwright page object

        Returns:
            List of pin URLs found on page
        """
        # Pinterest pin URLs are typically in <a> tags with href="/pin/..."
        try:
            # Extract all links that match Pinterest pin pattern
            urls = await page.evaluate('''() => {
                const links = Array.from(document.querySelectorAll('a[href*="/pin/"]'));
                return links
                    .map(link => link.href)
                    .filter(href => href.includes('pinterest.com/pin/'))
                    .map(href => href.split('?')[0]); // Remove query params
            }''')

            # Filter to unique valid URLs
            valid_urls = []
            for url in urls:
                if url and 'pinterest.com/pin/' in url:
                    # Clean up the URL
                    clean_url = url.split('#')[0].split('?')[0]
                    if clean_url.endswith('/'):
                        clean_url = clean_url[:-1]
                    valid_urls.append(clean_url)

            return list(set(valid_urls))

        except Exception as e:
            print(f"  ⚠️  Error extracting URLs: {e}")
            return []

    async def _human_like_scroll(self, page: Page) -> None:
        """
        Perform human-like scrolling.

        Args:
            page: Playwright page object
        """
        # Random scroll distance
        scroll_distance = random.randint(600, 1200)

        # Scroll in small increments for more natural behavior
        increments = random.randint(3, 5)
        step = scroll_distance // increments

        for _ in range(increments):
            await page.evaluate(f'window.scrollBy(0, {step})')
            await asyncio.sleep(random.uniform(0.1, 0.3))

    def _save_to_file(self, urls: List[str], file_path: str) -> None:
        """
        Save collected URLs to a text file.

        Args:
            urls: List of URLs to save
            file_path: Path to output file
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for url in urls:
                f.write(url + '\n')


async def collect_pins(
    query: str,
    max_pins: int = 50,
    output_file: str = "pinterest_pins_collected.txt",
    config_path: Optional[str] = None
) -> List[str]:
    """
    Convenience function to collect Pinterest pins.

    Args:
        query: Search query
        max_pins: Maximum number of pins to collect
        output_file: Path to save collected URLs
        config_path: Path to config file (optional)

    Returns:
        List of collected pin URLs
    """
    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    config = load_config(config_path)
    collector = PinterestPinCollector(config)

    return await collector.search(query, max_pins, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Pinterest pin URLs")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max-pins", "-n", type=int, default=50, help="Maximum pins to collect")
    parser.add_argument("--output", "-o", default="pinterest_pins_collected.txt", help="Output file")
    parser.add_argument("--headless", type=bool, default=None, help="Run in headless mode")
    parser.add_argument("--config", "-c", default=None, help="Config file path")

    args = parser.parse_args()

    # Load config
    try:
        from config import load_config
    except ImportError:
        from .config import load_config

    config = load_config(args.config)

    # Override headless setting if provided
    if args.headless is not None:
        config.config['pinterest']['headless'] = args.headless

    print("=" * 60)
    print("Pinterest Pin Collector")
    print("=" * 60)

    # Run collector
    collector = PinterestPinCollector(config)
    urls = asyncio.run(collector.search(args.query, args.max_pins, args.output))

    print("\n" + "=" * 60)
    print(f"✅ Successfully collected {len(urls)} pin URLs")
    print("=" * 60)
