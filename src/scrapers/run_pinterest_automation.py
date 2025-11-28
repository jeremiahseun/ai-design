"""
End-to-end Pinterest automation workflow.

This script automates the complete process:
1. Search Pinterest and collect pin URLs
2. Extract image URLs from collected pins
3. Save results to files
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List

try:
    from .config import load_config
    from .pinterest_pin_collector import PinterestPinCollector
    from .pinterest_scrapers import get_image_url
except ImportError:
    from config import load_config
    from pinterest_pin_collector import PinterestPinCollector
    from pinterest_scrapers import get_image_url


async def run_automation(
    query: str,
    max_pins: int = 50,
    pins_output: str = "pinterest_pins_collected.txt",
    images_output: str = "pinterest_images.txt",
    headless: bool = True,
    rate_limit_delay: float = 2.0
) -> tuple[List[str], List[str]]:
    """
    Run complete Pinterest automation workflow.

    Args:
        query: Search query
        max_pins: Maximum number of pins to collect
        pins_output: File to save pin URLs
        images_output: File to save image URLs
        headless: Run browser in headless mode
        rate_limit_delay: Delay between image extraction requests

    Returns:
        (list_of_pin_urls, list_of_image_urls)
    """
    print("=" * 70)
    print("Pinterest Automation Workflow".center(70))
    print("=" * 70)
    print()

    # Step 1: Collect pin URLs
    print("STEP 1: Collecting Pin URLs")
    print("-" * 70)

    config = load_config()
    if headless is not None:
        config.config['pinterest']['headless'] = headless

    collector = PinterestPinCollector(config)
    pin_urls = await collector.search(query, max_pins, pins_output)

    if not pin_urls:
        print("❌ No pins collected. Exiting.")
        return [], []

    print()

    # Step 2: Extract image URLs from pins
    print("STEP 2: Extracting Image URLs from Pins")
    print("-" * 70)
    print(f"📋 Processing {len(pin_urls)} pins")
    print(f"⏱️  Rate limit: {rate_limit_delay}s between requests\n")

    image_urls = []
    success_count = 0

    for idx, pin_url in enumerate(pin_urls, 1):
        print(f"[{idx}/{len(pin_urls)}] 🔎 {pin_url}")

        image_url = get_image_url(pin_url)

        if image_url:
            print(f"         ✅ {image_url}")
            image_urls.append(image_url)
            success_count += 1
        else:
            print(f"         ❌ Failed to extract image URL")

        # Rate limiting
        if idx < len(pin_urls):
            await asyncio.sleep(rate_limit_delay)
        print()

    # Step 3: Save image URLs to file
    print("STEP 3: Saving Results")
    print("-" * 70)

    output_path = Path(images_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for url in image_urls:
            f.write(url + '\n')

    print(f"💾 Image URLs saved to: {images_output}")
    print()

    # Summary
    print("=" * 70)
    print("Summary".center(70))
    print("=" * 70)
    print(f"📌 Pin URLs collected:       {len(pin_urls):>4}")
    print(f"🖼️  Image URLs extracted:     {len(image_urls):>4}")
    print(f"✅ Success rate:             {success_count}/{len(pin_urls)} ({100*success_count/len(pin_urls) if pin_urls else 0:.1f}%)")
    print()
    print(f"📁 Pin URLs file:    {pins_output}")
    print(f"📁 Image URLs file:  {images_output}")
    print("=" * 70)

    return pin_urls, image_urls


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Pinterest Pin Link Automation - End-to-End Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_pinterest_automation.py --query "graphic design poster" --max-pins 20

  # Watch browser in action (non-headless)
  python run_pinterest_automation.py --query "minimal design" --max-pins 10 --visible

  # Custom output files and rate limiting
  python run_pinterest_automation.py -q "ui design" -n 30 --pins my_pins.txt --images my_images.txt --delay 3.0
        """
    )

    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Search query (e.g., 'graphic design poster')"
    )

    parser.add_argument(
        "--max-pins", "-n",
        type=int,
        default=50,
        help="Maximum number of pins to collect (default: 50)"
    )

    parser.add_argument(
        "--pins",
        default="pinterest_pins_collected.txt",
        help="Output file for pin URLs (default: pinterest_pins_collected.txt)"
    )

    parser.add_argument(
        "--images",
        default="pinterest_images.txt",
        help="Output file for image URLs (default: pinterest_images.txt)"
    )

    parser.add_argument(
        "--visible",
        action="store_true",
        help="Run browser in visible mode (not headless)"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between image extraction requests in seconds (default: 2.0)"
    )

    args = parser.parse_args()

    # Run the automation
    try:
        pin_urls, image_urls = asyncio.run(run_automation(
            query=args.query,
            max_pins=args.max_pins,
            pins_output=args.pins,
            images_output=args.images,
            headless=not args.visible,
            rate_limit_delay=args.delay
        ))

        if image_urls:
            print("\n✅ Automation completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Automation completed but no images were extracted.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Automation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Automation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
