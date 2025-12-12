#!/usr/bin/env python3
"""
Quick start script for design dataset collection.

Usage:
    python run_collection.py

This will:
1. Validate your configuration
2. Check for input files (figma_urls.txt, pinterest_urls.txt)
3. Run the complete pipeline
4. Generate statistics
"""

import sys
from pathlib import Path

from config import load_config
from label_pipeline import run_from_files


def main():
    print("=" * 70)
    print("Design Dataset Collection - Quick Start")
    print("=" * 70)
    print()

    # Load config
    print("Step 1: Loading configuration...")
    try:
        config = load_config()
        print("✓ Config loaded")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return

    # Validate config
    print("\nStep 2: Validating configuration...")
    is_valid, errors = config.validate()

    if not is_valid:
        print("✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print()
        print("Please configure API keys in config.json")
        print("See README.md for instructions")
        return

    print("✓ Configuration valid")

    # Check for input files
    print("\nStep 3: Checking for input files...")
    base_path = Path(__file__).parent
    figma_file = base_path / "figma_urls.txt"
    pinterest_file = base_path / "pinterest_urls.txt"

    has_figma = figma_file.exists()
    has_pinterest = pinterest_file.exists()

    if has_figma:
        print(f"✓ Found {figma_file}")
    else:
        print(f"⊘ Not found: {figma_file}")

    if has_pinterest:
        print(f"✓ Found {pinterest_file}")
    else:
        print(f"⊘ Not found: {pinterest_file}")

    if not has_figma and not has_pinterest:
        print()
        print("✗ No input files found!")
        print()
        print("Please create at least one of:")
        print("  - figma_urls.txt (Figma file URLs)")
        print("  - pinterest_urls.txt (Pinterest image URLs)")
        print()
        print("See README.md for examples")
        return

    # Get user confirmation
    print("\nStep 4: Ready to start collection")
    print()
    print("Settings:")
    print(f"  - AI labeling: Yes (Claude Vision)")
    print(f"  - Cost estimate: ~$0.003 per image")
    print(f"  - Output: scraped_data/final_dataset/")
    print()

    response = input("Start collection? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled")
        return

    # Run pipeline
    print()
    print("=" * 70)
    print("Starting Pipeline")
    print("=" * 70)
    print()

    try:
        summary = run_from_files(
            figma_file=str(figma_file) if has_figma else None,
            pinterest_file=str(pinterest_file) if has_pinterest else None,
            output_dir="scraped_data",
            use_ai=True,
            max_images=None  # No limit
        )

        if summary:
            print()
            print("=" * 70)
            print("✓ Collection Complete!")
            print("=" * 70)
            print()
            print(f"Total images: {summary['total_final']}")
            print(f"Output directory: {summary['output_dir']}")
            print()
            print("Next steps:")
            print("  1. Review scraped_data/final_dataset/stats.txt")
            print("  2. Spot-check some images in scraped_data/final_dataset/images/")
            print("  3. Use this dataset to retrain your DTF decoder!")
            print()
        else:
            print("✗ Pipeline failed. Check error messages above.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Partial results may be in scraped_data/")
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
