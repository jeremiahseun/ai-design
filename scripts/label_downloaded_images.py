"""
Processes and labels an existing directory of downloaded images to create a final dataset.

This script is for when you have already downloaded images and want to run the
labeling and dataset preparation pipeline on them.
"""

import os
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scrapers.config import load_config
from src.scrapers.label_pipeline import LabelPipeline
from src.scrapers.utils import save_metadata, ensure_dir

def main(input_dir: str, output_dir: str, use_ai: bool, max_images: int, save_interval: int = 100):
    """
    Main function to process existing images.
    """
    print("=" * 70)
    print("Labeling Pipeline for Existing Images")
    print("=" * 70)

    # 1. Load Config
    config = load_config()
    is_valid, errors = config.validate()
    if not use_ai:
        print("INFO: AI labeling is disabled. Only metadata labeling will be used.")
    elif not is_valid:
        print("⚠️  Warning: API keys not configured. AI labeling will fail.")
        print("   Falling back to metadata-only labeling.")
        use_ai = False

    # 2. Scan for images
    image_dir = Path(input_dir)
    if not image_dir.exists():
        print(f"❌ Error: Input directory not found at '{image_dir}'")
        return

    print(f"🔎 Scanning for images in '{image_dir}'...")
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    all_image_paths = [p for p in image_dir.glob('**/*') if p.suffix.lower() in image_extensions]

    if not all_image_paths:
        print("❌ No images found in the input directory.")
        return

    print(f"✅ Found {len(all_image_paths)} images.")

    if max_images and len(all_image_paths) > max_images:
        print(f"Limiting to {max_images} images.")
        all_image_paths = all_image_paths[:max_images]

    # 3. Create initial metadata list
    initial_metadata = []
    for img_path in all_image_paths:
        initial_metadata.append({
            'image_path': str(img_path),
            'source': 'pinterest_downloaded',
            'title': img_path.stem,
            'description': '',
            'tags': [],
            'url': ''
        })

    # Setup directories
    final_dir = ensure_dir(Path(output_dir) / "final_dataset")
    labeled_dir = ensure_dir(Path(output_dir) / "labeled")

    # 4. Run the labeling and preparation pipeline
    pipeline = LabelPipeline(config)

    # Step 4a: Label images
    print("\n" + "=" * 70)
    print("STEP 1: Labeling Images")
    print("=" * 70)
    labeled_metadata = pipeline._label_images(
        initial_metadata,
        labeled_dir=labeled_dir,
        use_ai=use_ai,
        ai_threshold=0.7,
        save_interval=save_interval
    )

    # Step 4b: Prepare final dataset
    print("\n" + "=" * 70)
    print("STEP 2: Preparing Final Dataset")
    print("=" * 70)
    final_metadata = pipeline._prepare_final_dataset(
        labeled_metadata,
        output_dir,
        target_size=(256, 256),
        save_interval=save_interval
    )

    # 5. Save final manifest and stats
    final_manifest_path = final_dir / "metadata.json"
    save_metadata(final_metadata, str(final_manifest_path))

    stats = format_dataset_stats(final_metadata)
    stats_path = final_dir / "stats.txt"
    with open(stats_path, 'w') as f:
        f.write(stats)

    print("\n" + "=" * 70)
    print("✅ Pipeline Complete!")
    print("=" * 70)
    print(f"Final dataset created at: {output_dir}")
    print(f"Total images processed: {len(final_metadata)}")
    print(f"Review stats at: {stats_path}")
    print("\nYou are now ready for Phase 1, Step 2: Train the Decoder on Real Data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label a directory of existing images.")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='src/scrapers/images/',
        help="Directory containing the downloaded images."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/real_designs',
        help="Directory to save the final labeled dataset."
    )
    parser.add_argument(
        '--no-ai',
        action='store_true',
        help="Disable AI labeling and use only metadata heuristics."
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help="Maximum number of images to process."
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help="Interval at which to save partial progress (number of images)."
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, not args.no_ai, args.max_images, args.save_interval)
