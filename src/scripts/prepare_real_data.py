import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrapers.label_pipeline import LabelPipeline
from scrapers.config import load_config

def prepare_from_local_images(images_dir: str, output_dir: str):
    """
    Runs the labeling and dataset preparation pipeline on a directory of existing images.
    """
    print("=" * 70)
    print("Prepare Real Dataset from Local Images")
    print("=" * 70)
    print()

    # Load config
    config = load_config()
    is_valid, errors = config.validate()
    if not is_valid:
        print("✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"✗ Error: Images directory not found at '{images_dir}'")
        return

    # 1. Find all images and create initial metadata
    print(f"1. Searching for images in '{images_dir}'...")
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    image_paths = [p for p in images_path.rglob('*') if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print("✗ No images found.")
        return
        
    print(f"✓ Found {len(image_paths)} images.")

    all_metadata = []
    for i, img_path in enumerate(image_paths):
        all_metadata.append({
            'image_path': str(img_path),
            'title': img_path.stem,
            'source': 'local_import',
            'url': img_path.resolve().as_uri()
        })

    # 2. Run the labeling and preparation pipeline
    pipeline = LabelPipeline(config)
    
    labeled_dir = Path(output_dir) / "labeled"
    final_dir = Path(output_dir) / "final_dataset"
    
    labeled_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Step 3 (from original pipeline): Label images
    print("\n" + "=" * 70)
    print("STEP 2: Labeling Images")
    print("=" * 70)

    labeled_metadata = pipeline._label_images(
        all_metadata,
        labeled_dir=labeled_dir,
        use_ai=True
    )
    
    labeled_manifest_path = labeled_dir / "manifest.json"
    with open(labeled_manifest_path, 'w') as f:
        json.dump(labeled_metadata, f, indent=2)
    print(f"\n✓ Labeled metadata saved to: {labeled_manifest_path}")

    # Step 4 (from original pipeline): Prepare final dataset
    print("\n" + "=" * 70)
    print("STEP 3: Preparing Final Dataset")
    print("=" * 70)

    final_metadata = pipeline._prepare_final_dataset(
        labeled_metadata,
        str(final_dir)
    )

    # Save final manifest
    final_manifest_path = final_dir / "metadata.json"
    with open(final_manifest_path, 'w') as f:
        json.dump(final_metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ Pipeline Complete!")
    print(f"Final dataset created at: {final_dir}")
    print(f"Total images processed: {len(final_metadata)}")
    print("=" * 70)

if __name__ == '__main__':
    # Note: This script assumes it is run from the project root.
    # The paths are relative to the root.
    images_directory = 'src/scrapers/images'
    output_directory = 'scraped_data'
    prepare_from_local_images(images_directory, output_directory)