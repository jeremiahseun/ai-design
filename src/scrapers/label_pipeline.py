"""
Main pipeline for scraping and labeling design images.

Orchestrates:
1. Scraping from Figma and Pinterest
2. Labeling with metadata-based or AI-based methods
3. Dataset preparation for DTF training
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

try:
    # Try relative imports (when used as package)
    from .config import Config, load_config
    from .figma_scraper import FigmaScraper, load_file_keys_from_txt
    from .pinterest_scraper import PinterestScraper, load_urls_from_txt
    from .metadata_labeler import MetadataLabeler
    from .ai_labeler import AILabeler
    from .utils import (
        ensure_dir,
        save_metadata,
        load_metadata,
        resize_image,
        format_dataset_stats,
        validate_label
    )
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import Config, load_config
    from figma_scraper import FigmaScraper, load_file_keys_from_txt
    from pinterest_scraper import PinterestScraper, load_urls_from_txt
    from metadata_labeler import MetadataLabeler
    from ai_labeler import AILabeler
    from utils import (
        ensure_dir,
        save_metadata,
        load_metadata,
        resize_image,
        format_dataset_stats,
        validate_label
    )


class LabelPipeline:
    """Complete pipeline for scraping and labeling designs."""

    def __init__(self, config: Config):
        """
        Initialize pipeline.

        Args:
            config: Config object
        """
        self.config = config
        self.metadata_labeler = MetadataLabeler()
        self.ai_labeler = None  # Lazy init

    def run(
        self,
        figma_file_keys: Optional[List[str]] = None,
        pinterest_urls: Optional[List[str]] = None,
        output_dir: str = "scraped_data",
        use_ai_labeling: bool = True,
        ai_threshold: float = 0.7,
        max_images: Optional[int] = None,
        target_size: tuple = (256, 256)
    ) -> Dict:
        """
        Run complete pipeline.

        Args:
            figma_file_keys: List of Figma file keys
            pinterest_urls: List of Pinterest URLs
            output_dir: Output directory
            use_ai_labeling: Whether to use AI labeling
            ai_threshold: Confidence threshold for AI labeling (below this, use metadata)
            max_images: Maximum total images
            target_size: Target image size

        Returns:
            Summary dict with statistics
        """
        print("=" * 70)
        print("Design Dataset Collection Pipeline")
        print("=" * 70)
        print()

        # Setup directories
        raw_dir = ensure_dir(f"{output_dir}/raw")
        figma_dir = ensure_dir(f"{output_dir}/raw/figma")
        pinterest_dir = ensure_dir(f"{output_dir}/raw/pinterest")
        labeled_dir = ensure_dir(f"{output_dir}/labeled")
        final_dir = ensure_dir(f"{output_dir}/final_dataset")

        all_metadata = []

        # Step 1: Scrape from Figma
        if figma_file_keys:
            print("\n" + "=" * 70)
            print("STEP 1: Scraping from Figma")
            print("=" * 70)

            figma_scraper = FigmaScraper(self.config)
            figma_metadata = figma_scraper.scrape_files(
                figma_file_keys,
                str(figma_dir),
                max_files=max_images
            )
            all_metadata.extend(figma_metadata)

            print(f"\n✓ Figma scraping complete: {len(figma_metadata)} images")

        # Step 2: Scrape from Pinterest
        if pinterest_urls:
            print("\n" + "=" * 70)
            print("STEP 2: Scraping from Pinterest")
            print("=" * 70)

            pinterest_scraper = PinterestScraper(self.config)
            pinterest_metadata = pinterest_scraper.scrape_from_urls(
                pinterest_urls,
                str(pinterest_dir)
            )
            all_metadata.extend(pinterest_metadata)

            print(f"\n✓ Pinterest scraping complete: {len(pinterest_metadata)} images")

        if not all_metadata:
            print("\n⚠️  No images scraped. Check your input files.")
            return {}

        # Limit total images
        if max_images and len(all_metadata) > max_images:
            print(f"\nLimiting to {max_images} images...")
            all_metadata = all_metadata[:max_images]

        # Step 3: Label images
        print("\n" + "=" * 70)
        print("STEP 3: Labeling Images")
        print("=" * 70)

        labeled_metadata = self._label_images(
            all_metadata,
            use_ai=use_ai_labeling,
            ai_threshold=ai_threshold
        )

        # Save labeled metadata
        labeled_manifest_path = labeled_dir / "manifest.json"
        save_metadata(labeled_metadata, str(labeled_manifest_path))
        print(f"\n✓ Labeled metadata saved to: {labeled_manifest_path}")

        # Step 4: Prepare final dataset
        print("\n" + "=" * 70)
        print("STEP 4: Preparing Final Dataset")
        print("=" * 70)

        final_metadata = self._prepare_final_dataset(
            labeled_metadata,
            str(final_dir),
            target_size=target_size
        )

        # Save final manifest
        final_manifest_path = final_dir / "metadata.json"
        save_metadata(final_metadata, str(final_manifest_path))

        # Step 5: Generate statistics
        print("\n" + "=" * 70)
        print("STEP 5: Dataset Statistics")
        print("=" * 70)

        stats = format_dataset_stats(final_metadata)
        print(stats)

        # Save stats
        stats_path = final_dir / "stats.txt"
        with open(stats_path, 'w') as f:
            f.write(stats)

        # Summary
        summary = {
            'total_scraped': len(all_metadata),
            'total_labeled': len(labeled_metadata),
            'total_final': len(final_metadata),
            'sources': {
                'figma': len([m for m in all_metadata if m['source'] == 'figma']),
                'pinterest': len([m for m in all_metadata if m['source'] == 'pinterest']),
            },
            'labeling_methods': {
                'metadata': len([m for m in labeled_metadata if m.get('method') == 'metadata_heuristic']),
                'ai': len([m for m in labeled_metadata if m.get('method') == 'claude_vision']),
                'fallback': len([m for m in labeled_metadata if m.get('method') == 'metadata_fallback']),
            },
            'output_dir': str(final_dir)
        }

        print("\n" + "=" * 70)
        print("Pipeline Complete!")
        print("=" * 70)
        print(f"\nFinal dataset: {final_dir}")
        print(f"Total images: {len(final_metadata)}")
        print(f"Ready for DTF decoder training!")
        print()

        return summary

    def _label_images(
        self,
        metadata_list: List[Dict],
        use_ai: bool = True,
        ai_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Label images using metadata and/or AI.

        Args:
            metadata_list: List of metadata dicts from scrapers
            use_ai: Whether to use AI labeling
            ai_threshold: Confidence threshold for using AI

        Returns:
            List of labeled metadata dicts
        """
        labeled = []

        # First pass: Metadata labeling
        print("\nPhase 1: Metadata-based labeling...")
        for i, metadata in enumerate(metadata_list, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(metadata_list)}")

            # Load image
            image = Image.open(metadata['image_path']).convert('RGB')

            # Metadata labeling
            labels = self.metadata_labeler.label_from_metadata(
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                tags=metadata.get('tags', []),
                width=metadata.get('width'),
                height=metadata.get('height'),
                image=image
            )

            # Merge with original metadata
            labeled_item = {**metadata, **labels}
            labeled.append(labeled_item)

        print(f"✓ Metadata labeling complete: {len(labeled)} images")

        # Second pass: AI labeling for low-confidence items
        if use_ai:
            print("\nPhase 2: AI-based labeling for low-confidence items...")

            # Filter items below threshold
            low_confidence = [
                (i, item) for i, item in enumerate(labeled)
                if item.get('confidence', 0) < ai_threshold
            ]

            if low_confidence:
                print(f"  {len(low_confidence)} images below confidence threshold")
                print(f"  Estimated cost: ${self._estimate_ai_cost(len(low_confidence)):.2f}")

                # Initialize AI labeler
                if self.ai_labeler is None:
                    self.ai_labeler = AILabeler(self.config)

                # Re-label with AI
                for idx, (original_idx, item) in enumerate(low_confidence, 1):
                    if idx % 10 == 0:
                        print(f"  Progress: {idx}/{len(low_confidence)}")

                    ai_labels = self.ai_labeler.label_image(
                        item['image_path'],
                        metadata=item
                    )

                    # Update in place
                    labeled[original_idx].update(ai_labels)

                print(f"✓ AI labeling complete")

                # Show stats
                if self.ai_labeler:
                    stats = self.ai_labeler.get_stats()
                    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            else:
                print("  All items have high confidence, skipping AI labeling")

        return labeled

    def _prepare_final_dataset(
        self,
        labeled_metadata: List[Dict],
        output_dir: str,
        target_size: tuple = (256, 256)
    ) -> List[Dict]:
        """
        Prepare final dataset with resized images and cleaned metadata.

        Args:
            labeled_metadata: List of labeled metadata dicts
            output_dir: Output directory
            target_size: Target image size

        Returns:
            List of final metadata dicts
        """
        output_dir = Path(output_dir)
        images_dir = ensure_dir(output_dir / "images")

        final_metadata = []

        print(f"\nPreparing final dataset ({target_size[0]}x{target_size[1]})...")

        for i, item in enumerate(labeled_metadata, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(labeled_metadata)}")

            try:
                # Load and resize image
                image = Image.open(item['image_path']).convert('RGB')
                resized = resize_image(image, target_size)

                # Save resized image
                new_filename = f"design_{i:05d}.png"
                new_path = images_dir / new_filename
                resized.save(new_path)

                # Create clean metadata
                clean_meta = {
                    'filename': new_filename,
                    'image_path': str(new_path),
                    'v_Goal': item['v_Goal'],
                    'v_Format': item['v_Format'],
                    'v_Tone': item['v_Tone'],
                    'source': item['source'],
                    'original_url': item.get('original_url', ''),
                    'labeling_method': item.get('method', 'unknown'),
                    'confidence': item.get('confidence', 0.0),
                }

                final_metadata.append(clean_meta)

            except Exception as e:
                print(f"  Warning: Failed to process {item.get('image_path')}: {e}")
                continue

        print(f"✓ Final dataset prepared: {len(final_metadata)} images")

        return final_metadata

    def _estimate_ai_cost(self, num_images: int) -> float:
        """Estimate AI labeling cost."""
        cost_per_image = 0.003
        return num_images * cost_per_image


def run_from_files(
    config_path: Optional[str] = None,
    figma_file: Optional[str] = None,
    pinterest_file: Optional[str] = None,
    output_dir: str = "scraped_data",
    use_ai: bool = True,
    max_images: Optional[int] = None
) -> Dict:
    """
    Run pipeline from input files.

    Args:
        config_path: Path to config.json
        figma_file: Path to text file with Figma file keys (one per line)
        pinterest_file: Path to text file with Pinterest URLs (one per line)
        output_dir: Output directory
        use_ai: Whether to use AI labeling
        max_images: Maximum total images

    Returns:
        Summary dict
    """
    # Load config
    config = load_config(config_path)

    # Validate config
    is_valid, errors = config.validate()
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  ✗ {error}")
        print("\nPlease configure API keys in config.json")
        return {}

    # Load file keys/URLs
    figma_keys = None
    if figma_file:
        figma_keys = load_file_keys_from_txt(figma_file)
        print(f"Loaded {len(figma_keys)} Figma file keys from {figma_file}")

    pinterest_urls = None
    if pinterest_file:
        pinterest_urls = load_urls_from_txt(pinterest_file)
        print(f"Loaded {len(pinterest_urls)} Pinterest URLs from {pinterest_file}")

    if not figma_keys and not pinterest_urls:
        print("Error: No input files provided")
        print("Provide --figma-file and/or --pinterest-file")
        return {}

    # Run pipeline
    pipeline = LabelPipeline(config)
    summary = pipeline.run(
        figma_file_keys=figma_keys,
        pinterest_urls=pinterest_urls,
        output_dir=output_dir,
        use_ai_labeling=use_ai,
        max_images=max_images
    )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Design dataset collection pipeline")
    parser.add_argument('--config', type=str, help='Path to config.json')
    parser.add_argument('--figma-file', type=str, help='Text file with Figma file keys')
    parser.add_argument('--pinterest-file', type=str, help='Text file with Pinterest URLs')
    parser.add_argument('--output', type=str, default='scraped_data', help='Output directory')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI labeling')
    parser.add_argument('--max-images', type=int, help='Maximum total images')

    args = parser.parse_args()

    summary = run_from_files(
        config_path=args.config,
        figma_file=args.figma_file,
        pinterest_file=args.pinterest_file,
        output_dir=args.output,
        use_ai=not args.no_ai,
        max_images=args.max_images
    )

    if summary:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(json.dumps(summary, indent=2))
