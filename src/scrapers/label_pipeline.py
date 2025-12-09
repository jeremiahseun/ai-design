"""
Main pipeline for scraping and labeling design images.

Orchestrates:
1. Scraping from Figma and Pinterest
2. Labeling with metadata-based or AI-based methods
3. Dataset preparation for DTF training
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures
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
        target_size: tuple = (256, 256),
        save_interval: int = 100
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
            labeled_dir, # Pass labeled_dir
            use_ai=use_ai_labeling,
            ai_threshold=ai_threshold,
            save_interval=save_interval # Pass save_interval
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
            target_size=target_size,
            save_interval=save_interval # Pass save_interval
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
        labeled_dir: Path,
        use_ai: bool = True,
        ai_threshold: float = 0.7,
        save_interval: int = 100
    ) -> List[Dict]:
        """
        Label images using metadata and/or AI, with resume capability and periodic saving.
        """
        partial_manifest_path = labeled_dir / "partial_labeled_metadata.json"
        labeled = []
        start_index = 0

        # Attempt to resume from partial data
        if partial_manifest_path.exists():
            print(f"Resuming labeling from partial manifest: {partial_manifest_path}")
            try:
                labeled = load_metadata(str(partial_manifest_path))
                start_index = len(labeled)
                print(f"Resumed from {start_index} previously labeled images.")
            except Exception as e:
                print(f"Error loading partial manifest: {e}. Starting from scratch.")
                labeled = []
                start_index = 0

        # Ensure metadata_list is long enough for resumed data
        if start_index > 0 and start_index < len(metadata_list):
            # If resuming, ensure the metadata_list matches the resumed data
            # This is a simple check, more robust would be to compare content
            if len(labeled) != start_index:
                print("Warning: Mismatch between partial manifest length and expected start index. Starting from scratch.")
                labeled = []
                start_index = 0
            else:
                # Skip already processed items in metadata_list
                metadata_list = metadata_list[start_index:]
                print(f"Processing remaining {len(metadata_list)} images.")
        elif start_index >= len(metadata_list):
            print("All images already labeled in partial manifest. Skipping labeling phase.")
            return labeled # All done, return existing labeled data

        # First pass: Metadata labeling
        print("\nPhase 1: Metadata-based labeling...")
        for i, metadata in enumerate(metadata_list, 1):
            current_total_index = start_index + i - 1
            if current_total_index % 50 == 0:
                print(f"  Progress: {current_total_index}/{len(metadata_list) + start_index}")

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

            # Periodic save
            if current_total_index > 0 and current_total_index % save_interval == 0:
                print(f"  Saving partial labeled metadata ({current_total_index} items)...")
                save_metadata(labeled, str(partial_manifest_path))

        print(f"✓ Metadata labeling complete: {len(labeled)} images")

        # Second pass: AI labeling for low-confidence items
        if use_ai:
            print("\nPhase 2: AI-based labeling for low-confidence items...")

            # Filter items below threshold
            # Need to re-filter based on the full 'labeled' list, not just remaining
            low_confidence_indices = [
                idx for idx, item in enumerate(labeled)
                if item.get('confidence', 0) < ai_threshold
            ]

            # If resuming, we need to know which low_confidence items were already processed by AI
            # For simplicity, we'll re-process all low_confidence items if resuming,
            # or implement a more complex state tracking. For now, re-process.

            if low_confidence_indices:
                print(f"  {len(low_confidence_indices)} images below confidence threshold for AI labeling.")
                print(f"  Estimated cost for AI labeling: ${self._estimate_ai_cost(len(low_confidence_indices)):.2f}")

                # Initialize AI labeler
                if self.ai_labeler is None:
                    self.ai_labeler = AILabeler(self.config)

                # Use ThreadPoolExecutor for parallel API calls
                max_workers = 10 # Adjust based on your quota and CPU cores
                processed_count = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Create a list of (original_idx, item) tuples for items to be processed
                    items_to_process = [(original_idx, labeled[original_idx]) for original_idx in low_confidence_indices if original_idx >= start_index]

                    future_to_original_idx = {
                        executor.submit(
                            self.ai_labeler.label_image,
                            item['image_path'],
                            item
                        ): original_idx
                        for original_idx, item in items_to_process
                    }

                    for future in concurrent.futures.as_completed(future_to_original_idx):
                        original_idx = future_to_original_idx[future]
                        processed_count += 1
                        current_ai_index = start_index + processed_count - 1 # Adjust for overall progress

                        try:
                            ai_labels = future.result()
                            labeled[original_idx].update(ai_labels)
                        except Exception as exc:
                            print(f"  ⚠️  Error labeling image {labeled[original_idx]['image_path']}: {exc}")
                            # Fallback already handled within label_image, so just log here

                        if processed_count % 10 == 0:
                            print(f"  Progress: {processed_count}/{len(items_to_process)} (Total AI: {current_ai_index})")

                        # Periodic save during AI labeling
                        if current_ai_index > 0 and current_ai_index % save_interval == 0:
                            print(f"  Saving partial labeled metadata (AI phase, {current_ai_index} items)...")
                            save_metadata(labeled, str(partial_manifest_path))

                print(f"✓ AI labeling complete")

                # Show stats
                if self.ai_labeler:
                    stats = self.ai_labeler.get_stats()
                    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            else:
                print("  All items have high confidence, skipping AI labeling")

        # Final save of labeled metadata (will be overwritten by run method)
        # and cleanup of partial file
        if partial_manifest_path.exists():
            os.remove(partial_manifest_path)
            print(f"Cleaned up partial manifest: {partial_manifest_path}")

        return labeled

    def _prepare_final_dataset(
        self,
        labeled_metadata: List[Dict],
        output_dir: str,
        target_size: tuple = (256, 256),
        save_interval: int = 100
    ) -> List[Dict]:
        """
        Prepare final dataset with resized images and cleaned metadata, with resume capability and periodic saving.
        """
        output_dir = Path(output_dir)
        images_dir = ensure_dir(output_dir / "images")
        partial_final_manifest_path = output_dir / "partial_final_metadata.json"

        final_metadata = []
        start_index = 0

        # Attempt to resume from partial data
        if partial_final_manifest_path.exists():
            print(f"Resuming final dataset preparation from partial manifest: {partial_final_manifest_path}")
            try:
                final_metadata = load_metadata(str(partial_final_manifest_path))
                start_index = len(final_metadata)
                print(f"Resumed from {start_index} previously prepared images.")
            except Exception as e:
                print(f"Error loading partial final manifest: {e}. Starting from scratch.")
                final_metadata = []
                start_index = 0

        # Ensure labeled_metadata is long enough for resumed data
        if start_index > 0 and start_index < len(labeled_metadata):
            # Skip already processed items in labeled_metadata
            labeled_metadata = labeled_metadata[start_index:]
            print(f"Preparing remaining {len(labeled_metadata)} images.")
        elif start_index >= len(labeled_metadata):
            print("All images already prepared in partial final manifest. Skipping preparation phase.")
            return final_metadata # All done, return existing final data

        print(f"\nPreparing final dataset ({target_size[0]}x{target_size[1]})...")

        for i, item in enumerate(labeled_metadata, 1):
            current_total_index = start_index + i - 1
            if current_total_index % 50 == 0:
                print(f"  Progress: {current_total_index}/{len(labeled_metadata) + start_index}")

            try:
                # Load and resize image
                image = Image.open(item['image_path']).convert('RGB')
                resized = resize_image(image, target_size)

                # Save resized image
                new_filename = f"design_{current_total_index:05d}.png" # Use current_total_index for filename
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

                # Periodic save
                if current_total_index > 0 and current_total_index % save_interval == 0:
                    print(f"  Saving partial final metadata ({current_total_index} items)...")
                    save_metadata(final_metadata, str(partial_final_manifest_path))

            except Exception as e:
                print(f"  Warning: Failed to process {item.get('image_path')}: {e}")
                continue

        print(f"✓ Final dataset prepared: {len(final_metadata)} images")

        # Clean up partial file
        if partial_final_manifest_path.exists():
            os.remove(partial_final_manifest_path)
            print(f"Cleaned up partial final manifest: {partial_final_manifest_path}")

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
