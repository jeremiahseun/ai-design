
import os
import glob
import argparse
from src.intelligence.design_analyzer import DesignAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Run design analysis on scraped images.")
    parser.add_argument("--limit", type=int, default=20, help="Number of images to analyze")
    parser.add_argument("--output", type=str, default="data/design_patterns_raw.json", help="Output JSON file")
    args = parser.parse_args()

    # directories to search
    base_dirs = [
        "src/scrapers/images",
        "src/scrapers/scraped_data/raw/pinterest",
        "src/scrapers/scraped_data/images"
    ]

    all_images = []
    for d in base_dirs:
        if os.path.exists(d):
            # recursive search for images
            found = glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
            print(f"Found {len(found)} images in {d}")
            all_images.extend(found)

    # deduplicate
    all_images = list(set(all_images))
    print(f"Total unique images found: {len(all_images)}")

    if not all_images:
        print("No images found to analyze!")
        return

    # Initialize analyzer
    try:
        analyzer = DesignAnalyzer()
        # Analyze batch (analyzer handles skipping existing)
        # We slice to the limit, but analyzer checks validity.
        # Actually analyzer.analyze_batch takes a list.
        # If we want to ADD 20 new ones, it's tricky because analyzer skips existing inside.
        # So passing all images is safer, the analyzer will skip the ones done and do 'limit' new ones?
        # The analyzer.analyze_batch iterates over all provided.
        # We should probably filter before passing if we want to strictly limit the RUN time.

        # Simple logic: Read existing, exclude from list, then take top N
        import json
        existing = []
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                try:
                    existing = json.load(f)
                except:
                    pass

        existing_paths = {r.get('file_path') for r in existing}
        to_process = [p for p in all_images if p not in existing_paths]

        print(f"Images already analyzed: {len(existing_paths)}")
        print(f"Images remaining to process: {len(to_process)}")

        batch = to_process[:args.limit]
        if not batch:
             print("No new images to analyze.")
             return

        analyzer.analyze_batch(batch, args.output)

    except Exception as e:
        print(f"Job failed: {e}")

if __name__ == "__main__":
    main()
