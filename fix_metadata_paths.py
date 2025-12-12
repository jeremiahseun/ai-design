import json
from pathlib import Path

def fix_paths():
    """
    Reads the existing metadata.json, fixes the 'image_path' to be relative,
    and saves a new corrected file without re-running any labeling.
    """
    metadata_path = Path("scraped_data/final_dataset/metadata.json")
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        return

    print(f"Reading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    fixed_count = 0
    for entry in metadata:
        if 'image_path' in entry:
            p = Path(entry['image_path'])
            # Make the path relative, e.g., "scraped_data/final_dataset/images/..." -> "images/..."
            if len(p.parts) > 2:
                relative_path = Path(*p.parts[-2:]) # Takes the last two parts, e.g., ('images', 'filename.png')
                entry['image_path'] = str(relative_path)
                fixed_count += 1

    output_path = metadata_path.parent / "metadata_fixed.json"
    print(f"Fixed {fixed_count} paths.")
    print(f"Saving corrected metadata to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\nDone. Please perform the following steps:")
    print(f"1. (Optional) Rename '{metadata_path}' to '{metadata_path}.bak'")
    print(f"2. Rename '{output_path}' to '{metadata_path}'")
    print("3. Proceed with zipping the 'final_dataset' directory and uploading to Kaggle.")

if __name__ == '__main__':
    fix_paths()
