
import os
import sys
from pathlib import Path
import json
from datasets import load_dataset, get_dataset_config_names
from PIL import Image
from tqdm import tqdm
import shutil

# Add src to path so we can use our project's modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
# The name of the dataset on Hugging Face
CGL_DATASET_NAME = "creative-graphic-design/CGL-Dataset"

# The directory where we will save the new dataset
# This will be created inside your project structure
OUTPUT_DIR = Path("data/imported/cgl_dataset")
IMAGE_DIR = OUTPUT_DIR / "images"

# The path to your existing metadata file
EXISTING_METADATA_PATH = Path("data/real_designs/final_dataset/metadata.json")

# The path for the final, merged metadata file
MERGED_METADATA_PATH = Path("data/final_merged_dataset.json")


# --- Metadata Mapping ---
# This section requires inspection of the CGL-Dataset's features.
# We will map the CGL dataset's categories to your existing schema (v_Goal, v_Format, v_Tone).
# I am making some educated guesses here based on the dataset's description.
# You may need to adjust this mapping in the Kaggle notebook once you inspect the data.

# To inspect the data on Kaggle, you can add a cell with:
# from datasets import load_dataset
# dataset = load_dataset("creative-graphic-design/CGL-Dataset", name="CGL-Dataset")
# print(dataset['train'][0])

# Based on the dataset card, it has 'category' and 'element' information.
# Let's assume a simple mapping for now.
CGL_CATEGORY_TO_V_GOAL = {
    # This is a guess. Replace with actual categories from the dataset.
    "food": "product",
    "education": "education",
    "movie": "event",
    "game": "promotion",
    "travel": "promotion",
    "cosmetics": "product",
    "sports": "event",
    "conference": "event",
    # Add more mappings as you discover categories.
    "default": "other" # A fallback category
}

def get_v_goal(cgl_category: str) -> str:
    """Maps a CGL category to a v_Goal."""
    return CGL_CATEGORY_TO_V_GOAL.get(cgl_category.lower(), CGL_CATEGORY_TO_V_GOAL["default"])

def process_dataset():
    """
    Downloads the CGL-Dataset, processes it, and saves it in the project's format.
    """
    print(f"Starting download and processing for '{CGL_DATASET_NAME}'...")

    # 1. Load dataset from Hugging Face
    # The 'datasets' library will handle caching, so it won't re-download every time.
    try:
        # First, try to get the configuration names
        configs = get_dataset_config_names(CGL_DATASET_NAME)
        print(f"Available configurations: {configs}")
        # Use the first config name, which is likely the main one
        dataset = load_dataset(CGL_DATASET_NAME, name=configs[0])
    except Exception as e:
        print(f"Could not load dataset with specific config. Trying default. Error: {e}")
        dataset = load_dataset(CGL_DATASET_NAME)


    # 2. Create output directories
    print(f"Creating output directories at '{OUTPUT_DIR}'...")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    new_metadata = []
    
    # We'll process the 'train' split. You can add 'validation' or 'test' if needed.
    data_split = dataset['train']

    print(f"Processing {len(data_split)} images from the dataset...")
    for i, example in enumerate(tqdm(data_split, desc="Processing images")):
        try:
            image = example['image']
            
            # --- Metadata Transformation ---
            # This part is highly dependent on the dataset's structure.
            # I am assuming there is a 'category' field. Please verify this.
            cgl_category = example.get('category', 'default') # Safely get the category
            
            v_goal = get_v_goal(cgl_category)
            v_format = "poster"  # Assuming most are posters, a safe default.
            v_tone = 0.5         # No tone information, so we use a neutral default.

            # --- Image Saving ---
            filename = f"cgl_{i:06d}.png"
            image_path = IMAGE_DIR / filename
            
            # Ensure image is in RGB format before saving
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_path, "PNG")

            # --- Create Metadata Record ---
            record = {
                "filename": filename,
                "image_path": str(image_path.relative_to(Path(".").resolve())),
                "v_Goal": v_goal,
                "v_Format": v_format,
                "v_Tone": v_tone,
                "source": "cgl_dataset",
                "original_url": "",
                "labeling_method": "inferred",
                "confidence": 0.8 # Assign a default confidence
            }
            new_metadata.append(record)

        except Exception as e:
            print(f"Skipping image {i} due to an error: {e}")

    # 3. Save the new metadata
    print(f"Saving metadata for {len(new_metadata)} new images to '{METADATA_FILE}'...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(new_metadata, f, indent=2)

    return new_metadata

def merge_metadata(new_metadata):
    """
    Merges the new CGL metadata with the existing real designs metadata.
    """
    print("\nMerging datasets...")
    
    # 1. Load existing metadata
    if not EXISTING_METADATA_PATH.exists():
        print(f"Warning: Existing metadata not found at '{EXISTING_METADATA_PATH}'. Creating a new merged file from CGL data only.")
        existing_metadata = []
    else:
        with open(EXISTING_METADATA_PATH, 'r') as f:
            existing_metadata = json.load(f)
    
    print(f"  Found {len(existing_metadata)} existing records.")
    print(f"  Found {len(new_metadata)} new records from CGL.")

    # 2. Combine and save
    merged_dataset = existing_metadata + new_metadata
    print(f"  Total records in merged dataset: {len(merged_dataset)}")

    with open(MERGED_METADATA_PATH, 'w') as f:
        json.dump(merged_dataset, f, indent=2)
        
    print(f"Successfully saved merged metadata to '{MERGED_METADATA_PATH}'!")


def main():
    """Main function to run the import process."""
    
    # Step 1: Download and process the CGL dataset
    cgl_metadata = process_dataset()
    
    # Step 2: Merge with existing dataset
    if cgl_metadata:
        merge_metadata(cgl_metadata)
    else:
        print("No new metadata was generated. Skipping merge.")
        
    print("\nProcess complete!")
    print(f"You can now update your training script to use the merged metadata file:")
    print(f"  --metadata_path {MERGED_METADATA_PATH}")


if __name__ == '__main__':
    main()
