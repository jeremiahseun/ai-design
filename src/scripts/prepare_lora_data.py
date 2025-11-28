"""
Prepare Dataset for LoRA Training (Module 8)

Converts the 'real_designs' dataset (Images + JSON Metadata) into a Hugging Face compatible dataset (Images + Captions).
Output format:
    data/lora_dataset/
        metadata.jsonl  (contains {"file_name": "img.png", "text": "caption..."})
        img1.png
        img2.png
        ...
"""

import os
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Add src to path to import SDDecoder logic
sys.path.append('src')
from models.sd_decoder import SDDecoder

def prepare_lora_dataset(source_dir: str, output_dir: str):
    """
    Prepare dataset for LoRA training
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print(f"Preparing LoRA dataset...")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")

    # Initialize helper to generate prompts
    # We don't need to load the model, just the helper methods
    # But SDDecoder loads model in __init__.
    # Let's instantiate it with cpu to avoid heavy loading if possible,
    # or just copy the logic.
    # Actually, let's just copy the logic here to avoid loading the 4GB model just for text processing.

    # Get all metadata files
    meta_dir = source_path / 'metadata'
    image_dir = source_path / 'images'

    meta_files = sorted(list(meta_dir.glob('*.json')))
    print(f"  Found {len(meta_files)} samples")

    metadata_lines = []

    for meta_file in tqdm(meta_files, desc="Processing"):
        # Load metadata
        with open(meta_file, 'r') as f:
            data = json.load(f)

        v_meta = data['v_meta']
        filename = meta_file.stem

        # Check image exists
        src_img = image_dir / f"{filename}.png"
        if not src_img.exists():
            continue

        # Generate caption
        # Logic copied from SDDecoder.meta_to_prompt to avoid loading model
        goal_id = v_meta['v_Goal']
        format_id = v_meta['v_Format']
        tone = v_meta['v_Tone']

        caption = meta_to_prompt(goal_id, format_id, tone)

        # Copy image to output dir
        dst_img = output_path / f"{filename}.png"
        shutil.copy2(src_img, dst_img)

        # Add to metadata list
        metadata_lines.append({
            "file_name": f"{filename}.png",
            "text": caption
        })

    # Save metadata.jsonl
    with open(output_path / 'metadata.jsonl', 'w') as f:
        for line in metadata_lines:
            f.write(json.dumps(line) + '\n')

    print(f"✅ Dataset prepared successfully!")
    print(f"  Images: {len(metadata_lines)}")
    print(f"  Location: {output_path}")


def meta_to_prompt(goal_id: int, format_id: int, tone: float) -> str:
    """
    Convert metadata to text prompt (Same logic as SDDecoder)
    """
    # Mappings
    goals = {
        0: "informative, educational, clear information hierarchy, infographic style",
        1: "persuasive, compelling, call to action, marketing focus",
        2: "entertaining, fun, engaging, playful elements",
        3: "inspiring, motivational, emotional, artistic"
    }

    formats = {
        0: "poster design, vertical layout, print quality",
        1: "social media post, square format, digital marketing",
        2: "flyer design, promotional material, handout",
        3: "banner, horizontal layout, header"
    }

    # Tone logic
    if tone < 0.4:
        tone_desc = "minimalist, calm, clean, soft pastel colors, elegant, sophisticated, whitespace"
    elif tone < 0.7:
        tone_desc = "professional, modern, balanced, corporate, trustworthy, clear"
    else:
        tone_desc = "vibrant, energetic, bold colors, dynamic, high contrast, loud, exciting"

    # Construct prompt
    base_prompt = "professional graphic design, high quality, 4k, trending on behance, vector art style"

    goal_text = goals.get(goal_id, "graphic design")
    format_text = formats.get(format_id, "design")

    prompt = f"{format_text}, {goal_text}, {tone_desc}, {base_prompt}"
    return prompt


if __name__ == "__main__":
    prepare_lora_dataset(
        source_dir='data/real_designs',
        output_dir='data/lora_dataset'
    )
