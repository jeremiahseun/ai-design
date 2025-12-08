"""
Test Script for Stable Diffusion Decoder (Module 8)
"""

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sd_decoder import SDDecoder
from core.schemas import DEVICE

def visualize_generation(v_meta, image, desc, save_path):
    """Visualize and save image"""
    goal_id = int(v_meta[0])
    format_id = int(v_meta[1])
    tone = float(v_meta[2])

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')

    title = f"{desc}\nGoal: {goal_id} | Format: {format_id} | Tone: {tone:.2f}"
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print("="*60)
    print("STABLE DIFFUSION DECODER TEST")
    print("="*60)

    # Initialize
    try:
        decoder = SDDecoder(device=DEVICE.type if DEVICE.type != 'cuda' else 'cuda')

        # Check for LoRA weights
        lora_path = Path('checkpoints/pytorch_lora_weights.safetensors')
        if lora_path.exists():
            decoder.load_lora(str(lora_path))
            print("✨ Using Fine-Tuned LoRA Model")
        else:
            print("⚠️  LoRA weights not found, using base model")

    except Exception as e:
        print(f"Failed to initialize SD: {e}")
        return

    # Output dir
    vis_dir = Path('visualizations/sd_test')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Test cases (Same as before)
    test_cases = [
        {"goal": 1, "format": 0, "tone": 0.3, "desc": "Persuade_Poster_Calm"},
        {"goal": 1, "format": 2, "tone": 0.8, "desc": "Persuade_Flyer_Energetic"},
        {"goal": 0, "format": 0, "tone": 0.5, "desc": "Inform_Poster_Neutral"}, # SD can handle Goal 0!
        {"goal": 3, "format": 3, "tone": 0.9, "desc": "Inspire_Banner_Bold"},   # SD can handle Goal 3!
    ]

    print(f"\nGenerating {len(test_cases)} designs...")

    # Prepare batch
    v_meta_list = []
    for case in test_cases:
        v_meta_list.append([case['goal'], case['format'], case['tone']])

    v_meta_tensor = torch.tensor(v_meta_list)

    # Generate
    images = decoder.generate(v_meta_tensor, num_inference_steps=30) # 30 is enough for SD

    # Save
    for i, (case, img) in enumerate(zip(test_cases, images)):
        save_path = vis_dir / f"sd_{case['desc']}.png"
        visualize_generation(v_meta_tensor[i], img, case['desc'], save_path)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print(f"Check {vis_dir} for results")
    print("="*60)

if __name__ == "__main__":
    main()
