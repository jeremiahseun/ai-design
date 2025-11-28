"""
Test Hybrid Pipeline (SD v1.5 + Smart Text Renderer)
"""

import torch
from pathlib import Path
from src.models.sd_decoder import SDDecoder
from src.generators.text_renderer import TextRenderer

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = Path("visualizations/hybrid_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_hybrid_pipeline():
    print("============================================================")
    print("HYBRID PIPELINE TEST (SD v1.5 + Text Renderer)")
    print("============================================================")

    # 1. Initialize SD Decoder
    try:
        decoder = SDDecoder(device=DEVICE.type if DEVICE.type != 'cuda' else 'cuda')

        # Load LoRA if available
        lora_path = Path('checkpoints/pytorch_lora_weights.safetensors')
        if lora_path.exists():
            decoder.load_lora(str(lora_path))
            print("✨ Using Fine-Tuned LoRA Model")
        else:
            print("⚠️  LoRA weights not found, using base model")

    except Exception as e:
        print(f"Failed to initialize SD: {e}")
        return

    # 2. Initialize Text Renderer
    renderer = TextRenderer()

    # 3. Define Test Cases
    test_cases = [
        # (Goal, Format, Tone, Name)
        (1, 0, 0.2, "Persuade_Poster_Calm"),   # Sale, Elegant
        (1, 2, 0.9, "Persuade_Flyer_Loud"),    # Sale, Bold
        (0, 0, 0.5, "Inform_Poster_Neutral"),  # Info, Clean
        (3, 3, 0.8, "Inspire_Banner_Bold")     # Inspire, Energetic
    ]

    print(f"\nGenerating {len(test_cases)} hybrid designs...")

    for goal, fmt, tone, name in test_cases:
        # A. Generate Background (SD)
        prompt = decoder.meta_to_prompt(goal, fmt, tone)
        print(f"  Generating: {name}...")

        # Use negative prompt to discourage bad text from SD
        image = decoder.pipe(
            prompt,
            num_inference_steps=30,
            negative_prompt="text, watermark, signature, blurry, low quality, distorted"
        ).images[0]

        # B. Render Text (Code)
        metadata = {
            'v_Goal': goal,
            'v_Tone': tone,
            'v_Format': fmt
        }
        final_image = renderer.render_text(image, metadata)

        # Save
        save_path = OUTPUT_DIR / f"{name}.png"
        final_image.save(save_path)
        print(f"  Saved: {save_path}")

    print("\n============================================================")
    print("TEST COMPLETE")
    print(f"Check {OUTPUT_DIR} for results")
    print("============================================================")

if __name__ == "__main__":
    test_hybrid_pipeline()
