import os
import sys
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# Add src to path
sys.path.append('src')

from core.schemas import DEVICE
from models.encoder import UNetEncoder
from models.abstractor import Abstractor

def load_models(encoder_path, abstractor_path, device):
    print(f"Loading models on {device}...")

    # Load Encoder
    encoder = UNetEncoder(n_channels=3, n_color_classes=18, bilinear=False).to(device)
    encoder_ckpt = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    encoder.eval()
    print("  Encoder loaded")

    # Load Abstractor
    abstractor = Abstractor(n_goal_classes=4, n_format_classes=3, pretrained=False).to(device)
    abstractor_ckpt = torch.load(abstractor_path, map_location=device)
    abstractor.load_state_dict(abstractor_ckpt['model_state_dict'])
    abstractor.eval()
    print("  Abstractor loaded")

    return encoder, abstractor

def process_images(input_dir, output_dir, encoder, abstractor, device):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    images_dir = output_path / 'images'
    meta_dir = output_path / 'metadata'

    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # [0, 1]
    ])

    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
    print(f"Found {len(image_files)} images in {input_dir}")

    success_count = 0
    error_count = 0

    for i, img_file in enumerate(tqdm(image_files)):
        try:
            # Load and resize image
            img = Image.open(img_file).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device) # [1, 3, 256, 256]

            # Auto-labeling pipeline
            with torch.no_grad():
                # 1. Encoder: RGB -> F_Tensor
                f_tensor = encoder.predict(img_tensor) # [1, 4, 256, 256]

                # 2. Abstractor: F_Tensor -> V_Meta
                outputs = abstractor(f_tensor)

                # Extract predictions
                v_goal = torch.argmax(outputs['v_goal'], dim=1).item()
                v_format = torch.argmax(outputs['v_format'], dim=1).item()
                v_tone = outputs['v_tone'].item()

                # Also get grammar scores for reference
                v_grammar = outputs['v_grammar'][0].tolist()

            # Save processed image
            out_filename = f"{i:06d}"
            img_save_path = images_dir / f"{out_filename}.png"
            img.resize((256, 256)).save(img_save_path)

            # Save metadata
            meta_save_path = meta_dir / f"{out_filename}.json"
            metadata = {
                "filename": f"{out_filename}.png",
                "original_file": img_file.name,
                "v_meta": {
                    "v_Goal": v_goal,
                    "v_Format": v_format,
                    "v_Tone": v_tone
                },
                "v_grammar": v_grammar # Optional, but good to have
            }

            with open(meta_save_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            success_count += 1

        except Exception as e:
            # print(f"Error processing {img_file}: {e}")
            error_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare Real Data for Training')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw images')
    parser.add_argument('--output_dir', type=str, default='data/real_designs', help='Output directory for dataset')
    parser.add_argument('--encoder_ckpt', type=str, default='checkpoints/encoder_best.pth', help='Path to Encoder checkpoint')
    parser.add_argument('--abstractor_ckpt', type=str, default='checkpoints/abstractor_best.pth', help='Path to Abstractor checkpoint')

    args = parser.parse_args()

    if not os.path.exists(args.encoder_ckpt) or not os.path.exists(args.abstractor_ckpt):
        print("Error: Checkpoints not found. Please ensure 'checkpoints/encoder_best.pth' and 'checkpoints/abstractor_best.pth' exist.")
        return

    device = DEVICE

    encoder, abstractor = load_models(args.encoder_ckpt, args.abstractor_ckpt, device)

    process_images(args.input_dir, args.output_dir, encoder, abstractor, device)

if __name__ == '__main__':
    main()
