"""
Dataset Generation Script
Generates synthetic design dataset using the full Phase 1 pipeline:
    Generator -> Renderer -> Extractor -> Grammar -> Save

This creates the training data for Phase 2 models.
"""

import os
import sys
import argparse
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generators.generator import DesignGenerator
from generators.renderer import DesignRenderer
from generators.extractor import FeatureExtractor
from generators.grammar import GrammarEngine


class DatasetGenerator:
    """
    Orchestrates the full synthetic data pipeline
    """

    def __init__(self, output_dir: str = 'data/synthetic_dataset', seed: int = None):
        """
        Initialize dataset generator

        Args:
            output_dir: Directory to save dataset
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.seed = seed

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'f_tensors').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)

        # Initialize pipeline components
        self.generator = DesignGenerator(seed=seed)
        self.renderer = DesignRenderer()
        self.extractor = FeatureExtractor()
        self.grammar = GrammarEngine()

        print(f"Dataset will be saved to: {self.output_dir}")

    def generate_sample(self, index: int) -> dict:
        """
        Generate a single training sample

        Returns:
            Dictionary containing all components
        """
        # 1. Generate design JSON (Source of Truth)
        design = self.generator.generate()

        # 2. Render to image (P_Image)
        rendered_img = self.renderer.render(design)

        # 3. Extract structural features (F_Tensor)
        f_tensor = self.extractor.extract(design)

        # 4. Calculate grammar scores (V_Grammar)
        grammar_scores = self.grammar.calculate_all(f_tensor, rendered_img)

        # Package into sample
        sample = {
            'index': index,
            'design_json': design,
            'p_image': rendered_img,  # [H, W, 3]
            'f_tensor': f_tensor,     # [4, H, W]
            'v_grammar': np.array([
                grammar_scores['Alignment'],
                grammar_scores['Contrast'],
                grammar_scores['Whitespace'],
                grammar_scores['Hierarchy']
            ], dtype=np.float32),
            'v_meta': design['meta']
        }

        return sample

    def save_sample(self, sample: dict):
        """
        Save a sample to disk
        """
        idx = sample['index']

        # Save rendered image as PNG
        img_path = self.output_dir / 'images' / f'{idx:06d}.npy'
        np.save(img_path, sample['p_image'])

        # Save F_Tensor
        f_tensor_path = self.output_dir / 'f_tensors' / f'{idx:06d}.npy'
        np.save(f_tensor_path, sample['f_tensor'])

        # Save metadata (JSON, V_Meta, V_Grammar)
        meta_path = self.output_dir / 'metadata' / f'{idx:06d}.json'
        metadata = {
            'index': idx,
            'design_json': sample['design_json'],
            'v_meta': sample['v_meta'],
            'v_grammar': sample['v_grammar'].tolist()
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_dataset(self, num_samples: int, batch_size: int = 100):
        """
        Generate full dataset

        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for progress reporting
        """
        print(f"\nGenerating {num_samples} samples...")
        print("=" * 60)

        stats = {
            'alignment': [],
            'contrast': [],
            'whitespace': [],
            'hierarchy': []
        }

        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Generate and save sample
            sample = self.generate_sample(i)
            self.save_sample(sample)

            # Collect statistics
            grammar = sample['v_grammar']
            stats['alignment'].append(grammar[0])
            stats['contrast'].append(grammar[1])
            stats['whitespace'].append(grammar[2])
            stats['hierarchy'].append(grammar[3])

        # Save dataset statistics
        self._save_statistics(stats, num_samples)

        print("\n" + "=" * 60)
        print("Dataset generation complete!")
        print(f"Samples saved to: {self.output_dir}")

    def _save_statistics(self, stats: dict, num_samples: int):
        """
        Save dataset statistics
        """
        stats_path = self.output_dir / 'dataset_stats.json'

        statistics = {
            'num_samples': num_samples,
            'grammar_scores': {
                'alignment': {
                    'mean': float(np.mean(stats['alignment'])),
                    'std': float(np.std(stats['alignment'])),
                    'min': float(np.min(stats['alignment'])),
                    'max': float(np.max(stats['alignment']))
                },
                'contrast': {
                    'mean': float(np.mean(stats['contrast'])),
                    'std': float(np.std(stats['contrast'])),
                    'min': float(np.min(stats['contrast'])),
                    'max': float(np.max(stats['contrast']))
                },
                'whitespace': {
                    'mean': float(np.mean(stats['whitespace'])),
                    'std': float(np.std(stats['whitespace'])),
                    'min': float(np.min(stats['whitespace'])),
                    'max': float(np.max(stats['whitespace']))
                },
                'hierarchy': {
                    'mean': float(np.mean(stats['hierarchy'])),
                    'std': float(np.std(stats['hierarchy'])),
                    'min': float(np.min(stats['hierarchy'])),
                    'max': float(np.max(stats['hierarchy']))
                }
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)

        print("\nDataset Statistics:")
        print("-" * 60)
        for dimension, values in statistics['grammar_scores'].items():
            print(f"{dimension.capitalize():12s}: μ={values['mean']:.3f}, σ={values['std']:.3f}, "
                  f"range=[{values['min']:.3f}, {values['max']:.3f}]")

    def create_visualization_samples(self, num_samples: int = 10):
        """
        Create visualization samples for inspection
        """
        print(f"\nCreating {num_samples} visualization samples...")

        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        for i in range(num_samples):
            sample = self.generate_sample(i)

            # Create side-by-side visualization
            from PIL import Image

            # Rendered image
            img = sample['p_image']

            # F_Tensor channels
            f_vis = self.extractor.visualize_channels(sample['f_tensor'])

            # Combine horizontally
            combined = np.concatenate([img, f_vis], axis=1)

            # Add grammar scores as text
            combined_img = Image.fromarray(combined.astype(np.uint8))

            # Save
            vis_path = vis_dir / f'sample_{i:03d}.png'
            combined_img.save(vis_path)

        print(f"Visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic design dataset')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='data/synthetic_dataset',
                       help='Output directory (default: data/synthetic_dataset)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--visualize', type=int, default=10,
                       help='Number of visualization samples to create (default: 10)')
    parser.add_argument('--test', action='store_true',
                       help='Generate only 100 samples for testing')

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.num_samples = 100
        print("TEST MODE: Generating only 100 samples")

    # Create generator
    dataset_gen = DatasetGenerator(output_dir=args.output_dir, seed=args.seed)

    # Generate visualizations first
    if args.visualize > 0:
        dataset_gen.create_visualization_samples(num_samples=args.visualize)

    # Generate dataset
    dataset_gen.generate_dataset(num_samples=args.num_samples)

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  1. Inspect visualizations in: data/synthetic_dataset/visualizations/")
    print("  2. Check dataset_stats.json for distribution info")
    print("  3. Proceed to Phase 2: Model training")
    print("=" * 60)


if __name__ == '__main__':
    main()
