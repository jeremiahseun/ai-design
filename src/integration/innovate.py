"""
Module 9: Innovation Loop
Optimizes design metadata (V_Meta) to maximize grammar scores through gradient ascent

The "Magic" - This is where the AI actually learns to design!

Process:
    1. Load all three trained models (Encoder, Abstractor, Decoder)
    2. Start with initial V_Meta (design metadata)
    3. Enable gradients on V_Meta
    4. Loop:
        a. Generate image from V_Meta (Decoder)
        b. Extract features from image (Encoder)
        c. Predict grammar scores (Abstractor)
        d. Calculate loss = -sum(grammar_scores)
        e. Backpropagate to V_Meta
        f. Update V_Meta via gradient ascent
    5. Track progression and create visualizations

Expected outcome: Grammar scores increase, designs improve!
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io

from models.encoder import UNetEncoder
from models.abstractor import Abstractor
from models.decoder import ConditionalUNet
from models.diffusion_utils import DiffusionSchedule


class InnovationLoop:
    """
    Innovation Loop: Optimize V_Meta to maximize design quality
    """

    def __init__(
        self,
        encoder_checkpoint: str,
        abstractor_checkpoint: str,
        decoder_checkpoint: str,
        device: torch.device
    ):
        """
        Initialize the innovation loop with trained models

        Args:
            encoder_checkpoint: Path to encoder checkpoint
            abstractor_checkpoint: Path to abstractor checkpoint
            decoder_checkpoint: Path to decoder checkpoint
            device: Device to run on (cuda/mps/cpu)
        """
        self.device = device
        print("="*60)
        print("Initializing Innovation Loop (Module 9)")
        print("="*60)

        # Load models
        print("\nLoading models...")
        self.encoder = self._load_encoder(encoder_checkpoint)
        self.abstractor = self._load_abstractor(abstractor_checkpoint)
        self.decoder, self.diffusion = self._load_decoder(decoder_checkpoint)

        # Set all models to eval mode (no training, just inference)
        self.encoder.eval()
        self.abstractor.eval()
        self.decoder.eval()

        # Freeze all model parameters (they won't be updated)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.abstractor.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        print("\n✅ All models loaded and frozen")
        print("="*60)

    def _load_encoder(self, checkpoint_path: str) -> UNetEncoder:
        """Load trained encoder"""
        print("  [1/3] Loading Encoder...")
        model = UNetEncoder(n_channels=3, n_color_classes=18, bilinear=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device).eval()
        print(f"    ✅ Encoder loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
        return model

    def _load_abstractor(self, checkpoint_path: str) -> Abstractor:
        """Load trained abstractor"""
        print("  [2/3] Loading Abstractor...")
        model = Abstractor(n_goal_classes=4, n_format_classes=3, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device).eval()
        print(f"    ✅ Abstractor loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
        return model

    def _load_decoder(self, checkpoint_path: str) -> Tuple[ConditionalUNet, DiffusionSchedule]:
        """Load trained decoder and diffusion schedule"""
        print("  [3/3] Loading Decoder...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            channel_multipliers=(1, 2, 4, 8),
            n_goal_classes=10,
            n_format_classes=4,
            time_emb_dim=256,
            meta_emb_dim=256,
            attention_levels=(3,),
            dropout=0.1,
            use_gradient_checkpointing=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device).eval()

        timesteps = checkpoint.get('timesteps', 1000)
        diffusion = DiffusionSchedule(
            timesteps=timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type='linear',
            device=self.device
        )

        print(f"    ✅ Decoder loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
        return model, diffusion

    @torch.enable_grad()
    def generate_with_gradients(
        self,
        v_meta: torch.Tensor,
        num_inference_steps: int = 20
    ) -> torch.Tensor:
        """
        Generate image from V_Meta with gradient tracking

        This is different from normal generation - we need gradients!

        Args:
            v_meta: [1, 3] Design metadata (goal, format, tone)
            num_inference_steps: Number of denoising steps

        Returns:
            Generated image [1, 3, 256, 256] with gradients
        """
        batch_size = v_meta.shape[0]

        # Start from noise (no gradients needed here)
        with torch.no_grad():
            x_t = torch.randn(batch_size, 3, 256, 256, device=self.device)

        # Denoising steps
        timesteps = torch.linspace(
            self.diffusion.timesteps - 1, 0, num_inference_steps,
            dtype=torch.long, device=self.device
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Forward pass WITH gradients (v_meta has gradients)
            predicted_noise = self.decoder(x_t, t_batch, v_meta)

            # Denoising step (simplified for gradient flow)
            alpha_t = self.diffusion.alphas_cumprod[t]
            alpha_t_prev = self.diffusion.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=self.device)

            # Predict original image
            pred_original = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_original = torch.clamp(pred_original, -1, 1)

            # Update x_t (last step has no noise)
            if t > 0:
                noise = torch.randn_like(x_t)
                variance = self.diffusion.posterior_variance[t]
                x_t = (
                    torch.sqrt(alpha_t_prev) * pred_original +
                    torch.sqrt(1 - alpha_t_prev - variance) * predicted_noise +
                    torch.sqrt(variance) * noise
                )
            else:
                x_t = torch.sqrt(alpha_t_prev) * pred_original

        # Denormalize to [0, 1]
        images = (x_t + 1) / 2
        return torch.clamp(images, 0, 1)

    def forward_pass(self, v_meta: torch.Tensor, num_inference_steps: int = 20) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through all models

        Args:
            v_meta: [1, 3] Design metadata with gradients

        Returns:
            Dictionary with intermediate results and grammar scores
        """
        # 1. Generate image from V_Meta
        p_image = self.generate_with_gradients(v_meta, num_inference_steps)

        # 2. Extract features
        f_tensor = self.encoder.predict(p_image)

        # 3. Predict grammar scores
        predictions = self.abstractor(f_tensor)
        v_grammar = predictions['v_grammar']  # [1, 4]

        return {
            'p_image': p_image,
            'f_tensor': f_tensor,
            'v_grammar': v_grammar,
            'grammar_scores': {
                'alignment': v_grammar[0, 0].item(),
                'contrast': v_grammar[0, 1].item(),
                'whitespace': v_grammar[0, 2].item(),
                'hierarchy': v_grammar[0, 3].item(),
                'total': v_grammar.sum().item()
            },
            'total_score_tensor': v_grammar.sum()  # Keep as tensor for gradients
        }

    def optimize(
        self,
        initial_v_meta: torch.Tensor,
        num_iterations: int = 20,
        lr: float = 0.1,
        num_inference_steps: int = 20,
        save_progression: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run the innovation loop: optimize V_Meta to maximize grammar scores

        This is the MAGIC! 🪄

        Args:
            initial_v_meta: [1, 3] Starting design metadata
            num_iterations: Number of optimization steps
            lr: Learning rate for gradient ascent
            num_inference_steps: Diffusion sampling steps (less = faster)
            save_progression: Save images at each step
            output_dir: Directory to save progression images

        Returns:
            Dictionary with results and history
        """
        print("\n" + "="*60)
        print("RUNNING INNOVATION LOOP")
        print("="*60)
        print(f"Iterations: {num_iterations}")
        print(f"Learning Rate: {lr}")
        print(f"Inference Steps: {num_inference_steps}")
        print("="*60)

        # Prepare output directory
        if save_progression and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Initialize V_Meta with gradients
        v_meta = initial_v_meta.clone().detach().requires_grad_(True)

        # Track history
        history = {
            'grammar_scores': [],
            'alignment': [],
            'contrast': [],
            'whitespace': [],
            'hierarchy': [],
            'v_meta': []
        }

        images = []

        print("\nOptimization Progress:")
        print("-"*60)

        for iteration in range(num_iterations):
            # Forward pass
            results = self.forward_pass(v_meta, num_inference_steps)

            # Extract scores
            scores = results['grammar_scores']
            total_score = scores['total']

            # Get tensor version for backprop
            total_score_tensor = results['total_score_tensor']

            # Loss = negative of grammar score (we want to MAXIMIZE)
            loss = -total_score_tensor

            # Backward pass (compute gradients)
            if v_meta.grad is not None:
                v_meta.grad.zero_()
            loss.backward()

            # Gradient ascent update (move in direction that increases scores)
            with torch.no_grad():
                v_meta += lr * v_meta.grad  # Note: += not -= (gradient ASCENT)

                # Clip v_meta to valid ranges
                # goal [0, 9], format [0, 3], tone [0, 1]
                v_meta[0, 0] = torch.clamp(v_meta[0, 0], 0, 9)
                v_meta[0, 1] = torch.clamp(v_meta[0, 1], 0, 3)
                v_meta[0, 2] = torch.clamp(v_meta[0, 2], 0, 1)

            # Re-enable gradients
            v_meta = v_meta.detach().requires_grad_(True)

            # Log progress
            history['grammar_scores'].append(total_score)
            history['alignment'].append(scores['alignment'])
            history['contrast'].append(scores['contrast'])
            history['whitespace'].append(scores['whitespace'])
            history['hierarchy'].append(scores['hierarchy'])
            history['v_meta'].append(v_meta.detach().cpu().numpy().copy())

            # Print progress
            print(f"Iter {iteration:2d} | Total: {total_score:.3f} | "
                  f"A:{scores['alignment']:.2f} C:{scores['contrast']:.2f} "
                  f"W:{scores['whitespace']:.2f} H:{scores['hierarchy']:.2f}")

            # Save image
            if save_progression:
                img_tensor = results['p_image'][0].detach().cpu()
                images.append(img_tensor)

                if output_dir:
                    img_path = output_path / f'iter_{iteration:03d}.png'
                    self._save_image(img_tensor, img_path)

        print("-"*60)
        print(f"\n✅ Optimization Complete!")
        print(f"Initial Score: {history['grammar_scores'][0]:.3f}")
        print(f"Final Score:   {history['grammar_scores'][-1]:.3f}")
        print(f"Improvement:   {history['grammar_scores'][-1] - history['grammar_scores'][0]:.3f}")

        # Create GIF if we saved images
        if save_progression and output_dir and len(images) > 0:
            gif_path = output_path / 'optimization_progression.gif'
            self._create_gif(images, gif_path)
            print(f"\n🎬 Created GIF: {gif_path}")

        return {
            'final_v_meta': v_meta.detach(),
            'final_score': history['grammar_scores'][-1],
            'history': history,
            'images': images if save_progression else None
        }

    def _save_image(self, img_tensor: torch.Tensor, path: Path):
        """Save image tensor to file"""
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(path)

    def _create_gif(self, images: List[torch.Tensor], path: Path, duration: int = 200):
        """Create animated GIF from image tensors"""
        pil_images = []
        for img_tensor in images:
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        # Save as GIF
        pil_images[0].save(
            path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0
        )

    def plot_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot optimization history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot total grammar score
        ax1.plot(history['grammar_scores'], linewidth=2, marker='o')
        ax1.set_xlabel('Iteration', fontweight='bold')
        ax1.set_ylabel('Total Grammar Score', fontweight='bold')
        ax1.set_title('Optimization Progress', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot individual scores
        ax2.plot(history['alignment'], label='Alignment', marker='o', alpha=0.7)
        ax2.plot(history['contrast'], label='Contrast', marker='s', alpha=0.7)
        ax2.plot(history['whitespace'], label='Whitespace', marker='^', alpha=0.7)
        ax2.plot(history['hierarchy'], label='Hierarchy', marker='d', alpha=0.7)
        ax2.set_xlabel('Iteration', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Individual Grammar Scores', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.close()


if __name__ == '__main__':
    """Demo of the innovation loop"""
    from core.schemas import DEVICE

    print("="*60)
    print("MODULE 9: INNOVATION LOOP DEMO")
    print("="*60)

    # Initialize
    loop = InnovationLoop(
        encoder_checkpoint='checkpoints/encoder_best.pth',
        abstractor_checkpoint='checkpoints/abstractor_best.pth',
        decoder_checkpoint='checkpoints/decoder_best.pth',
        device=DEVICE
    )

    # Starting V_Meta (random design)
    initial_v_meta = torch.tensor([[
        5.0,  # goal (0-9)
        2.0,  # format (0-3)
        0.5   # tone (0-1)
    ]], device=DEVICE)

    print(f"\nInitial V_Meta:")
    print(f"  Goal: {initial_v_meta[0, 0].item()}")
    print(f"  Format: {initial_v_meta[0, 1].item()}")
    print(f"  Tone: {initial_v_meta[0, 2].item()}")

    # Run optimization
    results = loop.optimize(
        initial_v_meta=initial_v_meta,
        num_iterations=10,
        lr=0.05,
        num_inference_steps=10,  # Few steps for speed
        save_progression=True,
        output_dir='visualizations/innovation_loop'
    )

    # Plot results
    loop.plot_history(
        results['history'],
        save_path='visualizations/innovation_loop/optimization_history.png'
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("Check visualizations/innovation_loop/ for results")
