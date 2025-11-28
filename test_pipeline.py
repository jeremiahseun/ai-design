"""
End-to-End Pipeline Test
Tests the complete DTF pipeline: V_Meta → Decoder → P_Image → Encoder → F_Tensor → Abstractor → V_Grammar

This validates that all three models work together correctly, which is essential for Module 9.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from core.schemas import DEVICE
from models.encoder import UNetEncoder
from models.abstractor import Abstractor
from models.decoder import ConditionalUNet
from models.diffusion_utils import DiffusionSchedule


def load_all_models(device):
    """Load all three trained models"""
    print("="*60)
    print("Loading All Models")
    print("="*60)

    models = {}

    # 1. Load Encoder
    print("\n[1/3] Loading Encoder...")
    encoder = UNetEncoder(n_channels=3, n_color_classes=18, bilinear=False)
    checkpoint = torch.load('checkpoints/encoder_best.pth', map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder = encoder.to(device).eval()
    models['encoder'] = encoder
    print(f"   ✅ Encoder loaded (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 2. Load Abstractor
    print("\n[2/3] Loading Abstractor...")
    abstractor = Abstractor(n_goal_classes=4, n_format_classes=3, pretrained=False)
    checkpoint = torch.load('checkpoints/abstractor_best.pth', map_location=device)
    abstractor.load_state_dict(checkpoint['model_state_dict'])
    abstractor = abstractor.to(device).eval()
    models['abstractor'] = abstractor
    print(f"   ✅ Abstractor loaded (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 3. Load Decoder
    print("\n[3/3] Loading Decoder...")
    decoder = ConditionalUNet(
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
    checkpoint = torch.load('checkpoints/decoder_best.pth', map_location=device)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder = decoder.to(device).eval()
    models['decoder'] = decoder

    # Create diffusion schedule
    timesteps = checkpoint.get('timesteps', 1000)
    diffusion = DiffusionSchedule(
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type='linear',
        device=device
    )
    models['diffusion'] = diffusion
    print(f"   ✅ Decoder loaded (Epoch {checkpoint.get('epoch', 'N/A')})")

    print("\n" + "="*60)
    print("✅ All models loaded successfully!")
    print("="*60)

    return models


@torch.no_grad()
def sample_image_from_decoder(decoder, diffusion, v_meta, device, num_steps=50):
    """Sample image from decoder (simplified for pipeline test)"""
    batch_size = v_meta.shape[0]
    x_t = torch.randn(batch_size, 3, 256, 256, device=device)

    timesteps = torch.linspace(
        diffusion.timesteps - 1, 0, num_steps, dtype=torch.long, device=device
    )

    for t in tqdm(timesteps, desc="Generating", leave=False):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = decoder(x_t, t_batch, v_meta)

        alpha_t = diffusion.alphas_cumprod[t]
        alpha_t_prev = diffusion.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=device)

        pred_original = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_original = torch.clamp(pred_original, -1, 1)

        if t > 0:
            noise = torch.randn_like(x_t)
            variance = diffusion.posterior_variance[t]
            x_t = (
                torch.sqrt(alpha_t_prev) * pred_original +
                torch.sqrt(1 - alpha_t_prev - variance) * predicted_noise +
                torch.sqrt(variance) * noise
            )
        else:
            # Last step - no noise
            x_t = torch.sqrt(alpha_t_prev) * pred_original

    images = (x_t + 1) / 2
    return torch.clamp(images, 0, 1)


def run_full_pipeline(models, v_meta, device, num_steps=30):
    """
    Run the complete pipeline:
    V_Meta → Decoder → P_Image → Encoder → F_Tensor → Abstractor → V_Grammar

    Args:
        models: Dict with 'encoder', 'abstractor', 'decoder', 'diffusion'
        v_meta: [B, 3] Input metadata (goal, format, tone)
        device: Device to run on
        num_steps: Number of diffusion sampling steps

    Returns:
        Dictionary with all intermediate results
    """
    results = {}
    results['v_meta_input'] = v_meta.clone()

    print("\n→ Step 1: Decoder (V_Meta → P_Image)")
    p_image = sample_image_from_decoder(
        models['decoder'],
        models['diffusion'],
        v_meta,
        device,
        num_steps=num_steps
    )
    results['p_image'] = p_image
    print(f"   Generated image: {p_image.shape}")

    print("\n→ Step 2: Encoder (P_Image → F_Tensor)")
    f_tensor = models['encoder'].predict(p_image)
    results['f_tensor'] = f_tensor
    print(f"   Extracted features: {f_tensor.shape}")

    print("\n→ Step 3: Abstractor (F_Tensor → V_Grammar + V_Meta)")
    predictions = models['abstractor'](f_tensor)
    v_grammar = predictions['v_grammar']
    v_meta_pred = {
        'v_goal': predictions['v_goal'].argmax(dim=1),
        'v_format': predictions['v_format'].argmax(dim=1),
        'v_tone': predictions['v_tone']
    }
    results['v_grammar'] = v_grammar
    results['v_meta_predicted'] = v_meta_pred
    print(f"   Grammar scores: {v_grammar.shape}")
    print(f"   Predicted metadata: goal={v_meta_pred['v_goal'].item()}, "
          f"format={v_meta_pred['v_format'].item()}, tone={v_meta_pred['v_tone'].item():.2f}")

    return results


def visualize_pipeline_results(results, save_path=None):
    """
    Visualize the complete pipeline results

    Args:
        results: Dictionary from run_full_pipeline
        save_path: Path to save visualization
    """
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)

    # Input metadata (text box)
    ax_meta = fig.add_subplot(gs[0, 0])
    ax_meta.axis('off')
    v_meta = results['v_meta_input'][0]
    meta_text = (f"INPUT:\n\n"
                 f"Goal: {int(v_meta[0])}\n"
                 f"Format: {int(v_meta[1])}\n"
                 f"Tone: {v_meta[2]:.2f}")
    ax_meta.text(0.5, 0.5, meta_text, ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_meta.set_title('V_Meta (Input)', fontweight='bold')

    # Generated image
    ax_img = fig.add_subplot(gs[:, 1:3])
    p_image = results['p_image'][0].cpu().numpy().transpose(1, 2, 0)
    ax_img.imshow(p_image)
    ax_img.set_title('Generated P_Image\n(Decoder Output)', fontsize=12, fontweight='bold')
    ax_img.axis('off')

    # F_Tensor channels
    f_tensor = results['f_tensor'][0].cpu().numpy()
    channel_names = ['Text\nMask', 'Image\nMask', 'Color\nID', 'Hierarchy']
    for i, name in enumerate(channel_names):
        ax = fig.add_subplot(gs[0, 3+i])
        ax.imshow(f_tensor[i], cmap='viridis')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Grammar scores (bar chart)
    ax_grammar = fig.add_subplot(gs[1, 3:])
    v_grammar = results['v_grammar'][0].cpu().numpy()
    grammar_names = ['Alignment', 'Contrast', 'Whitespace', 'Hierarchy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = ax_grammar.bar(grammar_names, v_grammar, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax_grammar.set_ylim(0, 1)
    ax_grammar.set_ylabel('Score', fontweight='bold')
    ax_grammar.set_title('V_Grammar Scores\n(Abstractor Output)', fontsize=12, fontweight='bold')
    ax_grammar.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, v_grammar):
        height = bar.get_height()
        ax_grammar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Predicted metadata (text box)
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_pred.axis('off')
    v_meta_pred = results['v_meta_predicted']
    pred_text = (f"PREDICTED:\n\n"
                 f"Goal: {v_meta_pred['v_goal'].item()}\n"
                 f"Format: {v_meta_pred['v_format'].item()}\n"
                 f"Tone: {v_meta_pred['v_tone'].item():.2f}")
    ax_pred.text(0.5, 0.5, pred_text, ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax_pred.set_title('V_Meta (Predicted)', fontweight='bold')

    plt.suptitle('Complete DTF Pipeline: V_Meta → Decoder → P_Image → Encoder → F_Tensor → Abstractor → V_Grammar',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {save_path}")

    plt.close()


def test_pipeline_with_test_cases(models, device):
    """Test pipeline with predefined test cases"""
    print("\n" + "="*60)
    print("RUNNING END-TO-END PIPELINE TEST")
    print("="*60)

    # Create output directory
    vis_dir = Path('visualizations/pipeline_test')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Test cases
    test_cases = [
        {"goal": 0, "format": 0, "tone": 0.3, "desc": "Case1_Inform_Poster_Calm"},
        {"goal": 1, "format": 1, "tone": 0.8, "desc": "Case2_Persuade_Social_Bold"},
        {"goal": 3, "format": 2, "tone": 0.5, "desc": "Case3_Inspire_Flyer_Neutral"},
    ]

    all_results = []

    for idx, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {idx+1}/{len(test_cases)}: {case['desc']}")
        print(f"{'='*60}")
        print(f"Input: Goal={case['goal']}, Format={case['format']}, Tone={case['tone']:.2f}")

        # Create input v_meta
        v_meta = torch.tensor([[
            case['goal'],
            case['format'],
            case['tone']
        ]], dtype=torch.float32).to(device)

        # Run pipeline
        results = run_full_pipeline(models, v_meta, device, num_steps=30)

        # Print results
        v_grammar = results['v_grammar'][0].cpu().numpy()
        print(f"\n📊 Output Grammar Scores:")
        print(f"   Alignment:   {v_grammar[0]:.3f}")
        print(f"   Contrast:    {v_grammar[1]:.3f}")
        print(f"   Whitespace:  {v_grammar[2]:.3f}")
        print(f"   Hierarchy:   {v_grammar[3]:.3f}")
        print(f"   Average:     {v_grammar.mean():.3f}")

        # Check metadata consistency
        v_meta_pred = results['v_meta_predicted']
        goal_match = v_meta_pred['v_goal'].item() == case['goal']
        format_match = v_meta_pred['v_format'].item() == case['format']
        tone_error = abs(v_meta_pred['v_tone'].item() - case['tone'])

        print(f"\n🔍 Metadata Consistency:")
        print(f"   Goal match:   {'✅' if goal_match else '❌'} (Pred: {v_meta_pred['v_goal'].item()}, Input: {case['goal']})")
        print(f"   Format match: {'✅' if format_match else '❌'} (Pred: {v_meta_pred['v_format'].item()}, Input: {case['format']})")
        print(f"   Tone error:   {tone_error:.3f} (Pred: {v_meta_pred['v_tone'].item():.2f}, Input: {case['tone']:.2f})")

        # Visualize
        save_path = vis_dir / f'pipeline_{case["desc"]}.png'
        visualize_pipeline_results(results, save_path)

        all_results.append({
            'case': case,
            'results': results,
            'grammar_avg': v_grammar.mean(),
            'metadata_consistent': goal_match and format_match and tone_error < 0.2
        })

    # Summary
    print("\n" + "="*60)
    print("PIPELINE TEST SUMMARY")
    print("="*60)

    avg_grammar = np.mean([r['grammar_avg'] for r in all_results])
    consistency_rate = np.mean([r['metadata_consistent'] for r in all_results])

    print(f"\nAverage Grammar Score: {avg_grammar:.3f}")
    print(f"Metadata Consistency:  {consistency_rate*100:.0f}%")

    print(f"\n📁 Visualizations saved to: {vis_dir}")

    # Final verdict
    if avg_grammar > 0.5 and consistency_rate > 0.6:
        print("\n✅ PASS: Pipeline is working correctly!")
        print("   All models integrate successfully.")
        print("   ✓ Ready for Module 9: Innovation Loop")
    elif avg_grammar > 0.3:
        print("\n⚠️  WARNING: Pipeline works but performance is suboptimal")
        print("   Consider retraining some models before Module 9")
    else:
        print("\n❌ FAIL: Pipeline performance is poor")
        print("   Models may not be trained properly")

    return all_results


def main():
    print("="*60)
    print("END-TO-END PIPELINE TEST")
    print("Design Tensor Framework - All Modules Integration")
    print("="*60)

    # Setup
    device = DEVICE
    print(f"\nDevice: {device}")

    # Check all checkpoints exist
    required_checkpoints = [
        'checkpoints/encoder_best.pth',
        'checkpoints/abstractor_best.pth',
        'checkpoints/decoder_best.pth'
    ]

    missing = [cp for cp in required_checkpoints if not os.path.exists(cp)]
    if missing:
        print(f"\n❌ Missing checkpoints:")
        for cp in missing:
            print(f"   - {cp}")
        print("\nPlease ensure all models are trained and checkpoints are available.")
        return

    # Load all models
    models = load_all_models(device)

    # Run pipeline tests
    print("\nℹ️  This will generate 3 designs and run them through the complete pipeline.")
    print("Expected time: ~3-5 minutes")
    response = input("\nContinue? [Y/n]: ")

    if response.lower() != 'n':
        results = test_pipeline_with_test_cases(models, device)

    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
