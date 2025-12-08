"""
Demo Script: Run the Innovation Loop (Module 9)

This demonstrates the "magic" - AI optimizing design metadata to improve quality!

Usage:
    python3 run_innovation_loop.py

What it does:
    1. Loads all trained models
    2. Starts with random design metadata
    3. Optimizes through gradient ascent to maximize grammar scores
    4. Saves progression as images and GIF
    5. Plots optimization curves

Expected outcome (with working decoder):
    - Grammar scores increase over iterations
    - Generated designs improve visually
    - GIF shows design "snapping into alignment"

Current reality (broken decoder):
    - Code runs successfully
    - Shows the optimization process
    - But generated images are noise (decoder issue)
    - Conceptually demonstrates the innovation loop!
"""

import os
import sys
import torch

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.schemas import DEVICE
from integration.innovate import InnovationLoop


def main():
    print("="*70)
    print(" "*15 + "MODULE 9: INNOVATION LOOP")
    print(" "*10 + "Where AI Learns to Design Through Optimization")
    print("="*70)

    print(f"\n📍 Device: {DEVICE}")

    # Check checkpoints exist
    required = [
        'checkpoints/encoder_best.pth',
        'checkpoints/abstractor_best.pth',
        'checkpoints/decoder_best.pth'
    ]

    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Missing checkpoints:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease ensure all models are trained.")
        return

    # Initialize innovation loop
    loop = InnovationLoop(
        encoder_checkpoint='checkpoints/encoder_best.pth',
        abstractor_checkpoint='checkpoints/abstractor_best.pth',
        decoder_checkpoint='checkpoints/decoder_best.pth',
        device=DEVICE
    )

    # Configure optimization
    print("\n" + "="*70)
    print("OPTIMIZATION CONFIGURATION")
    print("="*70)

    num_iterations = 15
    learning_rate = 0.05
    inference_steps = 10  # Low for speed (10-20 is fast, 50+ is slower but better quality)

    print(f"Iterations:      {num_iterations}")
    print(f"Learning Rate:   {learning_rate}")
    print(f"Inference Steps: {inference_steps} (faster sampling)")
    print("\n⏱️  Estimated time: ~3-5 minutes on M3 Pro")

    # Starting point: Random design metadata
    initial_v_meta = torch.tensor([[
        5.0,  # goal: 0-9 (e.g., 5 = Promote)
        2.0,  # format: 0-3 (e.g., 2 = Flyer)
        0.5   # tone: 0-1 (e.g., 0.5 = Neutral)
    ]], device=DEVICE, dtype=torch.float32)

    print("\n" + "="*70)
    print("INITIAL DESIGN METADATA")
    print("="*70)
    print(f"Goal:   {initial_v_meta[0, 0].item():.1f} (int 0-9)")
    print(f"Format: {initial_v_meta[0, 1].item():.1f} (int 0-3)")
    print(f"Tone:   {initial_v_meta[0, 2].item():.2f} (float 0-1)")

    # Run optimization
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION...")
    print("="*70)
    print("The loop will:")
    print("  1. Generate image from V_Meta")
    print("  2. Extract features (Encoder)")
    print("  3. Predict grammar scores (Abstractor)")
    print("  4. Backprop gradients to V_Meta")
    print("  5. Update V_Meta to increase scores")
    print()
    print("⚠️  Note: Decoder generates noise, so visual quality won't improve.")
    print("    But the optimization process demonstrates the concept!")
    print()

    input("Press Enter to start optimization...")

    results = loop.optimize(
        initial_v_meta=initial_v_meta,
        num_iterations=num_iterations,
        lr=learning_rate,
        num_inference_steps=inference_steps,
        save_progression=True,
        output_dir='visualizations/innovation_loop'
    )

    # Plot results
    print("\n" + "="*70)
    print("Creating Optimization Plots...")
    print("="*70)

    loop.plot_history(
        results['history'],
        save_path='visualizations/innovation_loop/optimization_history.png'
    )

    # Final summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)

    initial_score = results['history']['grammar_scores'][0]
    final_score = results['final_score']
    improvement = final_score - initial_score

    print(f"\n📊 Grammar Score Changes:")
    print(f"   Initial: {initial_score:.3f}")
    print(f"   Final:   {final_score:.3f}")
    print(f"   Change:  {improvement:+.3f} ({(improvement/initial_score)*100:+.1f}%)")

    print(f"\n📊 Individual Score Changes:")
    for score_name in ['alignment', 'contrast', 'whitespace', 'hierarchy']:
        initial = results['history'][score_name][0]
        final = results['history'][score_name][-1]
        change = final - initial
        print(f"   {score_name.capitalize():12s}: {initial:.3f} → {final:.3f} ({change:+.3f})")

    print(f"\n📊 Final V_Meta:")
    final_meta = results['final_v_meta'][0]
    print(f"   Goal:   {final_meta[0].item():.2f}")
    print(f"   Format: {final_meta[1].item():.2f}")
    print(f"   Tone:   {final_meta[2].item():.3f}")

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("📁 visualizations/innovation_loop/")
    print("   ├── iter_000.png ... iter_014.png  (progression images)")
    print("   ├── optimization_progression.gif   (animated)")
    print("   └── optimization_history.png       (score plots)")

    print("\n" + "="*70)
    print("WHAT THIS DEMONSTRATES")
    print("="*70)
    print("✅ The innovation loop concept works:")
    print("   - Gradients flow through all three models")
    print("   - V_Meta is optimized via gradient ascent")
    print("   - The optimization process is traceable")
    print()
    print("⚠️  Visual quality is poor because:")
    print("   - Decoder generates noise (training issue)")
    print("   - Encoder extracts features from noise")
    print("   - Grammar scores are based on noisy features")
    print()
    print("🔧 To fix: Retrain decoder for 100+ epochs")
    print("   Once decoder works, this loop will:")
    print("   - Generate actual designs")
    print("   - Improve alignment, contrast, whitespace")
    print("   - Create the 'snap into alignment' effect!")

    print("\n" + "="*70)
    print("MODULE 9 IMPLEMENTATION COMPLETE! 🎉")
    print("="*70)


if __name__ == '__main__':
    main()
