# Phase 2: Neural Network Training Plan

**Status**: In Progress 🚧
**Started**: 2024
**Objective**: Train three neural networks to learn design decomposition

---

## Architecture Overview

```
P_Image → [Encoder] → F_Tensor → [Abstractor] → V_Grammar + V_Meta
                                       ↓
V_Meta ────────────────────→ [Decoder] → P_Image (generated)
```

---

## Module 5: U-Net Encoder ⏳ IN PROGRESS

### Goal
Learn to extract structural features from rendered designs.

**Input**: P_Image `[B, 3, 256, 256]`
**Output**: F_Tensor `[B, 4, 256, 256]`

### Architecture
- Standard U-Net with skip connections
- Encoder: 4 downsampling blocks
- Decoder: 4 upsampling blocks with skip connections
- Final conv layer outputs 4 channels

### Loss Function (CompositeLoss)
```python
Total_Loss = α₁·DiceLoss(K=0,1) + α₂·CrossEntropy(K=2) + α₃·MSE(K=3)
```
- **DiceLoss**: Text mask (K=0) + Image mask (K=1)
- **CrossEntropy**: Color ID classification (K=2)
- **MSE**: Hierarchy map regression (K=3)
- **Weights**: α₁=1.0, α₂=0.5, α₃=1.0

### Training Config
```yaml
epochs: 5-10
batch_size: 32 (16 if OOM)
learning_rate: 1e-4
optimizer: Adam
scheduler: ReduceLROnPlateau
device: MPS (Apple Silicon)
```

### Success Criteria
- ✅ Text mask IoU > 0.90
- ✅ Image mask IoU > 0.85
- ✅ Color ID accuracy > 0.80
- ✅ Hierarchy MSE < 0.05

### Files to Implement
- [x] `src/models/encoder.py` - U-Net architecture
- [x] `train_scripts/train_encoder.py` - Training loop
- [ ] `src/utils/visualization.py` - Visualization helpers
- [ ] Trained checkpoint: `checkpoints/encoder_best.pth`

### Commands
```bash
# Train encoder
python train_scripts/train_encoder.py --epochs 5 --batch_size 32

# Resume training
python train_scripts/train_encoder.py --resume checkpoints/encoder_epoch_3.pth

# Validate only
python train_scripts/train_encoder.py --validate --checkpoint checkpoints/encoder_best.pth
```

### Expected Timeline
**2-3 days**
- Day 1: Architecture + training script
- Day 2: First training run, debug
- Day 3: Optimization, validation

---

## Module 6: Abstractor 📋 TODO

### Goal
Learn to predict design quality and metadata from structure.

**Input**: F_Tensor `[B, 4, 256, 256]`
**Output**:
- V_Grammar `[B, 4]` (design scores)
- V_Meta components (goal, tone, format)

### Architecture
```
F_Tensor → ResNet-18 → Global Pool → FC-512
                        ├→ MLP Head 1 → V_Meta
                        └→ MLP Head 2 → V_Grammar [4]
```

### Loss Function
```python
Total_Loss = CE(v_Goal) + CE(v_Format) + MSE(v_Tone) + MSE(V_Grammar)
```

### Training Config
```yaml
epochs: 10-15
batch_size: 64
learning_rate: 1e-4
optimizer: Adam
pretrained_resnet: True (ImageNet weights)
```

### Success Criteria
- ✅ Grammar score MAE < 0.10 per dimension
- ✅ Goal classification accuracy > 85%
- ✅ Format classification accuracy > 85%

### Files to Implement
- [ ] `src/models/abstractor.py` - ResNet + Heads
- [ ] `train_scripts/train_abstractor.py` - Training loop
- [ ] Checkpoint: `checkpoints/abstractor_best.pth`

### Expected Timeline
**2 days**

---

## Module 7: Conditional DDPM Decoder 📋 TODO

### Goal
Generate designs from semantic metadata.

**Input**: V_Meta (embedded) + Noise ε
**Output**: P_Image `[B, 3, 256, 256]`

### Architecture
- Denoising Diffusion Probabilistic Model (DDPM)
- U-Net backbone with time embedding
- Cross-attention for V_Meta conditioning
- T=1000 timesteps

### Diffusion Process
```
Forward: x₀ → x₁ → ... → xₜ (add noise)
Reverse: xₜ → ... → x₁ → x₀ (denoise)
```

### Loss Function
```python
Loss = MSE(ε_predicted, ε_target)  # Noise prediction loss
```

### Training Config
```yaml
epochs: 30-50
batch_size: 16 (8 if OOM)
learning_rate: 2e-4
optimizer: AdamW
timesteps: 1000
beta_schedule: linear
```

### Success Criteria
- ✅ Generated images are visually coherent
- ✅ Conditioning works (same V_Meta → similar designs)
- ✅ FID score < 50

### Files to Implement
- [ ] `src/models/decoder.py` - DDPM U-Net
- [ ] `src/models/diffusion_utils.py` - Beta schedule, sampling
- [ ] `train_scripts/train_decoder.py` - Training loop
- [ ] `train_scripts/sample_decoder.py` - Sampling script
- [ ] Checkpoint: `checkpoints/decoder_best.pth`

### Expected Timeline
**5-7 days**

---

## Training Infrastructure

### Dataset Loader
```python
# Located in: src/utils/dataset.py
class SyntheticDesignDataset(torch.utils.data.Dataset):
    """Loads pre-generated synthetic dataset"""

    def __getitem__(self, idx):
        p_image = np.load(f'images/{idx:06d}.npy')
        f_tensor = np.load(f'f_tensors/{idx:06d}.npy')
        metadata = json.load(f'metadata/{idx:06d}.json')
        return p_image, f_tensor, metadata
```

### Checkpoint Management
```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f'checkpoints/model_epoch_{epoch}.pth')

# Load checkpoint
checkpoint = torch.load('checkpoints/model_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Visualization Strategy
1. **During Training**: Log loss curves, sample predictions
2. **After Epoch**: Generate validation visualizations
3. **Final**: Create comprehensive test set evaluation

---

## Progress Tracking

### Week 1: Encoder ⏳
- [x] Implement U-Net architecture
- [ ] Implement CompositeLoss
- [ ] Create training script
- [ ] Train for 5 epochs
- [ ] Validate and checkpoint

### Week 2: Abstractor 📋
- [ ] Implement ResNet + Heads
- [ ] Create training script
- [ ] Train for 15 epochs
- [ ] Evaluate predictions

### Week 3-4: Decoder 📋
- [ ] Implement DDPM architecture
- [ ] Implement diffusion utilities
- [ ] Create training + sampling scripts
- [ ] Train for 30+ epochs
- [ ] Generate samples, evaluate quality

---

## Common Issues & Solutions

### MPS (Apple Silicon) Issues

**Memory Issues**
```bash
# Reduce batch size
--batch_size 16  # or even 8
```

**MPS Op Not Supported**
```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

**Gradient Issues**
```python
# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Training Issues

**Loss Not Decreasing**
- Check learning rate (try 1e-5 if 1e-4 too high)
- Verify data normalization
- Check loss function weights

**Overfitting**
- Add dropout layers
- Use data augmentation
- Reduce model size

**Slow Training**
- Reduce batch size and use gradient accumulation
- Use mixed precision (if MPS supports)
- Cache dataset in memory

---

## Next Steps After Phase 2

Once all three models are trained:

### Phase 3: Integration & Innovation

**Module 8**: Fine-tuning loop (RL-style optimization)
```python
# Freeze Encoder + Abstractor
# Update Decoder to maximize grammar scores
loss = diffusion_loss - λ * sum(grammar_scores)
```

**Module 9**: Innovation loop (latent space optimization)
```python
# Fix all models in eval mode
# Optimize latent z via gradient ascent
# Generate designs that "snap" into alignment
```

---

## Resource Requirements

### Compute
- **Training Time**: ~20-30 hours total (across all models)
- **GPU/MPS Memory**: 8GB minimum
- **Storage**: ~5GB (dataset + checkpoints)

### Dependencies
All in `requirements.txt`:
- PyTorch 2.0+ (with MPS support)
- NumPy, OpenCV, Pillow
- tqdm, matplotlib

---

## Evaluation Metrics

### Encoder (Module 5)
- IoU (Intersection over Union) for masks
- Pixel accuracy for color IDs
- MSE for hierarchy map
- Visual inspection

### Abstractor (Module 6)
- MAE (Mean Absolute Error) for grammar scores
- Classification accuracy for meta fields
- Correlation with ground truth

### Decoder (Module 7)
- FID (Fréchet Inception Distance)
- Visual quality (human evaluation)
- Conditioning fidelity
- Diversity of outputs

---

## Questions & Support

If you encounter issues:
1. Check this document first
2. Review error messages in training logs
3. Inspect visualizations for debugging
4. Adjust hyperparameters as needed

---

**Current Status**: Implementing Module 5 (Encoder)
**Last Updated**: 2024-11-13
