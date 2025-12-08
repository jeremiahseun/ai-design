# File 2: User Workflow Guide

**Action:** Keep this for yourself. This is your step-by-step checklist to managing Claude Code.

## Step 1: Initialization

1. Create a folder `DTF_Prototype`.
2. Create a virtual environment: `python3 -m venv venv` and `source venv/bin/activate`.
3. **Prompt Claude:** "I am initializing the DTF project. Please read the `DTF_Master_Spec.md` file. First, create the directory structure and the `requirements.txt` file (include torch, torchvision, torchaudio, numpy, opencv-python, pillow, matplotlib, tqdm)."

## Step 2: The Data Factory (Phase 1)

*You need to verify the data generation before training anything.*

1. **Prompt Claude:** "Implement `src/core/schemas.py` and `src/generators/generator.py` (Module 1). Create a test script to generate 5 random JSON design briefs and print them."
2. **Prompt Claude:** "Implement `src/generators/renderer.py` and `src/generators/extractor.py`. Create a visualization script that saves a side-by-side image of the Rendered Image (`P_Image`) and the Feature Tensor masks (`F_Tensor`)."
    * *Verification:* Open the generated images. Does the "Text Mask" perfectly overlap the text in the image? If not, tell Claude to fix coordinate mapping.
3. **Prompt Claude:** "Implement `src/generators/grammar.py`. Run it on the generated tensors and output the 4 grammar scores. Verify that 'messy' designs get lower scores than 'aligned' designs."
4. **Prompt Claude:** "Create `generate_dataset.py`. Generate 10,000 samples and save them to the `data/` folder. Use a progress bar."

## Step 3: Training the Experts (Phase 2)

*Train these sequentially. Do not try to do all at once.*

1. **Encoder:**
    * **Prompt:** "Implement `src/models/encoder.py` and `train_scripts/train_encoder.py`. Use a standard U-Net. Ensure the input is 3 channels and output is 4 channels. Train for 5 epochs on the generated data using MPS."
2. **Abstractor:**
    * **Prompt:** "Implement `src/models/abstractor.py` and `train_scripts/train_abstractor.py`. Train it to predict the `V_Grammar` scores from `F_Tensors`."
3. **Decoder (The Hard Part):**
    * **Prompt:** "Implement `src/models/decoder.py` using a Conditional DDPM. Implement `train_scripts/train_decoder.py`. This training will take time. Create a sampling script to generate an image from a random `V_Meta` vector to test it."

## Step 4: The "Magic" (Phase 3)

1. **Prompt:** "Now for Module 9. Create `src/integration/innovate.py`. Load all three pre-trained models. Implement the latent space gradient ascent loop to optimize a random noise vector into a high-scoring design."
2. **Execution:** Run the script. Watch the loss values. If the Grammar Score goes **up** and the Loss goes **down**, your AI is designing!

## Troubleshooting for M3 Pro (MPS)

* **Error:** `Placeholder storage has not been allocated on MPS device`
  * **Fix:** Tell Claude: "Ensure all tensors are explicitly moved to `.to(device)` before operations."
* **Error:** `Op not implemented on MPS`
  * **Fix:** Tell Claude: "Set `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'` in the main script."
