"""
Stable Diffusion Decoder (Module 8)
Wraps Hugging Face Diffusers to generate designs from metadata prompts.
"""

import torch
from diffusers import StableDiffusionPipeline
from typing import Optional, List, Union, Dict

class SDDecoder:
    """
    Wrapper for Stable Diffusion to generate designs from V_Meta
    """

    def __init__(self,
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "mps",
                 use_auth_token: Union[bool, str] = False):
        """
        Initialize SD Pipeline

        Args:
            model_id: Hugging Face model ID
            device: 'cuda', 'mps', or 'cpu'
            use_auth_token: HF token if using gated models
        """
        self.device = device
        print(f"Loading Stable Diffusion model: {model_id} on {device}...")

        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32, # float16 causes NaNs on some MPS devices
                use_auth_token=use_auth_token,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(device)

            # Enable memory optimizations for Mac
            if device == 'mps':
                self.pipe.enable_attention_slicing()

            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def load_lora(self, lora_path: str):
        """
        Load LoRA weights into the pipeline
        """
        print(f"Loading LoRA weights from: {lora_path}")
        try:
            self.pipe.load_lora_weights(lora_path)
            print("✅ LoRA weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading LoRA: {e}")

    def meta_to_prompt(self, goal_id: int, format_id: int, tone: float) -> str:
        """
        Convert metadata to text prompt
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

    def generate(self,
                 v_meta: torch.Tensor,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5) -> List:
        """
        Generate images from metadata tensor

        Args:
            v_meta: [B, 3] tensor (Goal, Format, Tone)

        Returns:
            List of PIL Images
        """
        batch_size = v_meta.shape[0]
        prompts = []

        for i in range(batch_size):
            goal = int(v_meta[i, 0].item())
            fmt = int(v_meta[i, 1].item())
            tone = v_meta[i, 2].item()

            prompt = self.meta_to_prompt(goal, fmt, tone)
            prompts.append(prompt)
            print(f"  Prompt [{i}]: {prompt}")

        # Generate
        images = self.pipe(
            prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images

        return images

if __name__ == "__main__":
    # Simple test
    decoder = SDDecoder()

    # Test prompt generation
    print("\nTesting prompt generation:")
    print(decoder.meta_to_prompt(1, 0, 0.9))
