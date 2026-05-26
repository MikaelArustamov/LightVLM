"""SD 1.5 fp16 image generation — local, 2.5GB, good quality."""

import io
import base64

import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    def __init__(self):
        print("[Generator] Loading SD 1.5 fp16 (~2.5GB)...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            variant="fp16",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if torch.cuda.is_available():
            self.pipe.to("cuda")
            print("[Generator] Using CUDA")
        else:
            self.pipe.to("cpu")
            print("[Generator] Using CPU")

        # Optimization for CPU
        self.pipe.enable_attention_slicing()

        print("[Generator] SD 1.5 ready")

    def generate_to_base64(self, prompt: str) -> str | None:
        print(f"[Generator] Generating: {prompt[:50]}...")

        try:
            image = self.pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            print(f"[Generator] Error: {e}")
            return None