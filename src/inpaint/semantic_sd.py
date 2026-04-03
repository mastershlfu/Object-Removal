import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

class SemanticInpainter:
    def __init__(self, device="cuda"):
        print("⏳ Loading Stable Diffusion Inpainting...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"       
        ).to(device)
        self.pipe.enable_attention_slicing()

    def inpaint(self, image_rgb, mask_alpha, prompt="background, high quality"):
        # SD cần mask uint8 (255=xóa)
        mask_uint8 = (mask_alpha * 255).astype(np.uint8)
        
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(mask_uint8).convert("L")

        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            
        return np.array(output.resize((image_rgb.shape[1], image_rgb.shape[0])))