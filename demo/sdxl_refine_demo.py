import os
import sys
sys.path.insert(0, "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal")

import cv2
import numpy as np
import torch
from PIL import Image

os.environ['HF_HOME'] = "/media/ml4u/Challenge-4TB/baonhi"

PROJECT_ROOT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal"

LAMA_RESULT_PATH = os.path.join(PROJECT_ROOT, "outputs/lama_demo/3_lama_result.jpg")
MASK_PATH = os.path.join(PROJECT_ROOT, "outputs/lama_demo/2_mask.jpg")
ORIGINAL_PATH = os.path.join(PROJECT_ROOT, "outputs/lama_demo/1_original.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/sdxl_demo")

SDXL_PATH = "/media/ml4u/Challenge-4TB/baonhi/sdxl-inpaint-offline"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 50)
    print("SDXL Refinement Demo")
    print("=" * 50)
    
    # Load images
    print(f"\nLoading LaMa result: {LAMA_RESULT_PATH}")
    lama_result = cv2.imread(LAMA_RESULT_PATH)
    lama_result = cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB)
    print(f"LaMa result size: {lama_result.shape[1]}x{lama_result.shape[0]}")
    
    print(f"Loading original: {ORIGINAL_PATH}")
    original = cv2.imread(ORIGINAL_PATH)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    print(f"Loading mask: {MASK_PATH}")
    raw_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    
    kernel_dilate = np.ones((11, 11), np.uint8) 
    mask = cv2.dilate(raw_mask, kernel_dilate, iterations=1)
    
    # Load SDXL pipeline
    print(f"\nLoading SDXL model from: {SDXL_PATH}")
    from diffusers import StableDiffusionXLInpaintPipeline
    
    sdxl_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        SDXL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True
    ).to(DEVICE)
    
    sdxl_pipe.enable_model_cpu_offload()
    print("SDXL loaded successfully!")
    
    # Prepare input
    lama_pil = Image.fromarray(lama_result)
    mask_pil = Image.fromarray(mask)
    
    w_org, h_org = lama_pil.size
    w = w_org - (w_org % 8)
    h = h_org - (h_org % 8)
    
    if (w, h) != lama_pil.size:
        lama_pil = lama_pil.resize((w, h))
        mask_pil = mask_pil.resize((w, h))
    
    # Dynamic negative prompt from labels
    dynamic_neg_prompt = "Bracelet"
    
    # First pass - high strength
    # Chuẩn bị mask để Blend (làm nhòe viền)
    blend_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = (blend_mask.astype(float) / 255.0)
    alpha = np.expand_dims(alpha, axis=-1)
    
    # ---------------------------------------------------------
    # PASS 1: VẼ LẠI CHI TIẾT (Strength = 0.9)
    # ---------------------------------------------------------
    print("\nRunning SDXL refine pass 1 (strength=0.9)...")
    result1 = sdxl_pipe(
        prompt="clean architectural background, suitable context, clear surroundings, seamless texture, continuous surface, photorealistic, perfectly blended, highly detailed, matching lighting, natural continuation",
        negative_prompt=dynamic_neg_prompt + ", new objects, additional items, ghosts, shadows, distinct subjects, people, animals, vehicles, furniture, decor, text, watermark, artifacts, geometric shapes, blur",
        image=lama_pil,         # <--- Đầu vào là ảnh LaMa
        mask_image=mask_pil,
        num_inference_steps=100,
        guidance_scale=8.0,
        height=h,
        width=w,
        strength=0.9
    ).images[0]
    
    result1_np = np.array(result1.resize((w_org, h_org)))
    
    # Blend lần 1 để viền mượt lại trước khi đưa vào Pass 2
    pass1_blended = (result1_np * alpha) + (original * (1.0 - alpha))
    pass1_blended = np.clip(pass1_blended, 0, 255).astype(np.uint8)
    
    # ---------------------------------------------------------
    # PASS 2: KHỚP ÁNH SÁNG & LÀM MỊN VIỀN (Strength = 0.4)
    # ---------------------------------------------------------
    print("Running SDXL refine pass 2 (strength=0.4)...")
    
    # Phải convert ảnh đã blend của Pass 1 về PIL và resize cho SDXL
    pass1_pil = Image.fromarray(pass1_blended).resize((w, h))
    
    result2 = sdxl_pipe(
        prompt="clean architectural background, suitable context, clear surroundings, seamless texture, continuous surface, photorealistic, perfectly blended, highly detailed, matching lighting, natural continuation",
        negative_prompt=dynamic_neg_prompt + ", new objects, additional items, ghosts, shadows, distinct subjects, people, animals, vehicles, furniture, decor, text, watermark, artifacts, geometric shapes, blur",
        image=pass1_pil,        # <--- QUAN TRỌNG: Đầu vào là kết quả của Pass 1
        mask_image=mask_pil,
        num_inference_steps=100,
        guidance_scale=8.0,
        height=h,
        width=w,
        strength=0.4
    ).images[0]
    
    result2_np = np.array(result2.resize((w_org, h_org)))
    
    # Blend lần 2 (Final Blend)
    final_result = (result2_np * alpha) + (original * (1.0 - alpha))
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_original.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_lama_result.jpg"), cv2.cvtColor(lama_result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_sdxl_result.jpg"), cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    
    print(f"\nSaved to: {OUTPUT_DIR}")
    print("1. 1_original.jpg - Original image")
    print("2. 2_lama_result.jpg - LaMa result")
    print("3. 3_sdxl_result.jpg - SDXL refined result")
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()