import os
import sys
sys.path.insert(0, "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal")

import cv2
import numpy as np
import torch
from src.inpaint.lama import LaMaInpainter

PROJECT_ROOT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal"

LAMA_PATH = os.path.join(PROJECT_ROOT, "models/lama/big-lama.pt")
IMG_PATH = os.path.join(PROJECT_ROOT, "data/bracelet3.png")
MASK_PATH = os.path.join(PROJECT_ROOT, "outputs/sam_demo/3_sam_mask_bracelet.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/lama_demo")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 50)
    print("LaMa Inpainting Demo")
    print("=" * 50)
    
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    if not os.path.exists(MASK_PATH):
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")
    
    print(f"\nLoading image: {IMG_PATH}")
    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    
    print(f"\nLoading mask: {MASK_PATH}")
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    print(f"Mask size: {mask.shape[1]}x{mask.shape[0]}")
    
    print(f"\nLoading LaMa model...")
    lama = LaMaInpainter(LAMA_PATH, device=DEVICE)
    
    print(f"\nRunning LaMa inpainting...")
    result = lama.inpaint(img_rgb, mask)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_original.jpg"), img_bgr)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_mask.jpg"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_lama_result.jpg"), result_bgr)
    
    print(f"\nSaved to: {OUTPUT_DIR}")
    print("1. 1_original.jpg - Original image")
    print("2. 2_mask.jpg - SAM mask")
    print("3. 3_lama_result.jpg - LaMa inpainting result")
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()