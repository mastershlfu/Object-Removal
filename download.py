import os
from huggingface_hub import snapshot_download

# Định nghĩa đường dẫn đích trên ổ 4TB
DEST_PATH = "/media/ml4u/Challenge-4TB/baonhi/sdxl-inpaint-offline"

snapshot_download(
    repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    local_dir=DEST_PATH,
    allow_patterns=[
        "*.json",           # Tải tất cả file config
        "*.txt",            # Tải các file vocab
        "*fp16.safetensors" # CHỈ tải trọng số bản fp16
    ],
    ignore_patterns=[
        "*.bin", 
        "*.onnx", 
        "*.pb", 
        "*non_ema*"
    ]
)

print(f"✅ Tải xong hoàn toàn! Model đã nằm tại: {DEST_PATH}")