import os
os.environ["HF_HOME"] = "/media/ml4u/Challenge-4TB/baonhi/hf_cache"

import json
import torch

import cv2 
import numpy as np

from PIL import Image
from huggingface_hub import constants
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGE_DIR = "/media/ml4u/Challenge-4TB/baonhi/Places2/qwen_inputs_red_outline" 
OUTPUT_JSONL = "/media/ml4u/Challenge-4TB/baonhi/Places2/dataset_labels.jsonl"
MASK_DIR = "/media/ml4u/Challenge-4TB/baonhi/Places2/ground_truth_masks"

SYSTEM_INSTRUCTION = """
You are a strict Prompt Engineering expert for a Stable Diffusion Inpainting system.
Look at the image. You will see EXACTLY ONE object enclosed in a THICK RED OUTLINE.

YOUR TASK:
Imagine the object inside the red outline is completely cut out, leaving a HOLE. Your goal is to describe ONLY the exact materials, colors, and surfaces needed to fill that specific HOLE.

Step 1 (Thought Process):
- "Target:" Identify the red-outlined object (so you NEVER mention it).
- "Hidden Background:" What specific surfaces/materials are physically hidden BEHIND the target? (This is what you must draw).
- "Exclude:" Identify objects or scenery that are nearby or touching the border but are NOT blocked by the target (e.g., adjacent chairs, the sky in the distance, people standing next to it). You MUST NOT describe these.

Step 2 (Prompt Generation):
- Write a comma-separated list of descriptive keywords focusing EXCLUSIVELY on the "Hidden Background".
- When describing any texture, always try to include its color and material.
- ALWAYS start exactly with: "empty, seamless, photorealistic, high quality, "

🚨 STRICT CONSTRAINTS:
1. FOCUS ONLY INSIDE THE HOLE: Do not describe the surroundings, adjacent objects, or the general room.
2. IGNORE SCENERY: Do NOT mention the sky, trees, or distant objects unless the target is directly blocking them.
3. THE CONTINUATION RULE: You may ONLY describe objects, surfaces, or textures if they physically TOUCH the outside edge of the red outline AND clearly continue behind the target. Do NOT invent isolated objects floating inside the hole.
4. BE EXTREMELY SPECIFIC: Mention colors and materials (e.g., "dark oak wood floor", "smooth white plaster wall").
5. NO HALLUCINATION: Only describe visible textures continuing behind the object.
6. NO FORBIDDEN WORDS: DO NOT mention the red-outlined object.
7. ANTI-LOOP: STOP generating text immediately after writing the Prompt line.

OUTPUT FORMAT:
Thought: Target is [X]. Hidden Background is [Y]. Exclude [Z].
Prompt: empty, seamless, photorealistic, high quality, [ONLY the hidden background keywords]
"""


def get_cropped_image_by_mask(img_path, mask_path, padding_px=100):
    """
    Sử dụng trực tiếp Mask của YOLO để tìm Bounding Box và cắt ảnh.
    Nhanh, chuẩn xác và không bị nhiễu bởi màu sắc trong ảnh gốc.
    """
    # 1. Load ảnh gốc (có viền đỏ)
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        raise FileNotFoundError(f"Lỗi đọc ảnh gốc: {img_path}")
    h_orig, w_orig = cv_img.shape[:2]

    # 2. Load Mask của YOLO (Dạng Grayscale đen trắng)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"⚠️ Không tìm thấy Mask: {mask_path}. Dùng ảnh gốc.")
        return Image.open(img_path).convert("RGB")

    # 3. Tìm tọa độ ngay lập tức từ các pixel trắng (>0) của Mask
    coords = cv2.findNonZero(mask_img)
    if coords is None:
        print(f"⚠️ Mask trống (không có vật thể): {mask_path}. Dùng ảnh gốc.")
        return Image.open(img_path).convert("RGB")

    # 4. Lấy Bounding Box
    x_box, y_box, w_box, h_box = cv2.boundingRect(coords)

    # 5. Thêm Padding (mở rộng vùng cắt)
    x1 = max(0, x_box - padding_px)
    y1 = max(0, y_box - padding_px)
    x2 = min(w_orig, x_box + w_box + padding_px)
    y2 = min(h_orig, y_box + h_box + padding_px)

    # 6. Cắt ảnh gốc và chuyển sang PIL cho Qwen
    cropped_cv = cv_img[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(cropped_rgb)
# ==========================================
# TẢI MODEL QWEN 7B 
# ==========================================
print("⏳ Đang tải mô hình Qwen2.5-VL-7B lên GPU/CPU...")
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"📂 Thư mục Cache hiện tại: {constants.HF_HUB_CACHE}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", # Cho phép tràn VRAM xuống CPU RAM
)
processor = AutoProcessor.from_pretrained(model_id)
print("✅ Tải mô hình thành công!")

# ==========================================
# CƠ CHẾ RESUME
# ==========================================
processed_images = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                processed_images.add(data['image_file']) # Sửa key cho khớp
    print(f"🔄 Đã load {len(processed_images)} ảnh từ lần chạy trước.")

print(f"🚀 Bắt đầu quét thư mục: {IMAGE_DIR}")
all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images_to_process = [f for f in all_images if f not in processed_images]
total_images = len(all_images)
count = len(processed_images)

# ==========================================
# VÒNG LẶP XỬ LÝ
# ==========================================
with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f:
    for img_name in images_to_process:
        img_path = os.path.join(IMAGE_DIR, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        
        try:
            mask_path = os.path.join(MASK_DIR, mask_name)
            image = Image.open(img_path).convert("RGB")
            
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": [
                    {
                        "type": "image", 
                        "image": image,
                        "max_pixels": 589824 # Ép size
                    },
                    {
                        "type": "text", 
                        "text": "Analyze the area behind the red outline and generate the Thought and Prompt."
                    }
                ]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=150, # Tăng token để nó có chỗ viết Thought
                repetition_penalty=1.1,  
                temperature=0.4,          
                do_sample=False           
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            # TÁCH LẤY THOUGHT VÀ PROMPT
            if "Prompt:" in raw_output:
                reasoning = raw_output.split("Prompt:")[0].replace("Thought:", "").strip()
                prompt_text = raw_output.split("Prompt:")[-1].strip()
            else:
                reasoning = "N/A"
                prompt_text = raw_output.replace("Prompt:", "").strip()
            
            # Xóa dấu phẩy thừa
            prompt_text = prompt_text.rstrip(',')

            # Lưu kết quả
            data_point = {
                "image_file": img_name,
                "mask_file": mask_name,
                "thought": reasoning,
                "ground_truth_prompt": prompt_text
            }
            f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
            f.flush()
            
            count += 1
            print(f"[{count}/{total_images}] Thành công: {img_name}", flush=True)
            print(f"  🧠 Thought: {reasoning}", flush=True)
            print(f"  👉 Prompt: {prompt_text}\n", flush=True)
            
        except Exception as e:
            print(f"❌ Lỗi xử lý {img_name}: {e}")

print("✅ Hoàn thành toàn bộ dataset!")