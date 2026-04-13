from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch
import numpy as np
import cv2
import base64
from fastapi.responses import Response

from src.pipeline.remove_object import ObjectRemover

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lấy dir ảnh để đồng bộ cho inference
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(project_root, "data")

output_dir = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/outputs/demo"

# CKPT paths
YOLOV8x_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/yolov8x.pt"
YOLOV8m_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/yolov8/finetuned_yolo_v8/train_12_classes_v2/weights/epoch45.pt"
SAM_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
LAMA_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/lama/big-lama.pt"

# global variables for pipeline models
pipeline = None

# Start pipeline object remover (once only)
@app.on_event("startup")
def load_models():
    global pipeline
    try:
        print("🚀 Đang khởi tạo ObjectRemover Pipeline...")
        pipeline = ObjectRemover(
            yolov8x_path=YOLOV8x_PATH,
            yolov8m_finetuned_path=YOLOV8m_PATH, 
            sam_path=SAM_PATH, 
            lama_path=LAMA_PATH)
        print("✅ Pipeline đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo pipeline: {e}")
        raise e

class ROIBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class TargetObject(BaseModel):
    box: list[float]  # Mảng [x1, y1, x2, y2]
    label: str
    score: float = 0.0

class ScanPayload(BaseModel):
    image_name: str
    boxes: list[ROIBox]

class RemovePayload(BaseModel):
    image_name: str
    target_boxes: list[TargetObject]

@app.post("/submit_boxes")
def submit_boxes(data: ScanPayload):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng. Vui lòng thử lại sau.")
    
    # --- HÀM TÌM KIẾM ĐỆ QUY TRONG THƯ MỤC DATA ---
    def find_file_recursive(base_path, filename):
        for root, dirs, files in os.walk(base_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

    full_path = find_file_recursive(IMAGE_DIR, data.image_name)
    
    if full_path is None:
        raise HTTPException(status_code=404, detail=f"❌ Không tìm thấy ảnh '{data.image_name}' trong '{IMAGE_DIR}' hoặc các thư mục con!")

    file_exists = True 

    output_txt_path = os.path.join(project_root, "img_path.txt")
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_path)
        print(f"✅ Đã ghi đường dẫn vào: {output_txt_path}")
    except Exception as e:
        print(f"❌ Lỗi ghi file: {e}")

    print(f"Dữ liệu nhận được cho ảnh: {full_path}")
    print(f"boxes: {data.boxes}")
    
    # Ép kiểu và gọi pipeline
    roi_boxes_dict = [obj.model_dump() for obj in data.boxes] if data.boxes else None
    detected_results = pipeline.scan_for_objects(roi_boxes_dict)

    return {
        "status": "success",
        "num_boxes": len(data.boxes),
        "absolute_path": full_path,
        "file_exists": file_exists,
        "message": f"Dữ liệu đã sẵn sàng cho pipeline tại {full_path}",
        "objects": detected_results
    }

@app.post("/remove_objects")
def remove_objects(data: RemovePayload):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng. Vui lòng thử lại sau.")
    
    target_boxes_dict = [obj.model_dump() for obj in data.target_boxes]

    mask_uint8, lama_rgb, final_img_rgb = pipeline.remove_objects(target_boxes_dict)

    output_img_path = os.path.join(output_dir, f"removed_{data.image_name}")
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR))
    print(f"✅ Ảnh đã được lưu tại: {output_img_path}")

    def img_to_b64(img_arr, is_rgb=True):
        if is_rgb:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.png', img_arr)
        if not ret:
            raise HTTPException(status_code=500, detail="Lỗi mã hóa ảnh.")
        return base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "mask_b64": img_to_b64(mask_uint8, is_rgb=False), # Mask là đen trắng (grayscale)
        "lama_b64": img_to_b64(lama_rgb),
        "sdxl_b64": img_to_b64(final_img_rgb)
    }
    