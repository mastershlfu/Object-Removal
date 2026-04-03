from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch
import numpy as np
import cv2
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
IMAGE_DIR = os.path.join(project_root, "data", "LaMa_test_images", "images")

output_dir = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/outputs/demo"

# CKPT paths
RCNN_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/fasterrcnn_epoch_7.pth"
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
            rcnn_path=RCNN_PATH, 
            sam_path=SAM_PATH, 
            lama_path=LAMA_PATH)
        print("✅ Pipeline đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo pipeline: {e}")
        raise e

class Box(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = ""

class ScanPayload(BaseModel):
    image_name: str
    boxes: list[Box]

class RemovePayload(BaseModel):
    image_name: str
    target_boxes: list[Box]

@app.post("/submit_boxes")
def submit_boxes(data: ScanPayload):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng. Vui lòng thử lại sau.")
    # ghep path
    full_path = os.path.abspath(os.path.join(IMAGE_DIR, data.image_name))
    
    # kiem tra path
    file_exists = os.path.exists(full_path)
    output_txt_path = os.path.join(project_root, "img_path.txt")
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_path)
        print(f"✅ Đã ghi đường dẫn vào: {output_txt_path}")
    except Exception as e:
        print(f"❌ Lỗi ghi file: {e}")

    print(f"Dữ liệu nhận được cho ảnh: {full_path}")
    print(f"boxes: {data.boxes}")
    print(f"Full path: {full_path}")
    
    detected_results = []
    detected_results = pipeline.scan_for_objects(data.boxes)

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
    
    final_img_rgb = pipeline.remove_objects(data.target_boxes)

    output_img_path = os.path.join(output_dir, f"removed_{data.image_name}")
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

    cv2.imwrite(output_img_path, cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR))

    print(f"✅ Ảnh đã được lưu tại: {output_img_path}")

    final_img_bgr = cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR)
    ret, buffer = cv2.imencode('.png', final_img_bgr)

    if not ret:
        raise HTTPException(status_code=500, detail="Không thể mã hóa ảnh.")
    
    return Response(content=buffer.tobytes(), media_type="image/png")
    