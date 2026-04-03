import os
from ultralytics import YOLO

# 1. Cấu hình folder làm việc trên ổ 4TB
SAFE_DIR = "/media/ml4u/Challenge-4TB/baonhi/yolo11_lvis_project"
os.makedirs(SAFE_DIR, exist_ok=True) 

# Offline mode để log lưu tại máy, không sợ mất mạng
os.environ["WANDB_MODE"] = "offline" 
os.environ["WANDB_DIR"] = SAFE_DIR

def train_lvis_final():
    # 2. Load model pre-trained yolo11m-seg
    model = YOLO("yolo11m-seg.pt") 

    print(f"🚀 [TRAINING] 200 lớp - Lưu mỗi epoch - Có Log Validation")
    
    # 3. Cấu hình Train tối ưu
    model.train(
        data="lvis.yaml",        # Check kỹ: trỏ đúng train_rfs.txt và val_200.txt
        epochs=200,
        imgsz=640,
        batch=32,                # Tận dụng 24GB VRAM của 4090
        project=SAFE_DIR,   
        name='yolo11m_lvis_final_rfs', 
        device=0,
        
        # CẤU HÌNH LƯU TRỮ & LOG
        save=True,
        save_period=1,           # ĐÚNG Ý BÀ: Xong mỗi epoch là lưu ngay 1 file backup
        cache=True,              # Ổ 4TB thì bật cache cho nhanh
        val=True,             # Bật Val để có log mAP báo cáo thầy
        plots=True,              # Vẽ biểu đồ kết quả tự động
        
        exist_ok=True,
        lr0=0.01,                
        lrf=0.01,
        warmup_epochs=3,         
        workers=8,               
        amp=True                 
    )

if __name__ == "__main__":
    train_lvis_final()