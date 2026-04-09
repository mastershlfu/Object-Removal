import os
from ultralytics import YOLO

# 1. Cấu hình folder làm việc trên ổ 4TB
SAFE_DIR = "/media/ml4u/Challenge-4TB/baonhi/yolo11_lvis_project"
os.makedirs(SAFE_DIR, exist_ok=True) 

# Offline mode để log lưu tại máy, không sợ mất mạng
os.environ["WANDB_MODE"] = "offline" 
os.environ["WANDB_DIR"] = SAFE_DIR

def train_lvis_final():
    # 2. Load model, pre-trained yolo11m-seg.pt
    weight_path = f"{SAFE_DIR}/yolo11m_lvis_final_rfs/weights/last.pt"
    model = YOLO(weight_path) 

    print(f"🚀 [TRAINING] 200 lớp - Lưu mỗi epoch - Có Log Validation")
    
    # 3. Cấu hình Train tối ưu
    model.train(
        resume=True,
        data="lvis.yaml",        # Check kỹ: trỏ đúng train_rfs.txt và val_200.txt
        epochs=200,
        imgsz=640,
        batch=16,                
        project=SAFE_DIR,   
        name='yolo11m_lvis_final_rfs', 
        device=0,
        
        # CẤU HÌNH LƯU TRỮ & LOG
        save=True,
        save_period=5,           
        cache=True,              # Ổ 4TB thì bật cache cho nhanh
        val=True,                # Bật Val để có log mAP báo cáo thầy
        plots=True,              # Vẽ biểu đồ kết quả tự động
        
        exist_ok=True,
        lr0=0.01,                
        lrf=0.01,
        warmup_epochs=3,         
        workers=4,               
        amp=True       
    )

if __name__ == "__main__":
    train_lvis_final()