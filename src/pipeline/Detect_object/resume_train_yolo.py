import os
from ultralytics import YOLO

LAST_WEIGHT_PATH = "/media/ml4u/Challenge-4TB/baonhi/yolo11_lvis_project/yolo11m_lvis_final_rfs/weights/last.pt"

if not os.path.exists(LAST_WEIGHT_PATH):
    print("weights not found")
else:
    model = YOLO(LAST_WEIGHT_PATH)
    print("Resume training")
    
    model.train(resume=True)