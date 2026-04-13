import wandb
import os
from ultralytics import YOLO

os.environ["WANDB_PROJECT"] = "Finetune-Yolov8-12-classes" 

yaml_path = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/fine_tune_yolov8_v2/data.yaml"

save_path = "/media/ml4u/Challenge-4TB/baonhi/finetuned_yolo_v8/"

run_name = "train_12_classes_v2"

# model_path = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/yolov8/finetuned_yolo_v8/train_12_classes_v14/weights/last.pt"
model_path = "yolov8m.pt"
# wandb.init(
#     project="Finetune-Yolov8-12-classes", # Tên dự án trên web WandB
#     name=run_name,                    # Tên của biểu đồ (giống tên thư mục local cho dễ nhớ)
#     job_type="training"
# )
model = YOLO(model_path)

if __name__ == '__main__':
    print("Training custom YOLOv8m...")
    results = model.train(
        # resume=True,
        data=yaml_path,
        epochs=200,        
        imgsz=640,         
        batch=32,          
        patience=15,       
        device=0,          
        workers=8,   
         
        project=save_path, 
        name=run_name,             

        translate=0.5,  
        scale=0.5,     
        fliplr=0.5,     

        save_period=5,     
    )
    # wandb.finish()
    print("Done training")