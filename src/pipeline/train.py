import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import numpy as np
import wandb
from datetime import datetime
from src.utils.logger import Logger


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PROJECT_NAME = "Object Removal"
OUTPUT_DIR = "./models/faster_rcnn_logs"
#RESUME_CHECKPOINT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/fasterrcnn_epoch_0.pth"
RESUME_CHECKPOINT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/maskrcnn_epoch_0.pth"
BATCH_SIZE = 8 
NUM_EPOCHS = 10
LR = 0.005

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# --- CUSTOM DATASET  ---
class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        
        blockPrint()
        
        self.ids = []
        for img_id in sorted(self.coco.imgs.keys()):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.ids.append(img_id)
                
        enablePrint()
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_np = np.array(img)
        
        img_h, img_w, _ = img_np.shape

        boxes = []
        labels = []
        
        for i in range(len(coco_annotation)):
            x_coco, y_coco, w_coco, h_coco = coco_annotation[i]['bbox']
            
            xmin = x_coco
            ymin = y_coco
            xmax = x_coco + w_coco
            ymax = y_coco + h_coco
            
            xmin = max(0, min(xmin, img_w - 1))
            ymin = max(0, min(ymin, img_h - 1))
            xmax = max(0, min(xmax, img_w - 1))
            ymax = max(0, min(ymax, img_h - 1))
            
            if (xmax - xmin) > 1 and (ymax - ymin) > 1:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(coco_annotation[i]['category_id'])

        target = {}
        
        if self.transforms:
            if len(boxes) > 0:
                try:
                    transformed = self.transforms(image=img_np, bboxes=boxes, labels=labels)
                    img_tensor = transformed['image']
                    new_boxes = transformed['bboxes']
                    new_labels = transformed['labels']
                except ValueError as e:
                    transform_only = A.Compose([
                        A.Resize(800, 800),
                        ToTensorV2()
                    ])
                    transformed = transform_only(image=img_np)
                    img_tensor = transformed['image']
                    new_boxes = []
                    new_labels = []
            else:
                transform_only = A.Compose([
                    A.Resize(800, 800),
                    ToTensorV2()
                ])
                transformed = transform_only(image=img_np)
                img_tensor = transformed['image']
                new_boxes = []
                new_labels = []
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            new_boxes = boxes
            new_labels = labels

        if len(new_boxes) > 0:
            target["boxes"] = torch.as_tensor(new_boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.as_tensor(new_labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            
        target["image_id"] = torch.tensor([img_id])
        
        if isinstance(img_tensor, torch.Tensor) and img_tensor.max() > 1.0:
             img_tensor = img_tensor.float() / 255.0

        return img_tensor, target

    def __len__(self):
        return len(self.ids)

# gộp batch 
def collate_fn(batch):
    return tuple(zip(*batch))

# --- BUILD MODEL ---
def get_model(num_classes):
    #model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model = maskrcnn_resnet50_fpn_v2(weights="None")
    # Thay Head để phù hợp số class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    #Thay doi MaskPredictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# --- TRAINING LOOP ---
def main():
    #log = Logger(output_dir=OUTPUT_DIR, name="FasterRCNN_Train")
    log = Logger(output_dir=OUTPUT_DIR, name="MaskRCNN_Train")
    log.info("==========================================")
    log.info(f"   STARTING TRAINING PIPELINE: {PROJECT_NAME}")
    log.info(f"   Device: {DEVICE}")
    log.info(f"   Batch Size: {BATCH_SIZE}")
    log.info("==========================================")
    
    wandb.init(
        project=PROJECT_NAME,
        name="faster-rcnn-resnet50-run-1",
        config={
            "learning_rate": LR,
            "architecture": "Faster R-CNN ResNet50",
            "dataset": "COCO-2017",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE
        }
    )
    
    # Transform
    train_transform = A.Compose([
        A.Resize(800, 800), 
        A.HorizontalFlip(p=0.5),
        ToTensorV2() # (H, W, C) -> (C, H, W)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    log.info("  Loading Dataset...")
    # Dataset
    dataset = COCODataset(
        root='/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/train2017',
        annFile='/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/annotations/instances_train2017.json',
        transforms=train_transform
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    
    log.info(f"  Dataset Loaded. Total images: {len(dataset)}")

    
    log.info("  Building Model Faster R-CNN ResNet50...")
    # Model Setup
    model = get_model(num_classes=91)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    
    start_epoch = 0
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        log.info(f"  Resuming training from: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        
        # Load weights model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            log.info(f"  Loaded checkpoint. Resuming from Epoch {start_epoch + 1}")
        else:
            # Fallback nếu lỡ dùng file .pth cũ (chỉ chứa weights)
            model.load_state_dict(checkpoint)
            log.warning("  Old checkpoint format detected (weights only). Optimizer state reset.")
    else:
        log.info("  Starting training from scratch...")

    log.info("  Training Loop Started...")
    wandb.watch(model, log="all")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = datetime.now()
        
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            wandb.log({
                "batch_loss": losses.item(),
                "cls_loss": loss_dict['loss_classifier'].item(),
                "box_loss": loss_dict['loss_box_reg'].item()
            })

            if i % 50 == 0:
                log.info(f"   Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{i}/{len(data_loader)}] | Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(data_loader)
        duration = datetime.now() - start_time
        
        log.info(f"  EPOCH {epoch+1} FINISHED")
        log.info(f"   Duration: {duration}")
        log.info(f"   Avg Loss: {avg_loss:.4f}")
        
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch+1})
        
        # [RESUME]
        ckpt_path = os.path.join(OUTPUT_DIR, f"maskrcnn_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        log.info(f"  Saved Full Checkpoint: {ckpt_path}")
        
    wandb.finish()
    log.info(" TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()