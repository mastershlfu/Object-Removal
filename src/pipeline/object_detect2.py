import torch
import numpy as np
import cv2
import os
import sys
from PIL import Image
import albumentations as A 
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from lvis import LVIS
from pathlib import Path
# Đảm bảo đường dẫn import đúng
from src.pipeline.Detect_object.R_FCN import R_FCN
from src.utils.logger import Logger

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "Object Detection"
OUTPUT_DIR = "./models/r_fcn_logs"
BATCH_SIZE = 4 # Giảm xuống 4 cho an toàn bộ nhớ GPU
NUM_EPOCHS = 10
NUM_CLASSES = 1204
LR = 0.005
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_PATH = Path.cwd()
print(f"base path: {BASE_PATH}")
LVIS_ANN_PATH = os.path.join(BASE_PATH, 'data' , 'coco','annotations','lvis_v1_train.json')
COCO_IMG_ROOT = os.path.join(BASE_PATH,'data' , 'coco', 'train2017' )

class LVISDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.lvis = LVIS(annFile)
        self.ids = list(self.lvis.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        coco_annotation = self.lvis.load_anns(ann_ids)
        img_info = self.lvis.load_imgs([img_id])[0]
        
        fname = img_info.get('file_name') or img_info.get('coco_url', '').split('/')[-1]
        
        if not fname:
            raise KeyError(f"Không tìm thấy file_name trong img_id: {img_id}. Các keys hiện có: {img_info.keys()}")
        
        if 'train2017' in fname:
            full_path = os.path.join(os.path.dirname(self.root), fname)
        else:
            full_path = os.path.join(self.root, fname)
            
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        for ann in coco_annotation:
            xmin, ymin, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                # Lưu ý: category_id trong LVIS bắt đầu từ 1 đến 1203
                labels.append(int(ann['category_id']))

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return img / 255.0, target

    def __len__(self):
        return len(self.ids)

class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = [img_id for img_id in sorted(self.coco.imgs.keys()) 
                    if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        for ann in coco_annotation:
            xmin, ymin, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(int(ann['category_id'])) # Ép kiểu int tại đây

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        if len(boxes) == 0: # Tránh lỗi batch trống
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return img / 255.0, target # Chuẩn hóa ảnh về [0, 1]

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))
def train():
    log = Logger(output_dir=OUTPUT_DIR, name="R_FCN_Train")
    log.info(f"STARTING TRAINING on {DEVICE}")
    
    wandb.init(project=PROJECT_NAME, name=f"R_FCN-{datetime.now().strftime('%m%d-%H%M')}")
    
    train_transform = A.Compose([
        A.Resize(800, 800),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    dataset = LVISDataset(
        root=COCO_IMG_ROOT,
        annFile=LVIS_ANN_PATH,
        transforms=train_transform
    )
    
    # Giảm num_workers xuống 0 hoặc 2 nếu bạn gặp vấn đề về bộ nhớ/deadlock trên máy local
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Đảm bảo NUM_CLASSES truyền vào đúng
    model = R_FCN(num_classes=NUM_CLASSES) 
    model.to(DEVICE)

    # KIỂM TRA ĐẠO HÀM: In ra số lượng tham số để chắc chắn model không bị "đóng băng"
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        raise RuntimeError("Model không có tham số nào để học! Kiểm tra lại requires_grad trong R_FCN.py")

    optimizer = torch.optim.SGD(trainable_params, lr=LR, momentum=0.9, weight_decay=0.0005)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        
        for images, targets in pbar:
            # Chuyển dữ liệu lên GPU
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Tính tổng loss - Đảm bảo các loss này vẫn giữ liên kết grad_fn
            # losses = sum(loss for loss in loss_dict.values())
            losses = loss_dict["loss_classifier"] + loss_dict["loss_box_reg"]
            optimizer.zero_grad()
            # Backward pass
            losses.backward()
            # Clip gradient (Tùy chọn nhưng nên có để tránh bùng nổ gradient với R-FCN tự viết)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())
            wandb.log({"loss": losses.item()})

        log.info(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(data_loader):.4f}")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "R_FCN_latest.pth"))

    wandb.finish()

if __name__ == "__main__":
    train()