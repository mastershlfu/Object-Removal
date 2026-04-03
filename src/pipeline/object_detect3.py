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

# Đảm bảo đường dẫn import đúng
from src.pipeline.Detect_object.G_RCNN import G_RCNN
from src.utils.logger import Logger

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "Object Detection"
OUTPUT_DIR = "../models/g_rcnn_logs"
print(f"Output dir: {OUTPUT_DIR}")
BATCH_SIZE = 4 # Giảm xuống 4 cho an toàn bộ nhớ GPU
NUM_EPOCHS = 10
NUM_CLASSES = 1204
LR = 0.005
os.makedirs(OUTPUT_DIR, exist_ok=True)

LVIS_ANN_PATH = './data/coco/annotations/lvis_v1_train.json'
COCO_IMG_ROOT = './data/coco/train2017'

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
        
        # LVIS v1 file_name thường là "train2017/000000123456.jpg" 
        # hoặc chỉ là "000000123456.jpg". Ta cần xử lý để tránh trùng lặp path.
        fname = img_info['file_name']
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
    log = Logger(output_dir=OUTPUT_DIR, name="G_RCNN_Train")
    log.info(f"STARTING TRAINING on {DEVICE}")
    
    wandb.init(project=PROJECT_NAME, name=f"G_RCNN-{datetime.now().strftime('%m%d-%H%M')}")
    
    train_transform = A.Compose([
        A.Resize(800, 800),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # dataset = COCODataset(
    #     root='/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/train2017',
    #     annFile='/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/annotations/instances_train2017.json',
    #     transforms=train_transform
    # )
    
    dataset = LVISDataset(
        root=COCO_IMG_ROOT,
        annFile=LVIS_ANN_PATH,
        transforms=train_transform
    )
    
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = G_RCNN(num_classes=91) # COCO có 80 lớp + 1 background (tổng 91 ID)
    model.to(DEVICE)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=0.0005)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        
        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())
            wandb.log({"loss": losses.item()})

        log.info(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(data_loader):.4f}")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "G_RCNN_latest.pth"))

    wandb.finish()

if __name__ == "__main__":
    train()