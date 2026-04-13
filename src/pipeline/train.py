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

OUTPUT_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/faster-rcnn_checkpoint"
RESUME_CHECKPOINT = None

BATCH_SIZE = 8 
NUM_EPOCHS = 24
LR = 0.0005  # LR for RPN and ROI heads (frozen backbone)
LR_BACKBONE = 0.001  # Higher LR for backbone after unfreeze
WARMUP_STEPS = 500  # Warmup steps
UNFREEZE_EPOCH = 3  # Epoch to unfreeze backbone
GRAD_CLIP = 1.0
NUM_CLASSES = 81  # COCO 80 classes + 1 background

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
        
        cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        
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
        
        # print(coco.loadImgs(img_id)[0])
        # assert False
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
                # labels.append(coco_annotation[i]['category_id'])
                labels.append(self.cat2label[coco_annotation[i]['category_id']])

        target = {}
        
        if self.transforms:
            if len(boxes) > 0:
                try:
                    transformed = self.transforms(image=img_np, bboxes=boxes, labels=labels)
                    img_tensor = transformed['image']
                    new_boxes = transformed['bboxes']
                    new_labels = transformed['labels']
                except ValueError as e:
                    # transform_only = A.Compose([
                    #     A.Resize(800, 800),
                    #     ToTensorV2()
                    # ])
                    transform_only = A.Compose([
                        A.LongestMaxSize(max_size=800),
                        A.PadIfNeeded(800, 800),
                        ToTensorV2()
                    ])
                    transformed = transform_only(image=img_np)
                    img_tensor = transformed['image']
                    new_boxes = []
                    new_labels = []
            else:
                # transform_only = A.Compose([
                #     A.Resize(800, 800),
                #     ToTensorV2()
                # ])
                transform_only = A.Compose([
                    A.LongestMaxSize(max_size=800),
                    A.PadIfNeeded(800, 800),
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

    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')  # Pretrained on COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Freeze backbone initially (train only RPN + ROI heads)
    for param in model.backbone.parameters():
        param.requires_grad = False
        

    return model

# --- TRAINING LOOP ---
def main():
    log = Logger(output_dir=OUTPUT_DIR, name="FasterRCNN_Train")

    log.info("==========================================")
    log.info(f"   STARTING TRAINING PIPELINE: {PROJECT_NAME}")
    log.info(f"   Device: {DEVICE}")
    log.info(f"   Batch Size: {BATCH_SIZE}")
    log.info(f"   Learning Rate: {LR} (RPN/ROI), {LR_BACKBONE} (backbone after unfreeze)")
    log.info(f"   Epochs: {NUM_EPOCHS}")
    log.info(f"   Warmup: {WARMUP_STEPS} steps")
    log.info(f"   Unfreeze backbone at epoch: {UNFREEZE_EPOCH}")
    log.info("==========================================")
    
    wandb.init(
        project=PROJECT_NAME,
        name="faster-rcnn-finetune-v3",
        config={
            "learning_rate": LR,
            "learning_rate_backbone": LR_BACKBONE,
            "warmup_steps": WARMUP_STEPS,
            "unfreeze_epoch": UNFREEZE_EPOCH,
            "architecture": "Faster R-CNN ResNet50 FPN V2",
            "dataset": "COCO-2017",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_clip": GRAD_CLIP
        }
    )
    
    # Transform
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=800),
        A.PadIfNeeded(800, 800),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    log.info("  Loading Dataset...")
    # Dataset
    dataset = COCODataset(
        root='/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/images/train2017',
        annFile='/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/annotations/org_instances_train2017.json',
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
    model = get_model(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer - separate LR for backbone vs heads
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    
    param_groups = [
        {'params': backbone_params, 'lr': LR_BACKBONE if len(backbone_params) > 0 else LR},
        {'params': head_params, 'lr': LR}
    ]
    
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0.0005)
    
    # LR Scheduler - StepLR every 8 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    start_epoch = 0
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        log.info(f"  Resuming training from: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            log.info(f"  Loaded checkpoint. Resuming from Epoch {start_epoch + 1}")
        else:
            model.load_state_dict(checkpoint)
            log.warning("  Old checkpoint format detected (weights only). Optimizer state reset.")
    else:
        log.info("  Starting fine-tuning with pretrained weights...")

    log.info("  Training Loop Started...")
    wandb.watch(model, log="all")
    
    global_step = start_epoch * len(data_loader)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = datetime.now()
        
        # Unfreeze backbone at specified epoch
        if epoch == UNFREEZE_EPOCH:
            log.info("  Unfreezing backbone and increasing LR...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer with higher LR for backbone
            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
            head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            param_groups = [
                {'params': backbone_params, 'lr': LR_BACKBONE},
                {'params': head_params, 'lr': LR}
            ]
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0.0005)
        
        for i, (images, targets) in enumerate(data_loader):
            # Warmup: gradually increase LR from base_lr * 0.1 to base_lr
            if global_step < WARMUP_STEPS:
                warmup_factor = 0.1 + 0.9 * (global_step / WARMUP_STEPS)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * warmup_factor
            
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            
            # Restore LR after warmup adjustment
            if global_step < WARMUP_STEPS:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / max(warmup_factor, 0.001)

            epoch_loss += losses.item()

            # Log all loss components
            wandb.log({ 
                "batch_loss": losses.item(), 
                "loss_classifier": loss_dict['loss_classifier'].item(),
                "loss_box_reg": loss_dict['loss_box_reg'].item(),
                "loss_objectness": loss_dict['loss_objectness'].item(),
                "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'].item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "global_step": global_step
            })
            
            global_step += 1

            if i % 50 == 0:
                log.info(f"   Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{i}/{len(data_loader)}] | Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(data_loader)
        duration = datetime.now() - start_time
        
        log.info(f"  EPOCH {epoch+1} FINISHED")
        log.info(f"   Duration: {duration}")
        log.info(f"   Avg Loss: {avg_loss:.4f}")
        log.info(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Step the scheduler
        scheduler.step()
        
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch+1, "lr": optimizer.param_groups[0]['lr']})
        
        # Save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"fasterrcnn_epoch_{epoch}.pth")

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