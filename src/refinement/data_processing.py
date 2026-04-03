import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os

class InpaintingDataset(Dataset):
    def __init__(self, coarse_root, img_size=256):
        self.gt_paths = sorted(glob.glob(os.path.join(coarse_root, "val_large", "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(coarse_root, "masks", "*.jpg")))
        self.coarse_paths = sorted(glob.glob(os.path.join(coarse_root, "coarse_images", "*.jpg")))
        self.img_size = img_size
        
        assert len(self.gt_paths) == len(self.coarse_paths), \
            f"Số lượng GT ({len(self.gt_paths)}) và Coarse ({len(self.coarse_paths)}) không khớp!"
        # print("gt path: ", os.path.join(coarse_root, "gt", "*.png"))
        # print("mask path: ", os.path.join(coarse_root, "gt_masked", "*.png"))
        # print("coarse path: ", self.coarse_paths)

    def random_mask(self, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Random Rectangles
        num_rects = np.random.randint(1, 5)
        for _ in range(num_rects):
            x1 = np.random.randint(0, width - 50)
            y1 = np.random.randint(0, height - 50)
            w = np.random.randint(20, 100)
            h = np.random.randint(20, 100)
            cv2.rectangle(mask, (x1, y1), (x1+w, y1+h), 255, -1)
            
        # Random Brush Strokes (Free-form)
        num_strokes = np.random.randint(1, 5)
        for _ in range(num_strokes):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            thickness = np.random.randint(10, 40)
            # Vẽ đường ngẫu nhiên
            for _ in range(10):
                x2 = x1 + np.random.randint(-50, 50)
                y2 = y1 + np.random.randint(-50, 50)
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
                x1, y1 = x2, y2
                
        return mask

    def __getitem__(self, idx):
        # 1. Load GT
        gt = cv2.imread(self.gt_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        coarse = cv2.imread(self.coarse_paths[idx])
        
        gt = cv2.resize(gt, (self.img_size, self.img_size))
        coarse = cv2.resize(coarse, (self.img_size, self.img_size))
        
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        coarse = cv2.cvtColor(coarse, cv2.COLOR_BGR2RGB)
        
        # 2. Generate Mask (1 channel)
        mask = self.random_mask(self.img_size, self.img_size)
        
        # 3. Normalize & Tensor
        # Image: [3, H, W], Mask: [1, H, W]
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
        coarse_tensor = torch.from_numpy(coarse).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0) # [1, H, W]

        
        return gt_tensor, coarse_tensor, mask_tensor

    def __len__(self):
        return len(self.gt_paths)