import os
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.edge_connect.src.networks import *

DATA_TXT_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places/clean_places2.txt"

class EdgeInpaintDataset(Dataset):
    def __init__(self, txt_path, img_size = 512):
        super().__init__()
        self.image_size = img_size
        with open(txt_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        print(f"loaded {len(self.image_paths)} img from file txt")
    
    def __len__(self):
        return len(self.image_paths)
    
    def random_free_form_mask(self):
        mask = np.zeros((self.image_size, self.image_size), np.uint8)
        
        # draw random mask lines (Random Walk Algorithm)
        num_strokes = random.randint(3, 7)
        for _ in range(num_strokes):
            # random start
            x, y = random.randint(0, self.image_size), random.randint(0, self.image_size)
            
            max_length = random.randint(50, 200) # Length of brush
            thickness = random.randint(10, 50)   # Thickness
            angle = random.uniform(0, 2 * np.pi) # Angle to twist (0-360)
            
            for _ in range(max_length):
                # randomly twist(-1 to 1 rad)
                angle += random.uniform(-1, 1) 
                
                # move short distance
                step_size = random.uniform(5, 15)
                next_x = x + np.cos(angle) * step_size
                next_y = y + np.sin(angle) * step_size
                
                # draw a line to map points
                cv2.line(mask, (int(x), int(y)), (int(next_x), int(next_y)), 255, thickness)
                
                # update x and y
                x, y = next_x, next_y
                
                # if go outside, start new line
                if x < 0 or x > self.image_size or y < 0 or y > self.image_size:
                    break

        # Randomly generate big hole
        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            if random.random() > 0.5: # 50% chance
                cx, cy = random.randint(0, self.image_size), random.randint(0, self.image_size)
                # Bán kính 2 trục x, y khác nhau để tạo độ méo
                axes = (random.randint(30, 100), random.randint(20, 80)) 
                rotation_angle = random.randint(0, 360)
                
                cv2.ellipse(mask, (cx, cy), axes, rotation_angle, 0, 360, 255, -1)
                
        return mask / 255.0
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        if img is None: 
            return self.__getitem__((index + 1) % len(self))
            
        img = cv2.resize(img, (self.image_size, self.image_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ground Truth Edge (Use canny to convert img into lines, 100-200 is threshold for None Maximum Suppression)
        edge_gt = cv2.Canny(gray, 100, 200) / 255.0
        
        mask = self.random_free_form_mask()
        
        # Create holes from masks
        masked_gray = (gray / 255.0) * (1 - mask)
        masked_edge = edge_gt * (1 - mask)
        
        # Gộp 3 kênh đầu vào: [Masked Gray, Masked Edge, Mask]
        input_tensor = np.stack([masked_gray, masked_edge, mask], axis=0)
        
        return (
            torch.from_numpy(input_tensor).float(), 
            torch.from_numpy(edge_gt).float().unsqueeze(0) # Shape [1, H, W]
        )

from torch.utils.data import random_split # Thêm thư viện này ở đầu file

def train_edge_generator(pretrained_path, batch_size=2, start_ep=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Fine-tune on device {device}")

    model = EdgeGenerator(use_spectral_norm=True).to(device)
    
    if os.path.exists(pretrained_path):
        print(f"📥 Loading pretrained from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)

        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully")
    else:
        print(f"⚠️ Path {pretrained_path} not found. Training from scratch!")
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # ==========================================
    # TỰ ĐỘNG CHIA TẬP DATA (95% TRAIN - 5% EVAL)
    # ==========================================
    full_dataset = EdgeInpaintDataset(txt_path=DATA_TXT_PATH)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
    
    epochs = 20
    save_dir = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_ep, epochs):
        # TRAINING
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, target_edges in pbar:
            inputs, target_edges = inputs.to(device), target_edges.to(device)

            optimizer.zero_grad()
            predicted_edges = model(inputs)

            mask = inputs[:, 2:3, :, :]
            loss = criterion(predicted_edges * mask, target_edges * mask)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_loss = running_loss / len(train_loader)

        # GIAI ĐOẠN 2: EVALUATION 
        model.eval() 
        val_running_loss = 0.0
        
        val_viz_inputs, val_viz_targets, val_viz_preds = None, None, None

        with torch.no_grad(): 
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval]")
            for val_step, (inputs, target_edges) in enumerate(pbar_val):
                inputs, target_edges = inputs.to(device), target_edges.to(device)
                predicted_edges = model(inputs)
                
                mask = inputs[:, 2:3, :, :]
                loss = criterion(predicted_edges * mask, target_edges * mask)
                val_running_loss += loss.item()
                
                
                if val_step == 0:
                    val_viz_inputs = inputs
                    val_viz_targets = target_edges
                    val_viz_preds = predicted_edges

        val_loss = val_running_loss / len(val_loader)

        # ==========================================
        # LOGGING & SAVING
        # ==========================================
        viz_input_gray = val_viz_inputs[0, 0, :, :].cpu().numpy()
        viz_mask = val_viz_inputs[0, 2, :, :].cpu().numpy()
        viz_target = val_viz_targets[0, 0, :, :].cpu().numpy()
        viz_pred = val_viz_preds[0, 0, :, :].cpu().numpy()

        wandb.log({
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "epoch": epoch + 1,
            "Visuals_Eval/Input_Masked": wandb.Image(viz_input_gray, caption="Ảnh đầu vào"),
            "Visuals_Eval/Mask": wandb.Image(viz_mask, caption="Mash"),
            "Visuals_Eval/AI_Prediction": wandb.Image(viz_pred, caption="Nét vẽ"),
            "Visuals_Eval/Ground_Truth": wandb.Image(viz_target, caption="Đáp án")
        })

        print(f"✅ Epoch {epoch+1} done! Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"edge_finetuned_epoch_{epoch+1}.pth"))

