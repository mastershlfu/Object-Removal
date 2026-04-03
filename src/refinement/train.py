import argparse
import sys
import os

# --- SETUP ĐƯỜNG DẪN ĐỂ IMPORT MODULE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

# Import Logger
from src.utils.logger import Logger
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Import các module của bạn
from src.refinement.network import RefinementUNet
from src.refinement.loss import InpaintingLoss
from src.refinement.data_processing import InpaintingDataset
from src.inpaint.lama import LaMaInpainter
from src.inpaint.semantic_sd import SemanticInpainter
from src.utils.gaussian_blur import gaussian_blur


# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMA_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/lama/big-lama.pt"
DATA_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2" 
LOG_DIR = "./models/refinement_logs"
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 20
SAVE_INTERVAL = 1 # Lưu checkpoint mỗi bao nhiêu epoch

def train():
    # 1. Setup Logger
    log = Logger(output_dir=LOG_DIR, name="Refinement_Train")
    
    # 4. Khởi tạo Metrics 
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    # LPIPS mặc định dùng mạng VGG, giá trị càng thấp càng giống người nhìn
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE)
    
    log.info("========================================")
    log.info(f"   STARTING REFINEMENT TRAINING")
    log.info(f"   Device: {DEVICE}")
    log.info(f"   Batch Size: {BATCH_SIZE}")
    log.info(f"   Learning Rate: {LR}")
    log.info("========================================")

    # 2. Init WandB
    wandb.init(
        project="Object-Removal-Refinement", 
        name="RefineNet_Experiment",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "model": "UNet"
        }
    )

    # 3. Models
    log.info("  Loading Models...")
    
    # Teacher (LaMa) - Frozen
    # Lưu ý: Class LaMaInpainter cần có self.model là TorchScript loaded model
    lama = LaMaInpainter(LAMA_PATH, device=DEVICE)
    
    # Student (Refinement Net) - Trainable
    refine_net = RefinementUNet(in_channels=4, out_channels=3).to(DEVICE)
    
    # 4. Data & Loss
    log.info(f"  Loading Dataset from: {DATA_DIR}")
    dataset = InpaintingDataset(DATA_DIR, img_size=256)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    log.info(f"  Dataset size: {len(dataset)} images")
    
    criterion = InpaintingLoss(device=DEVICE)
    optimizer = optim.Adam(refine_net.parameters(), lr=LR, betas=(0.5, 0.999))
    
    # 5. Training Loop
    log.info("  Training Loop Started...")
    
    for epoch in range(EPOCHS):
        refine_net.train()
        epoch_loss = 0
        
        # Tqdm bar for visual progress
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for batch_idx, (gt_img, coarse_img, mask) in enumerate(loop):
            # Mask: 1=Hole (Xóa), 0=Valid
            gt_img, coarse_img, mask = gt_img.to(DEVICE), coarse_img.to(DEVICE), mask.to(DEVICE)
            
            # mask = 1.0 - mask
            # mask = (mask > 0.5).float()
            mask_soft = gaussian_blur(mask, kernel_size=11, sigma=3.0) 
            
            # CNN Predict
            # refined_pred = refine_net(coarse_img, mask)
            refined_pred = refine_net(coarse_img, 1.0 - mask)
            
            # coarse_final = (1 - mask) * gt_img + mask * refined_pred
            # soften mask
            # mask_blur = gaussian_blur(mask, kernel_size=7, sigma=2.0)

            # final_output = coarse_img * (1 - mask_blur) + refined_pred * mask_blur
            final_output = (1.0 - mask_soft) * gt_img + mask_soft * refined_pred
            
            # --- BƯỚC 4: TÍNH LOSS ---
            loss, loss_dict = criterion(final_output, gt_img, mask_soft)
            
            # --- BƯỚC 5: BACKPROP ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update Tqdm bar
            loop.set_postfix(loss=loss.item())
            
            # Log chi tiết mỗi 100 batch
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    # Đảm bảo các ảnh ở dải [0, 1] trước khi tính metric
                    # Tính trên final_output (ảnh đã được dán đè/blend)
                    
                    p_score = psnr_metric(final_output, gt_img)
                    s_score = ssim_metric(final_output, gt_img)
                    
                    # LPIPS yêu cầu input chuẩn hóa về [-1, 1] hoặc dải đặc thù tùy bản,
                    # nhưng torchmetrics thường tự xử lý nếu bạn đưa vào [0, 1].
                    l_score = lpips_metric(final_output, gt_img)

                    # LOG CẢ LAMA VÀ REFINED ĐỂ SO SÁNH TRÊN WANDB
                    p_score_lama = psnr_metric(coarse_img, gt_img)
                    l_score_lama = lpips_metric(coarse_img, gt_img)
                
                wandb.log({
                    "batch_loss": loss.item(),
                    "l1_loss": loss_dict["l1"],
                    "perc_loss": loss_dict["perc"],
                    "style_loss": loss_dict["style"],
                    "grad_loss": loss_dict["grad"],
                    "metrics/PSNR": p_score.item(),
                    "metrics/SSIM": s_score.item(),
                    "metrics/LPIPS": l_score.item(),
                    "compare/PSNR_improvement": p_score.item() - p_score_lama.item(),
                    "compare/LPIPS_improvement": l_score_lama.item() - l_score.item() # LPIPS càng thấp càng tốt
                })

            # Log hình ảnh mỗi 500 batch để kiểm tra tiến độ
            if batch_idx % 500 == 0:
                # Tạo ảnh so sánh: Input (Hole) | Coarse (LaMa) | Refined (Ours) | GT
                # Tạo input hole để dễ nhìn (bôi đen vùng mask)
                input_hole = gt_img * (1 - mask)
                
                vis = torch.cat([input_hole[0], coarse_img[0], final_output[0], gt_img[0]], dim=2)
                # Chuyển về CPU numpy để log ảnh
                vis_np = vis.permute(1, 2, 0).detach().cpu().numpy()
                vis_np = np.clip(vis_np, 0, 1)
                
                wandb.log({"visual_results": [wandb.Image(vis_np, caption="Input | LaMa | Refined | GT")]})

        # End of Epoch
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"✅ Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})
        
        # Save Checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(LOG_DIR, f"refine_epoch_{epoch+1}.pth")
            torch.save(refine_net.state_dict(), ckpt_path)
            log.info(f"  Saved checkpoint: {ckpt_path}")

    log.info("  Training Finished!")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Số lượng epoch')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--data_dir', type=str, default="./data/coarse_images")
    args = parser.parse_args()

    # Cập nhật các biến toàn cục từ tham số dòng lệnh
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    # LR = args.lr
    # DATA_DIR = args.data_dir

    train()