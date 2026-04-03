import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Thêm thư viện SAM
from segment_anything import sam_model_registry, SamPredictor

# Import mạng Edge Gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.edge_connect.src.networks import EdgeGenerator

# --- CẤU HÌNH PATH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
EDGE_MODEL_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/edge_finetuned_epoch_10.pth"
MODEL_TYPE = "vit_h"

def test_real_mask_inference(img_path, target_box):
    print("🚀 Đang khởi động AI SAM và EdgeGenerator...")
    
    # 1. Load SAM
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # 2. Load Edge Model (Cục 41MB xịn nhất)
    edge_model = EdgeGenerator(use_spectral_norm=True).to(DEVICE)
    checkpoint = torch.load(EDGE_MODEL_PATH, map_location=DEVICE)
    edge_model.load_state_dict(checkpoint['generator'] if 'generator' in checkpoint else checkpoint)
    edge_model.eval()

    # 3. Đọc ảnh và Bắn Box cho SAM
    print("📸 Đang trích xuất Mask bằng SAM...")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    sam_predictor.set_image(img_rgb)
    masks, _, _ = sam_predictor.predict(box=target_box, multimask_output=False)
    
    # Mask của SAM là boolean (True/False), ép về uint8 (0 hoặc 255)
    sam_mask = (masks[0] * 255).astype(np.uint8)

    # Dãn Mask (Dilate) y hệt như trong pipeline gốc của bạn để xóa tận gốc viền vật thể
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(sam_mask, kernel, iterations=1)

    # 4. Resize về 512x512 cho mượt (EdgeConnect được train chuẩn ở size này)
    print("🎨 Đang vẽ khung xương kiến trúc...")
    img_resized = cv2.resize(img_rgb, (512, 512))
    mask_resized = cv2.resize(dilated_mask, (512, 512))
    mask_normalized = mask_resized / 255.0 # Đưa về 0.0 - 1.0 cho EdgeGenerator dễ hiểu

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edge_gt = cv2.Canny(gray, 100, 200) / 255.0

    # 5. Đục lỗ trên nền xám và Canny
    masked_gray = (gray / 255.0) * (1 - mask_normalized)
    masked_edge = edge_gt * (1 - mask_normalized)

    # Gộp 3 kênh làm đầu vào cho Mạng Edge
    inputs = np.stack([masked_gray, masked_edge, mask_normalized], axis=0)
    inputs_tensor = torch.from_numpy(inputs).float().unsqueeze(0).to(DEVICE)

    # 6. AI Edge Trổ tài
    with torch.no_grad():
        pred_edge = edge_model(inputs_tensor)

    # Xử lý Thresholding ép nét về dạng Binary đen trắng
    pred_edge_np = pred_edge[0, 0].cpu().numpy()
    pred_edge_binary = (pred_edge_np > 0.5).astype(np.float32)

    # Phép thuật Composite: Nét thật ngoài nền + Nét AI vẽ trong lỗ
    final_edge = (edge_gt * (1 - mask_normalized)) + (pred_edge_binary * mask_normalized)

    # 7. Vẽ và Lưu hình ảnh kiểm chứng (5 Tấm)
    print("💾 Đang lưu ảnh kết quả báo cáo...")
    plt.figure(figsize=(25, 5)) # Kéo dài ra để chứa 5 ảnh
    
    # Ảnh 1: Ảnh gốc vẽ Box đỏ (Để xem mình đang chọn xóa cái gì)
    plt.subplot(1, 5, 1); plt.title("Ảnh gốc + Box Xóa"); 
    img_box = img_rgb.copy()
    cv2.rectangle(img_box, (target_box[0], target_box[1]), (target_box[2], target_box[3]), (255, 0, 0), 3)
    plt.imshow(img_box)

    # Ảnh 2: Mask Dilated của SAM
    plt.subplot(1, 5, 2); plt.title("SAM Mask (Đã Dilate)"); plt.imshow(mask_normalized, cmap='gray')
    
    # Ảnh 3: Canny bị đục lỗ
    plt.subplot(1, 5, 3); plt.title("Canny bị đục lỗ"); plt.imshow(masked_edge, cmap='gray')
    
    # Ảnh 4: Ruột AI vẽ
    plt.subplot(1, 5, 4); plt.title("Nét AI chắp vá"); plt.imshow(pred_edge_binary * mask_normalized, cmap='gray')
    
    # Ảnh 5: Khung xương Final
    plt.subplot(1, 5, 5); plt.title("Khung xương Hoàn chỉnh"); plt.imshow(final_edge, cmap='gray')

    plt.tight_layout()
    plt.savefig("ket_qua_sam_edge.png", dpi=150)
    print("✅ Đã xuất bản thành công tại: ket_qua_sam_edge.png")

if __name__ == "__main__":
    # SỬA 2 BIẾN NÀY LẠI CHO ĐÚNG:
    # 1. Đường dẫn 1 tấm ảnh thật có người/vật mà bạn muốn test
    test_img_path = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/LaMa_test_images/images/whitedog.jpg" 
    
    # 2. Tọa độ Box của vật thể đó (Tôi lấy tạm INPUT_BOX trong file gốc của bạn)
    # Cú pháp: [xmin, ymin, xmax, ymax]
    box = np.array([0, 100, 581, 1000]) 
    
    if not os.path.exists(test_img_path):
        print(f"❌ LỖI: Không tìm thấy ảnh tại {test_img_path}. Vui lòng sửa lại đường dẫn!")
    else:
        test_real_mask_inference(test_img_path, box)