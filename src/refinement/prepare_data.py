import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SDXL_VAE_PATH = "/media/ml4u/Challenge-4TB/baonhi/sdxl-inpaint-offline/vae" 
PLACES2_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places/val_large" 

OUTPUT_HQ = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places/dataset/HQ"
OUTPUT_LQ = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places/dataset/LQ"
CROP_SIZE = 256

os.makedirs(OUTPUT_HQ, exist_ok=True)
os.makedirs(OUTPUT_LQ, exist_ok=True)

print("🚀 Đang nạp VAE của SDXL...")
# Tải VAE ở float16 để chạy cho lẹ, tiết kiệm VRAM
SDXL_VAE_FILE = "/media/ml4u/Challenge-4TB/baonhi/sdxl-inpaint-offline/vae/diffusion_pytorch_model.fp16.safetensors"
vae = AutoencoderKL.from_single_file(
    SDXL_VAE_FILE, 
    torch_dtype=torch.float16
).to(DEVICE)
vae.eval()

# Transform để đưa ảnh vào VAE (-1 đến 1)
transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def create_random_mask(size=CROP_SIZE):
    """Tạo mask trắng đen ngẫu nhiên với viền mờ để giả lập vết cắt của SAM/LaMa"""
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Vẽ 1-2 hình tròn/elip ngẫu nhiên làm vùng bị inpaint
    num_shapes = random.randint(1, 2)
    for _ in range(num_shapes):
        center = (random.randint(50, size-50), random.randint(50, size-50))
        axes = (random.randint(30, 80), random.randint(30, 80))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    # Làm nhòe viền cực mạnh (Gaussian Blur) hệt như pipeline thực tế
    blur_kernel = random.choice([21, 31, 41])
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    
    # Chuẩn hóa về [0, 1] và thêm kênh (256, 256, 1)
    mask = mask.astype(np.float32) / 255.0
    return np.expand_dims(mask, axis=-1)

def process_image(img_path, save_name):
    # 1. Đọc và Random Crop ra HQ (Ground Truth)
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    if h < CROP_SIZE or w < CROP_SIZE: return
    
    y = random.randint(0, h - CROP_SIZE)
    x = random.randint(0, w - CROP_SIZE)
    img_hq = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
    
    # 2. Ép qua VAE để làm nhão (Tạo ảnh VAE)
    img_pil = Image.fromarray(img_hq)
    img_tensor = transform_to_tensor(img_pil).unsqueeze(0).to(DEVICE, dtype=torch.float16)
    
    with torch.no_grad():
        # Encode xuống Latent
        latent = vae.encode(img_tensor).latent_dist.sample()
        # Decode ngược lại lên Pixel
        decoded = vae.decode(latent).sample
        
    # Chuyển tensor kết quả về lại Numpy [0, 255]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    img_vae = decoded[0].cpu().permute(1, 2, 0).numpy()
    img_vae = (img_vae * 255).astype(np.uint8)
    
    img_vae = cv2.GaussianBlur(img_vae, (3, 3), 0)
    
    # 3. Trộn (Alpha Blending)
    mask = create_random_mask()
    # Mask = 1 (trắng): Lấy ảnh VAE nhão. Mask = 0 (đen): Lấy ảnh HQ nét.
    img_lq = (img_vae * mask) + (img_hq * (1.0 - mask))
    img_lq = img_lq.astype(np.uint8)
    
    # 4. Lưu kết quả ra ổ cứng (Chuyển lại BGR cho OpenCV)
    cv2.imwrite(os.path.join(OUTPUT_HQ, save_name), cv2.cvtColor(img_hq, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_LQ, save_name), cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR))

def main():
    # Quét tất cả ảnh trong thư mục Places2
    image_paths = glob.glob(os.path.join(PLACES2_DIR, "**/*.jpg"), recursive=True)
    random.shuffle(image_paths) # Trộn ngẫu nhiên
    
    # Lấy 36,000 ảnh (hoặc tùy bạn set)
    limit = min(36000, len(image_paths))
    image_paths = image_paths[:limit]
    
    print(f"🎯 Bắt đầu sinh dataset cho {limit} ảnh...")
    
    for i, img_path in enumerate(tqdm(image_paths)):
        save_name = f"{i:05d}.jpg" # Lưu tên dạng 00001.jpg, 00002.jpg
        process_image(img_path, save_name)
        
    print("✅ Sinh Dataset hoàn tất! Kiểm tra thư mục ./dataset/HQ và ./dataset/LQ")

if __name__ == "__main__":
    import glob
    main()