import os
from collections import Counter
from tqdm import tqdm
import numpy as np

# --- CẤU HÌNH ---
RFS_FILE = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/Detect_object/train_rfs.txt"

def get_label_path(img_path):
    # Logic chuẩn để trỏ từ images sang labels
    return img_path.replace("images/train2017", "labels/train2017").replace(".jpg", ".txt").replace(".png", ".txt")

if not os.path.exists(RFS_FILE):
    print(f"❌ Không thấy file: {RFS_FILE}")
    exit()

stats = Counter()

# 1. Đếm trực tiếp từ file (Stream reading)
print(f"📊 Đang quét file RFS (Chế độ tiết kiệm bộ nhớ)...")
with open(RFS_FILE, 'r') as f:
    for line in tqdm(f):
        path = line.strip()
        if not path: continue
        
        label_p = get_label_path(path)
        if os.path.exists(label_p):
            with open(label_p, 'r') as lf:
                for l_line in lf:
                    if l_line.strip():
                        cls_id = int(l_line.split()[0])
                        if cls_id < 200:
                            stats[cls_id] += 1

# 2. Xuất kết quả
counts = list(stats.values())
if not counts:
    print("❌ Đếm xong nhma không có instance nào. Check lại đường dẫn replace!")
else:
    print("\n--- THỐNG KÊ THỰC TẾ ---")
    print(f"🔹 Tổng số Instance: {sum(counts)}")
    print(f"🔹 Max: {max(counts)} | Min: {min(counts)}")
    print(f"🔹 Độ lệch: {max(counts)/min(counts):.2f} lần")