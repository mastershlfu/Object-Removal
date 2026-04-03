import os
import numpy as np
from tqdm import tqdm
from collections import Counter

# --- CẤU HÌNH ---
BASE_LIST = "train_200_base.txt"
OUTPUT_RFS = "train_rfs.txt"
T_THRESHOLD = 0.01  # Ngưỡng chuẩn trong paper LVIS

def get_label_p(img_p):
    return img_p.replace("images/train2017", "labels/train2017").replace(".jpg", ".txt").replace(".png", ".txt")

# 1. Đọc danh sách gốc
with open(BASE_LIST, 'r') as f:
    paths = [l.strip() for l in f if l.strip()]
num_total_images = len(paths)

# 2. Bước 1: Tính f_c (Tần suất lớp trên tập ảnh)
print("📊 Bước 1: Đang tính f_c theo chuẩn LVIS...")
img_ids_per_cat = {i: set() for i in range(200)}
for p in tqdm(paths):
    lab = get_label_p(p)
    if os.path.exists(lab):
        with open(lab, 'r') as f:
            classes = set([int(line.split()[0]) for line in f if line.strip() and int(line.split()[0]) < 200])
            for c in classes:
                img_ids_per_cat[c].add(p)

# 3. Bước 2: Tính r_c cho từng category
# Công thức: r_c = max(1, sqrt(t / f_c))
r_c = {}
for c in range(200):
    f_c = len(img_ids_per_cat[c]) / num_total_images
    r_c[c] = max(1.0, np.sqrt(T_THRESHOLD / f_c)) if f_c > 0 else 1.0

# 4. Bước 3: Thực hiện Sampling cho từng ảnh
# Công thức: r_i = max({r_c | c in ảnh i})
final_paths = []
print("\n🔄 Bước 2: Thực hiện nhân bản ảnh (r_i = max r_c)...")
for p in tqdm(paths):
    lab = get_label_p(p)
    if not os.path.exists(lab): continue
    
    with open(lab, 'r') as f:
        ids = set([int(line.split()[0]) for line in f if line.strip() and int(line.split()[0]) < 200])
    
    if not ids:
        # Nếu ảnh không chứa lớp nào trong 200 lớp (hiếm gặp sau lọc) thì giữ nguyên 1
        r_i = 1.0
    else:
        r_i = max([r_c[c] for c in ids])
    
    # Số lần lặp lại (làm tròn lên theo chuẩn thực thi của LVIS)
    repeat_count = int(np.ceil(r_i))
    
    for _ in range(repeat_count):
        final_paths.append(p)

# 5. Ghi file và Thống kê kết quả
with open(OUTPUT_RFS, "w") as f:
    f.write("\n".join(final_paths))

# Thống kê nhanh để bà làm báo cáo
final_stats = Counter()
for p in final_paths:
    lab = get_label_p(p)
    if os.path.exists(lab):
        with open(lab, 'r') as f:
            for line in f:
                cid = int(line.split()[0])
                if cid < 200: final_stats[cid] += 1

counts = list(final_stats.values())
print("\n" + "="*40)
print("📝 BÁO CÁO KẾT QUẢ RFS (CHUẨN LVIS)")
print("="*40)
print(f"🔹 Tổng số ảnh sau RFS: {len(final_paths):,}")
print(f"🔹 Tổng số Instance:     {sum(counts):,}")
print(f"🔹 Max: {max(counts):,} | Min: {min(counts):,}")
print(f"🔹 Mean: {np.mean(counts):.2f} | Median: {np.median(counts):.2f}")
print(f"🔹 Độ lệch (Max/Min): {max(counts)/min(counts):.2f} lần")
print("="*40)