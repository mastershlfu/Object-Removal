import os
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# --- CẤU HÌNH ---
RFS_FILE = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/Detect_object/train_rfs.txt"
OUTPUT_CHART = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/Detect_object/rfs_distribution_chart.png"

def get_label_path(img_path):
    # Trỏ đúng vào thư mục train2017_top200 đã sửa lỗi
    return img_path.replace("images/train2017", "labels/train2017_top200").replace(".jpg", ".txt").replace(".png", ".txt")

if not os.path.exists(RFS_FILE):
    print(f"❌ Không tìm thấy file: {RFS_FILE}")
    exit()

stats = Counter()

# 1. Đếm Instance
print(f"📊 Đang quét file RFS để đếm dữ liệu...")
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

# 2. Chuẩn bị dữ liệu để vẽ
# Sắp xếp danh sách đếm theo thứ tự giảm dần để hiển thị dạng Long-tail
counts = list(stats.values())
counts.sort(reverse=True)

if not counts:
    print("❌ Không có dữ liệu để vẽ, kiểm tra lại đường dẫn!")
    exit()

# 3. Tiến hành vẽ biểu đồ
plt.figure(figsize=(14, 6))

# Vẽ cột
plt.bar(range(len(counts)), counts, color='#4C72B0', edgecolor='black', linewidth=0.5, alpha=0.8)

# Thêm các đường chỉ thị ngang
plt.axhline(y=counts[0], color='red', linestyle='--', label=f'Max: {counts[0]:,}')
plt.axhline(y=counts[-1], color='green', linestyle='--', label=f'Min: {counts[-1]:,}')
plt.axhline(y=2000, color='orange', linestyle='-', linewidth=2, label='Ngưỡng lý tưởng (>2,000)')

# Trang trí biểu đồ
plt.title('Phân phối số lượng Instance của 200 lớp sau khi dùng RFS', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Các lớp (Được sắp xếp theo tần suất giảm dần)', fontsize=12)
plt.ylabel('Số lượng Instance', fontsize=12)

plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Tối ưu layout và lưu file
plt.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=300)

print(f"\n✅ Đã vẽ xong biểu đồ! Hãy mở file để xem: {OUTPUT_CHART}")