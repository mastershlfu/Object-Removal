import os
import cv2
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ
# ==========================================
SOURCE_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/fine_tune_yolov8"
OUTPUT_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/fine_tune_yolov8_v2"

# ID của các class cần Flip
TARGET_CLASSES = [0, 4, 8, 10] 
VAL_RATIO = 0.2

def gather_and_flip_data():
    all_data_pairs = []
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(SOURCE_DIR, split, 'images')
        lbl_dir = os.path.join(SOURCE_DIR, split, 'labels')
        
        if not os.path.exists(img_dir): continue
            
        for img_name in os.listdir(img_dir):
            if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue
            
            if "_flipped" in img_name: continue
            
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if os.path.exists(lbl_path):
                all_data_pairs.append((img_path, lbl_path))
                
                # --- XỬ LÝ FLIP ---
                has_target = False
                new_labels = []
                
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 0: continue
                        
                    class_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    
                    if class_id in TARGET_CLASSES:
                        has_target = True
                        
                    # Lật tọa độ (Xử lý mượt cả Box 4 tọa độ lẫn Polygon n tọa độ)
                    if len(coords) == 4:
                        # Box format: x_center, y_center, w, h -> Chỉ lật x_center
                        new_coords = [1.0 - coords[0], coords[1], coords[2], coords[3]]
                    else:
                        # Polygon format: x1, y1, x2, y2... -> Lật tất cả x (index chẵn)
                        new_coords = [1.0 - v if i % 2 == 0 else v for i, v in enumerate(coords)]
                    
                    # Format lại thành chuỗi
                    str_coords = " ".join([f"{v:.6f}" for v in new_coords])
                    new_labels.append(f"{class_id} {str_coords}\n")
                
                # Nếu có chứa class mục tiêu thì sinh ảnh flip
                if has_target:
                    img = cv2.imread(img_path)
                    img_flipped = cv2.flip(img, 1)
                    
                    flipped_img_name = f"{Path(img_path).stem}_flipped{Path(img_path).suffix}"
                    flipped_lbl_name = f"{Path(lbl_path).stem}_flipped.txt"
                    
                    flipped_img_path = os.path.join(img_dir, flipped_img_name)
                    flipped_lbl_path = os.path.join(lbl_dir, flipped_lbl_name)
                    
                    cv2.imwrite(flipped_img_path, img_flipped)
                    with open(flipped_lbl_path, 'w') as f:
                        f.writelines(new_labels)
                        
                    all_data_pairs.append((flipped_img_path, flipped_lbl_path))

    return all_data_pairs

def stratified_split(data_pairs):
    dataset_info = []
    global_class_counts = Counter()
    
    # Bước 1: Thu thập thông tin class có trong từng ảnh
    for img_path, lbl_path in data_pairs:
        classes_in_img = set()
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    c_id = int(parts[0])
                    classes_in_img.add(c_id)
                    global_class_counts[c_id] += 1
        
        dataset_info.append({
            'paths': (img_path, lbl_path),
            'classes': classes_in_img
        })

    # Bước 2: Tìm "Class hiếm nhất" cho mỗi bức ảnh
    groups = defaultdict(list)
    for item in dataset_info:
        if not item['classes']:
            groups[-1].append(item['paths']) # Ảnh không có vật thể (Background)
        else:
            # Tìm class xuất hiện ít nhất trên tổng toàn dataset
            rarest_class = min(item['classes'], key=lambda c: global_class_counts[c])
            groups[rarest_class].append(item['paths'])

    # Bước 3: Chia 80/20 bên trong từng nhóm class hiếm
    train_pairs, val_pairs = [], []
    for c_id, pairs in groups.items():
        random.shuffle(pairs) # Trộn ngẫu nhiên trong group
        split_idx = int(len(pairs) * (1 - VAL_RATIO))
        train_pairs.extend(pairs[:split_idx])
        val_pairs.extend(pairs[split_idx:])
        
    # Trộn lại một lần cuối cho đều
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    
    return train_pairs, val_pairs


if __name__ == "__main__":
    print("1. Đang gom dữ liệu và tiến hành lật ảnh (Flip)...")
    all_pairs = gather_and_flip_data()
    print(f"-> Tổng cộng có {len(all_pairs)} cặp ảnh/nhãn.")
    
    print("\n2. Đang phân tích và chia đều dữ liệu (Stratified 80/20)...")
    train_pairs, val_pairs = stratified_split(all_pairs)
    
    # Tạo folder
    for split in ['train', 'val']: 
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)
        
    def copy_to_new_dir(pairs, split_name):
        for img_path, lbl_path in pairs:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path)
            
            dst_img = os.path.join(OUTPUT_DIR, split_name, 'images', img_name)
            dst_lbl = os.path.join(OUTPUT_DIR, split_name, 'labels', lbl_name)
            
            shutil.copy(img_path, dst_img)
            shutil.copy(lbl_path, dst_lbl)
            
            if "_flipped" in img_name:
                os.remove(img_path)
                os.remove(lbl_path)

    print(f"-> Đang copy {len(train_pairs)} files vào Train...")
    copy_to_new_dir(train_pairs, 'train')
    
    print(f"-> Đang copy {len(val_pairs)} files vào Val...")
    copy_to_new_dir(val_pairs, 'val')
    
    print("\nHOÀN TẤT! Dữ liệu đã được chia đều đẹp đẽ.")