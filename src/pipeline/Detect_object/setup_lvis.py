import yaml

# --- CẤU HÌNH ĐƯỜNG DẪN ---
names_file = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/top_200_names.txt"
yaml_output = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco/lvis_top200.yaml"

# 1. Đọc danh sách 200 tên lớp
with open(names_file, 'r') as f:
    # Strip để bỏ dấu xuống dòng, tránh lỗi định dạng yaml
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# 2. Tạo cấu trúc dictionary cho file YAML
# YOLOv11 chấp nhận names dạng list (nó tự gán index từ 0) 
# hoặc dict {index: name}. Dùng list cho gọn.
data_config = {
    'path': '/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/coco',
    'train': 'images/train2017',
    'val': 'images/val2017',
    'nc': len(class_names),
    'names': class_names
}

# 3. Ghi ra file .yaml
with open(yaml_output, 'w') as f:
    # allow_unicode=True để nếu có tên lớp lạ nó không bị lỗi font
    # sort_keys=False để giữ nguyên thứ tự nc, train, val... cho dễ đọc
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"✅ Đã tạo xong file YAML tại: {yaml_output}")
print(f"📊 Kiểm tra: Đã add {len(class_names)} lớp vào file.")