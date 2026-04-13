import os
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
# Trỏ đến file best.pt của đợt train thần thánh vừa rồi
# Thay đổi tên thư mục train_12_classes_v2_resplit nếu bạn đặt tên khác
MODEL_PATH = "/media/ml4u/Challenge-4TB/baonhi/finetuned_yolo_v8/train_12_classes_v2/weights/epoch45.pt"

# Nguồn dữ liệu bạn muốn test (Có thể là 1 file ảnh, 1 file video, hoặc 1 thư mục chứa nhiều ảnh)
# Ví dụ 1: Test 1 ảnh
SOURCE = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/watch.png" 

# Ví dụ 2: Test cả 1 thư mục
# SOURCE = "/đường/dẫn/đến/thu_muc_anh_test/"

# Ví dụ 3: Test video
# SOURCE = "/đường/dẫn/đến/video_test.mp4"

# ==========================================
# 2. LOAD MODEL VÀ DỰ ĐOÁN
# ==========================================
if __name__ == '__main__':
    print(f"Đang load model từ: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"Đang inference trên: {SOURCE}")
    
    # Chạy inference
    results = model.predict(
        source=SOURCE,
        save=True,       # Bắt buộc = True để YOLO vẽ khung và lưu lại ảnh/video kết quả
        conf=0.5,        # Ngưỡng tự tin (Confidence Threshold). Chỉ vẽ khung nếu model chắc chắn > 40%
        iou=0.5,         # Ngưỡng IoU để chống vẽ đè nhiều khung lên cùng 1 vật thể (Non-Maximum Suppression)
        device=0,        # Dùng RTX 4090 để chạy cho lẹ
        show_labels=True,# Hiện tên class (VD: Earrings, Brooches)
        show_conf=True,  # Hiện % tự tin trên cái khung
        line_width=2     # Độ dày của viền khung (chỉnh nhỏ lại nếu vật thể như khuyên tai quá bé)
    )
    
    # In ra đường dẫn nơi lưu kết quả
    print("\n" + "="*50)
    print("HOÀN TẤT INFERENCE!")
    # YOLO thường lưu tự động vào thư mục runs/detect/predict...
    print(f"Kết quả đã được lưu tại: {results[0].save_dir}")
    print("="*50)