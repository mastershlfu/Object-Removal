import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

PROJECT_ROOT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal"

YOLOV8x_PATH = os.path.join(PROJECT_ROOT, "models/yolov8x.pt")
YOLOV8m_PATH = os.path.join(PROJECT_ROOT, "src/pipeline/models/yolov8/finetuned_yolo_v8/train_12_classes_v2/weights/epoch45.pt")
SAM_PATH = os.path.join(PROJECT_ROOT, "models/sam_vit_h_4b8939.pth")

IMG_PATH = os.path.join(PROJECT_ROOT, "data/bracelet3.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/sam_demo")

CONF_THRESHOLD = 0.5
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_OBJECT = "Bracelet"


def create_color_palette(num_objects):
    """Tạo palette màu phân biệt cho từng object."""
    np.random.seed(42)
    colors = []
    for i in range(num_objects):
        colors.append(tuple(np.random.randint(50, 255, 3).tolist()))
    return colors


def load_models():
    """Load YOLO and SAM models."""
    print("Loading YOLOv8x (general)...")
    yolo_general = YOLO(YOLOV8x_PATH)
    yolo_general.to(DEVICE)
    
    print("Loading YOLOv8m (finetuned)...")
    yolo_finetuned = YOLO(YOLOV8m_PATH)
    yolo_finetuned.to(DEVICE)
    
    print("Loading SAM...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_PATH)
    sam.to(DEVICE)
    sam_predictor = SamPredictor(sam)
    
    return yolo_general, yolo_finetuned, sam_predictor


def detect_all_objects(img_rgb, yolo_general, yolo_finetuned, conf_threshold=0.5):
    """Chạy YOLO detection với cả 2 models - trả về TẤT CẢ objects."""
    detected_objects = []
    
    def process_yolo_results(results, model, target_list, model_name):
        for box in results.boxes:
            score = box.conf[0].item()
            if score < conf_threshold:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            abs_box = [int(x1), int(y1), int(x2), int(y2)]
            
            label_id = int(box.cls[0].item())
            label_name = model.names[label_id]
            
            target_list.append({
                "box": abs_box,
                "label": label_name,
                "score": round(score, 3),
                "model": model_name
            })
    
    print("Running YOLOv8x (general - COCO 80 classes)...")
    results_general = yolo_general.predict(img_rgb, verbose=False)[0]
    process_yolo_results(results_general, yolo_general, detected_objects, "YOLOv8x")
    
    print("Running YOLOv8m (finetuned - 12 classes)...")
    results_custom = yolo_finetuned.predict(img_rgb, verbose=False)[0]
    process_yolo_results(results_custom, yolo_finetuned, detected_objects, "YOLOv8m")
    
    return detected_objects


def filter_target_object(objects, target_label):
    """Lọc object cụ thể (contains matching)."""
    target_lower = target_label.lower()
    filtered = [obj for obj in objects if target_lower in obj["label"].lower()]
    return filtered


# def generate_sam_mask(img_rgb, box, sam_predictor):
#     """Generate SAM mask từ một bounding box."""
#     sam_predictor.set_image(img_rgb)
    
#     box_np = np.array(box)
#     input_box = box_np[None, :]
    
#     mask, scores, _ = sam_predictor.predict(
#         box=input_box,
#         multimask_output=False
#     )
    
#     kernel = np.ones((30, 30), np.uint8)
#     dilated_mask = cv2.dilate(mask[0].astype(np.uint8) * 255, kernel, iterations=1)
    
#     return dilated_mask, scores[0] if scores is not None else 0.0

def generate_sam_mask(img_rgb, box, sam_predictor):
    sam_predictor.set_image(img_rgb)
    input_box = np.array(box)[None, :]
    
    masks, scores, _ = sam_predictor.predict(
        box=input_box,
        multimask_output=False
    )
    
    # Lấy mask đầu tiên và chuyển về uint8 (0 và 1)
    mask = masks[0].astype(np.uint8)
    
    # Kernel nhỏ thôi (ví dụ 5x5 hoặc 7x7) để mask mịn hơn
    kernel = np.ones((3, 3), np.uint8)
    # Nhân với 255 ở đây để mask có giá trị 0 và 255
    dilated_mask = cv2.dilate(mask * 255, kernel, iterations=1)
    
    return dilated_mask, scores[0]


def draw_all_boxes_with_labels(img, objects, colors):
    """Vẽ TẤT CẢ bounding boxes với màu sắc và label + confidence."""
    img_vis = img.copy()
    
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["box"]
        color = colors[i % len(colors)]
        label = obj["label"]
        score = obj["score"]
        model = obj["model"]
        
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)
        
        text = f"{label} {score:.2f} ({model})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(img_vis, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(img_vis, text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img_vis


def create_overlay(img_rgb, mask):
    """Tạo ảnh overlay: ảnh gốc + mask transparent màu đỏ."""
    overlay = img_rgb.copy()
    
    red_mask = np.zeros_like(img_rgb)
    red_mask[mask > 0] = [255, 0, 0]
    
    alpha = (mask > 0).astype(float) * 0.5
    
    result = img_rgb.copy()
    for c in range(3):
        result[:, :, c] = (overlay[:, :, c] * (1 - alpha) + red_mask[:, :, c] * alpha).astype(np.uint8)
    
    return result


# def visualize_results(img_rgb, all_objects, target_mask, output_dir):
#     """Tạo 4 ảnh visualization."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     colors = create_color_palette(len(all_objects) + 10)
    
#     print("\n=== Generating visualizations ===")
    
#     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
#     cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), img_bgr)
#     print(f"1. Saved: 1_original.jpg")
    
#     img_all_boxes = draw_all_boxes_with_labels(img_rgb, all_objects, colors)
#     cv2.imwrite(os.path.join(output_dir, "2_yolo_all_boxes.jpg"), cv2.cvtColor(img_all_boxes, cv2.COLOR_RGB2BGR))
#     print(f"2. Saved: 2_yolo_all_boxes.jpg ({len(all_objects)} objects)")
    
#     mask_vis = target_mask * 255
#     cv2.imwrite(os.path.join(output_dir, "3_sam_mask_bracelet.jpg"), mask_vis)
#     print(f"3. Saved: 3_sam_mask_bracelet.jpg")
    
#     overlay_img = create_overlay(img_rgb, target_mask)
#     cv2.imwrite(os.path.join(output_dir, "4_overlay.jpg"), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
#     print(f"4. Saved: 4_overlay.jpg")
    
#     print(f"\nOutput saved to: {output_dir}")

def visualize_results(img_rgb, all_objects, target_mask, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    colors = create_color_palette(len(all_objects) + 10)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Original
    cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), img_bgr)

    # 2. YOLO Boxes
    img_all_boxes = draw_all_boxes_with_labels(img_rgb, all_objects, colors)
    cv2.imwrite(os.path.join(output_dir, "2_yolo_all_boxes.jpg"), cv2.cvtColor(img_all_boxes, cv2.COLOR_RGB2BGR))

    # 3. SAM Mask (SỬA Ở ĐÂY: target_mask đã là 0-255 rồi, không nhân thêm)
    cv2.imwrite(os.path.join(output_dir, "3_sam_mask_bracelet.jpg"), target_mask)

    # 4. Overlay
    overlay_img = create_overlay(img_rgb, target_mask)
    cv2.imwrite(os.path.join(output_dir, "4_overlay.jpg"), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))


def main():
    print("=" * 50)
    print("SAM Mask Generation Demo")
    print(f"Target: {TARGET_OBJECT}")
    print("=" * 50)
    
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    
    print(f"\nImage: {IMG_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    
    yolo_general, yolo_finetuned, sam_predictor = load_models()
    
    print(f"\nLoading image...")
    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"Image size: {w}x{h}")
    
    all_objects = detect_all_objects(img_rgb, yolo_general, yolo_finetuned, CONF_THRESHOLD)
    
    if len(all_objects) == 0:
        raise ValueError("No objects detected!")
    
    print(f"\nTất cả objects detected ({len(all_objects)} objects):")
    for i, obj in enumerate(all_objects):
        print(f"  {i+1}. {obj['label']}: {obj['box']} (conf: {obj['score']:.3f}) - {obj['model']}")
    
    target_objects = filter_target_object(all_objects, TARGET_OBJECT)
    
    if len(target_objects) == 0:
        raise ValueError(f"{TARGET_OBJECT} not detected!")
    
    print(f"\nTarget object '{TARGET_OBJECT}' found: {len(target_objects)} object(s)")
    for i, obj in enumerate(target_objects):
        print(f"  Box: {obj['box']} (conf: {obj['score']:.3f})")
    
    target_obj = target_objects[0]
    target_box = target_obj["box"]
    
    print(f"\nGenerating SAM mask for {TARGET_OBJECT}...")
    target_mask, sam_score = generate_sam_mask(img_rgb, target_box, sam_predictor)
    print(f"SAM mask score: {sam_score:.3f}")
    
    visualize_results(img_rgb, all_objects, target_mask, OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()