# import sys
# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# sys.path.append(project_root)

# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import clip

# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # SAM
# from segment_anything import sam_model_registry, SamPredictor

# # LaMa
# from src.inpaint.lama import LaMaInpainter


# class ObjectRemovalSystem:
#     def __init__(self, rcnn_path, sam_path, lama_path, device="cuda"):
#         self.device = device
#         print("⏳ Initializing Object Removal Pipeline...")

#         # 1. Faster R-CNN
#         self.detector = self._load_rcnn_model(rcnn_path)

#         # 2. SAM
#         self.sam = sam_model_registry["vit_h"](checkpoint=sam_path)
#         self.sam.to(device)
#         self.sam.eval()
#         self.sam_predictor = SamPredictor(self.sam)

#         # 3. CLIP
#         self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

#         # 4. LaMa
#         self.lama = LaMaInpainter(lama_path, device=device)

#         print("✅ System Ready!")

#     # ------------------------
#     # LOAD MODELS
#     # ------------------------
#     def _load_rcnn_model(self, path):
#         print(f"📦 Loading Faster R-CNN from {path}...")
#         model = fasterrcnn_resnet50_fpn_v2(weights=None)
#         in_features = model.roi_heads.box_predictor.cls_score.in_features
#         model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 91)

#         checkpoint = torch.load(path, map_location=self.device)
#         if "model_state_dict" in checkpoint:
#             model.load_state_dict(checkpoint["model_state_dict"])
#         else:
#             model.load_state_dict(checkpoint)

#         model.to(self.device)
#         model.eval()
#         return model

#     # ------------------------
#     # DETECTION
#     # ------------------------
#     def _run_detector(self, image_bgr):
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
#         img_tensor = img_tensor.to(self.device)

#         with torch.no_grad():
#             pred = self.detector([img_tensor])[0]

#         boxes = pred["boxes"].cpu().numpy()
#         scores = pred["scores"].cpu().numpy()
#         keep = scores > 0.3
#         return boxes[keep]

#     def _select_best_box_with_clip(self, image_rgb, boxes, text_prompt):
#         if len(boxes) == 0:
#             return None, -1

#         text_token = clip.tokenize([text_prompt]).to(self.device)
#         best_score = -1
#         best_box = None

#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             h, w, _ = image_rgb.shape
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)

#             if x2 - x1 < 10 or y2 - y1 < 10:
#                 continue

#             crop = image_rgb[y1:y2, x1:x2]
#             crop_pil = Image.fromarray(crop)
#             img_input = self.clip_preprocess(crop_pil).unsqueeze(0).to(self.device)

#             with torch.no_grad():
#                 img_feat = self.clip_model.encode_image(img_input)
#                 text_feat = self.clip_model.encode_text(text_token)
#                 img_feat /= img_feat.norm(dim=-1, keepdim=True)
#                 text_feat /= text_feat.norm(dim=-1, keepdim=True)
#                 score = (img_feat @ text_feat.T).item()

#             if score > best_score:
#                 best_score = score
#                 best_box = np.array([x1, y1, x2, y2])

#         return best_box, best_score

#     # ------------------------
#     # MAIN PIPELINE
#     # ------------------------
#     def process(self, image_input, mask_input, text_prompt):
#         image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
#         ih, iw, _ = image_input.shape

#         # ------------------------
#         # 1. USER BOX
#         # ------------------------
#         user_box = None
#         if mask_input is not None:
#             mask_gray = mask_input[:, :, 0] if mask_input.ndim == 3 else mask_input
#             coords = cv2.findNonZero(mask_gray)
#             if coords is not None:
#                 x, y, w, h = cv2.boundingRect(coords)

#                 mh, mw = mask_gray.shape
#                 sx, sy = iw / mw, ih / mh

#                 x1 = int(x * sx)
#                 y1 = int(y * sy)
#                 x2 = int((x + w) * sx)
#                 y2 = int((y + h) * sy)

#                 user_box = np.array([x1, y1, x2, y2])

#         # ------------------------
#         # 2. ROUTER LOGIC
#         # ------------------------
#         final_box = None

#         if user_box is not None:
#             print("🚀 Mode: User Box (Trust User)")
#             final_box = user_box

#         elif text_prompt is not None and text_prompt.strip() != "":
#             print("🚀 Mode: Text Only")
#             boxes = self._run_detector(image_bgr)
#             final_box, _ = self._select_best_box_with_clip(
#                 image_input, boxes, text_prompt
#             )

#         if final_box is None:
#             return None, None, "❌ Không tìm thấy vật thể!"

#         # ------------------------
#         # 3. EXPAND BOX FOR SAM
#         # ------------------------
#         pad = int(0.15 * max(iw, ih))
#         x1, y1, x2, y2 = final_box

#         x1 = max(0, x1 - pad)
#         y1 = max(0, y1 - pad)
#         x2 = min(iw, x2 + pad)
#         y2 = min(ih, y2 + pad)

#         final_box = np.array([x1, y1, x2, y2])

#         print("🎯 SAM Box:", final_box)
#         print(
#             "📐 Box area ratio:",
#             (x2 - x1) * (y2 - y1) / (iw * ih),
#         )

#         # ------------------------
#         # 4. SAM SEGMENTATION
#         # ------------------------
#         # image_np = image_input.astype(np.uint8)
#         # self.sam_predictor.set_image(image_np)
        
#         if isinstance(image_input, torch.Tensor):
#             image_np = image_input.detach().cpu().numpy()
#         else:
#             image_np = image_input

#         if image_np.dtype != np.uint8:
#             image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

#         if image_np.ndim == 3 and image_np.shape[2] == 4:
#             image_np = image_np[:, :, :3]  # drop alpha

#         # SAM expects RGB uint8 HWC
#         self.sam_predictor.set_image(image_np)

#         masks, _, _ = self.sam_predictor.predict(
#             box=final_box,
#             multimask_output=False,
#         )

#         final_mask = masks[0].astype(np.uint8) * 255

#         kernel = np.ones((15, 15), np.uint8)
#         final_mask = cv2.dilate(final_mask, kernel, iterations=1)

#         # ------------------------
#         # 5. LAMA INPAINT
#         # ------------------------
#         print("🎨 Running LaMa Inpainting...")
        
#         lama_img = image_np

#         inpainted = self.lama.inpaint(lama_img, final_mask)

#         return final_mask, inpainted, "✅ Xóa thành công!"

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
import cv2
import numpy as np
from PIL import Image
import clip

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor #Change the arch

# SAM
from segment_anything import sam_model_registry, SamPredictor

# LaMa
from src.inpaint.lama import LaMaInpainter
# RePaint
from src.inpaint.repaint import RePaintInpainter

# Danh sách 91 class của COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ObjectRemovalSystem:
    def __init__(self, rcnn_path, sam_path, lama_path, device="cuda"):
        self.device = device
        print("  Initializing Object Removal Pipeline...")
        
        # 1. Faster R-CNN
        self.detector = self._load_rcnn_model(rcnn_path)

        # 2. SAM
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_path)
        self.sam.to(device)
        self.sam_predictor = SamPredictor(self.sam)
        
        # 3. CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # 4. LaMa
        self.lama = LaMaInpainter(lama_path, device=device)
        # self.repaint_conf = RePaintInpainter(repaint_conf_path, device=device)
        
        print("  System Ready!")

    def _load_rcnn_model(self, path):
        print(f"  Loading Faster R-CNN from {path}...")
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 91)
        
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        return model

    def _run_detector(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.detector([img_tensor])
            
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Lấy ngưỡng 0.3
        keep = scores > 0.3
        return boxes[keep], labels[keep], scores[keep]

    # [NEW] Hàm lọc trả về NHIỀU BOXES thay vì 1 box
    def _select_boxes(self, image_rgb, boxes, labels, text_prompt):
        if len(boxes) == 0: return []
        
        selected_boxes = []
        text_lower = text_prompt.lower()
        
        # 1. Ưu tiên lọc theo tên Class (Chính xác tuyệt đối)
        # Nếu user nhập "person", lấy tất cả box là person.
        class_match_indices = []
        for idx, label_id in enumerate(labels):
            try:
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
                # Kiểm tra từ khóa (vd: "person" in "remove person")
                if class_name in text_lower:
                    class_match_indices.append(idx)
            except: pass
            
        if len(class_match_indices) > 0:
            print(f"   --> Found {len(class_match_indices)} objects matching class name '{text_prompt}'")
            # Trả về tất cả các box khớp class
            return boxes[class_match_indices]

        # 2. Nếu không khớp tên class, dùng CLIP để tìm (Semantic Search)
        # Ví dụ: prompt "man in red shirt" (không phải tên class chuẩn)
        print("   --> No direct class match. Using CLIP semantic search...")
        text_token = clip.tokenize([f"a photo of a {text_prompt}"]).to(self.device)
        
        scores = []
        h, w, _ = image_rgb.shape

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 5 or y2 - y1 < 5: 
                scores.append(-1.0)
                continue

            crop_pil = Image.fromarray(image_rgb[y1:y2, x1:x2])
            img_input = self.clip_preprocess(crop_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                img_feat = self.clip_model.encode_image(img_input)
                text_feat = self.clip_model.encode_text(text_token)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
                score = (img_feat @ text_feat.T).item()
                scores.append(score)
        
        scores = np.array(scores)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Nếu CLIP tìm thấy box có điểm cao, trả về box đó
        if best_score > 0.22: # Ngưỡng CLIP
            print(f"   --> CLIP Selected best object (Score: {best_score:.3f})")
            return [boxes[best_idx]]
            
        return []

    def process(self, image_input, mask_input, text_prompt):
        # Fix input
        if isinstance(image_input, torch.Tensor):
            image_sam = image_input.cpu().numpy()
        else:
            image_sam = image_input
        image_sam = np.array(image_sam, dtype=np.uint8, copy=True)
        if image_sam.shape[-1] == 4: image_sam = image_sam[:, :, :3]
        
        image_bgr = cv2.cvtColor(image_sam, cv2.COLOR_RGB2BGR)
        ih, iw, _ = image_sam.shape

        # --- 1. USER MASK ---
        final_boxes = []
        
        if mask_input is not None:
            if mask_input.ndim == 3: mask_gray = mask_input[:, :, 0]
            else: mask_gray = mask_input
            coords = cv2.findNonZero(mask_gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                final_boxes.append(np.array([x, y, x + w, y + h]))
                print("🚀 Mode: User Box Added")

        # --- 2. TEXT PROMPT ---
        if text_prompt is not None and text_prompt.strip() != "":
            print(f"🚀 Mode: Text Processing ('{text_prompt}')")
            
            # Nếu user đã vẽ box, ta crop vùng đó để tìm kiếm trong đó
            if len(final_boxes) > 0:
                ux1, uy1, ux2, uy2 = final_boxes[0] # Lấy box vẽ đầu tiên
                crop_bgr = image_bgr[uy1:uy2, ux1:ux2]
                
                if crop_bgr.size > 0:
                    local_boxes, local_labels, _ = self._run_detector(crop_bgr)
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Tìm trong crop
                    selected_local_boxes = self._select_boxes(crop_rgb, local_boxes, local_labels, text_prompt)
                    
                    if len(selected_local_boxes) > 0:
                        # Thay thế box vẽ thô bằng box chính xác (Mapped to Global)
                        final_boxes = [] # Clear box vẽ
                        for lb in selected_local_boxes:
                            lx1, ly1, lx2, ly2 = lb
                            final_boxes.append(np.array([lx1 + ux1, ly1 + uy1, lx2 + ux1, ly2 + uy1]))
                        print(f"   -> Refined to {len(final_boxes)} precise objects within drawing.")
            
            # Nếu user không vẽ, tìm trên toàn ảnh
            else:
                pred_boxes, pred_labels, _ = self._run_detector(image_bgr)
                selected_boxes = self._select_boxes(image_sam, pred_boxes, pred_labels, text_prompt)
                final_boxes.extend(selected_boxes)

        if len(final_boxes) == 0:
            return None, None, "❌ Không tìm thấy vật thể!"

        print(f"🎯 Total Objects to Remove: {len(final_boxes)}")

        # --- 3. SAM (MULTI-BOX) ---
        self.sam_predictor.set_image(image_sam)
        
        # Chuyển list boxes thành Tensor [N, 4]
        input_boxes_tensor = torch.tensor(np.array(final_boxes), device=self.device)
        input_boxes_tensor = self.sam_predictor.transform.apply_boxes_torch(input_boxes_tensor, image_sam.shape[:2])
        
        # SAM dự đoán 1 lúc nhiều box
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes_tensor,
            multimask_output=False
        )
        
        # masks shape: [N, 1, H, W] -> Gộp lại thành 1 mask duy nhất
        combined_mask = torch.any(masks, dim=0).squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # Dilate Mask
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # --- 4. LAMA ---
        print("🎨 Running LaMa Inpainting...")
        inpainted_image = self.lama.inpaint(image_sam, dilated_mask)
        # print("  Running RePaint (Diffusion) to create Coarse Image...")
        # coarse_image = self.repaint_conf.inpaint(image_input, dilated_mask)
        # mask_bool = (dilated_mask > 127).astype(np.float32) / 255.0
        # mask_3ch = np.expand_dims(mask_bool, axis=-1)
        
        # coarse_cleaned = (image_input.astype(np.float32) * (1 - mask_3ch) + 
        #                   coarse_image.astype(np.float32) * mask_3ch).astype(np.uint8)

        return dilated_mask, inpainted_image, f"  Đã xóa {len(final_boxes)} vật thể!"