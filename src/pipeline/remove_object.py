import torch
import numpy as np
import cv2
import os
import torchvision.transforms as T

from utils.get_img import get_image_path_from_txt
from segment_anything import sam_model_registry, SamPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from src.inpaint.lama import LaMaInpainter
from transformers import pipeline
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline

os.environ['HF_HOME'] = "/media/ml4u/Challenge-4TB/baonhi"

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
IMAGE_PATH = get_image_path_from_txt()

INPUT_BOX = np.array([211, 526, 581, 1339])

NUM_CLASSES = 91 

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

class SDXLRefiner:
    def __init__(self, device=DEVICE):
        print("Loading SDXL Inpainting Pipeline (User Config)...")
        sdxl_path = "/media/ml4u/Challenge-4TB/baonhi/sdxl-inpaint-offline"

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            sdxl_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True
        ).to(device)
        
        self.pipe.enable_model_cpu_offload() 
    

    def refine(self, image_rgb, org_rgb, mask_uint8, dynamic_neg_prompt, strength = 0.5):
        """
        Input: 
            - image_rgb: Ảnh Numpy (H, W, 3) từ LaMa
            - mask_uint8: Mask Numpy (H, W) từ SAM
        """
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(mask_uint8)

        w_org, h_org = image_pil.size
        w = w_org - (w_org % 8)
        h = h_org - (h_org % 8)
        
        if (w, h) != image_pil.size:
            image_pil = image_pil.resize((w, h))
            mask_pil = mask_pil.resize((w, h))
        
        print(f"objects to remove: {dynamic_neg_prompt}")

        result = self.pipe(
            prompt = "clean architectural background, suitable context, clear surroundings, seamless texture, continuous surface, photorealistic, perfectly blended, highly detailed, matching lighting, natural continuation",
            negative_prompt=dynamic_neg_prompt + ", new objects, additional items, ghosts, shadows, distinct subjects, people, animals, vehicles, furniture, decor, text, watermark, artifacts, geometric shapes, blur",
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=100,    
            guidance_scale=8.0,
            height=h, # Ép chiều cao
            width=w,  # Ép chiều rộng
            strength=strength
        ).images[0]

        result_rgb = np.array(result.resize((w_org, h_org)))
        
        # Làm nhòe viền mask để vết cắt tàng hình
        blend_mask = cv2.GaussianBlur(mask_uint8, (21, 21), 0) 
        alpha = blend_mask.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        # Hợp nhất: Lõi AI vẽ + Viền và cảnh thật
        perfect_result = (result_rgb * alpha) + (org_rgb * (1.0 - alpha))
        
        return perfect_result.astype(np.uint8)


class ObjectRemover:
    def __init__(self, rcnn_path, sam_path, lama_path, img_path=IMAGE_PATH, device=DEVICE):
        self.device = device
        self.img_path = img_path

        # RCNN Detector
        self.detector = self._load_faster_rcnn(rcnn_path)

        # SAM Predictor
        self.sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_path)
        self.sam.to(device)
        self.sam_predictor = SamPredictor(self.sam)

        # LaMa
        self.lama = LaMaInpainter(lama_path, device=device)

        # SDXL
        try:
            self.sdxl = SDXLRefiner(device=device)
            self.use_sdxl = True
        except Exception as e:
            print(f"⚠️ Lỗi load SDXL: {e}")
            self.use_sdxl = False


    def _load_faster_rcnn(self, model_path):
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

        ckpt = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        
        model.to(self.device)
        model.eval()
        return model
    
    def read_image(self):
        IMAGE_PATH = get_image_path_from_txt()
        self.img_path = IMAGE_PATH
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại {self.img_path}")
        
        img_bgr = cv2.imread(self.img_path)
        if img_bgr is None:
            raise ValueError(f"Không thể đọc ảnh từ {self.img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def _box_in_any_roi(self, box, roi_boxes):
        x1, y1, x2, y2 = box

        for roi in roi_boxes:
            rx1, ry1, rx2, ry2 = map(int, [
                roi.xmin, roi.ymin, roi.xmax, roi.ymax
            ])

            if x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1:
                return True

        return False


    def scan_for_objects(self, user_roi_boxes=None):
        img_rgb = self.read_image()
        h, w = img_rgb.shape[:2]

        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img_rgb).to(self.device)

        with torch.no_grad():
            predictions = self.detector([img_tensor])[0]

        detected_objects = []

        for i in range(len(predictions["boxes"])):
            score = predictions["scores"][i].item()
            if score < 0.5:
                continue

            box = predictions["boxes"][i].cpu().numpy()
            label_id = predictions["labels"][i].item()

            abs_box = box.tolist()

            # nếu có ROI → check overlap
            if user_roi_boxes is not None:
                if not self._box_in_any_roi(abs_box, user_roi_boxes):
                    continue

            detected_objects.append({
                "box": abs_box,
                "label": COCO_INSTANCE_CATEGORY_NAMES[label_id],
                "score": score
            })

        return detected_objects

    
    def remove_objects(self, target_boxes):
        # 1. Tạo mask từ target_boxes bằng SAM
        # 2. Inpaint vùng mask bằng LaMa
        # 3. Refine kết quả bằng SDXL và 1 mạng refine nữa
        img_rgb = self.read_image()
        if len(target_boxes) == 0:
            return img_rgb

        self.sam_predictor.set_image(img_rgb)
        full_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

        labels_to_remove = []

        for box in target_boxes:
            input_box = np.array([box.xmin, box.ymin, box.xmax, box.ymax])
            masks, scores, _ = self.sam_predictor.predict(
                box=input_box[None, :],
                multimask_output=False
            )

            full_mask[masks[0] > 0] = 255

            if hasattr(box, 'label') and box.label:
                labels_to_remove.append(box.label)
        
        unique_labels = list(set(labels_to_remove))
        dynamic_neg_prompt = ", ".join(unique_labels)

        kernel = np.ones((30, 30), np.uint8)
        dilated_mask = cv2.dilate(full_mask, kernel, iterations=1)

        # 3. Inpainting với LaMa
        lama_result = self.lama.inpaint(img_rgb, dilated_mask)

        sdxl_kernel = np.ones((30,30), np.uint8)
        sdxl_mask = cv2.dilate(full_mask, sdxl_kernel, iterations=1)
        sdxl_mask = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
        # sdxl_mask = dilated_mask

        # 4. Refine với SDXL
        if self.use_sdxl:
            print("Running SDXL Refiner...")
            final_result = self.sdxl.refine(lama_result, img_rgb, sdxl_mask, dynamic_neg_prompt, strength=0.9)
            final_result = self.sdxl.refine(final_result, img_rgb, sdxl_mask, dynamic_neg_prompt, strength=0.35) 

            return final_result
        
        return img_rgb

def main():
    print("🚀 Starting Object Removal Pipeline...")

if __name__ == "__main__":
    main()