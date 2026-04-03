import gradio as gr
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# =========================================================
# PATH SETUP
# =========================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inpaint.lama import LaMaInpainter
from src.inpaint.semantic_sd import SemanticInpainter
from src.utils.image_ops import refine_mask_iopaint, apply_post_process, match_histograms 
from src.pipeline.fasterRCNN_SAM_LaMa import ObjectRemovalSystem
from src.refinement.network import RefinementUNet

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RCNN_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/fasterrcnn_epoch_7.pth"
SAM_PATH  = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
LAMA_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/lama/big-lama.pt"
REFINE_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/refinement/models/refinement_logs/refine_epoch_5.pth"

# CSS để sửa lỗi lệch cọ và làm giao diện linh hoạt
css = """
.container { max-width: 1200px !important; margin: auto; }
/* Sửa lỗi lệch tọa độ cọ: Không dùng height cố định cho Editor */
.image-editor-container {
    min-height: 500px;
}
/* Đảm bảo canvas của Gradio khớp với khung hiển thị */
.gradio-image-editor canvas {
    object-fit: contain !important;
}
/* Hiệu ứng cọ vẽ có độ trong suốt để nhìn thấy vật thể bên dưới */
.image-editor-canvas canvas {
    opacity: 0.6;
}
"""

# =========================================================
# INIT MODELS
# =========================================================
# 1. Hệ thống cũ lấy Mask (SAM)
pipeline = ObjectRemovalSystem(RCNN_PATH, SAM_PATH, LAMA_PATH, device=DEVICE)

# 2. Hệ thống Inpaint mới (Stable Diffusion)
sd_painter = SemanticInpainter(device=DEVICE)

# 3. Mạng Refinement đã train (CNN)
refine_net = RefinementUNet(in_channels=4, out_channels=3).to(DEVICE)
if os.path.exists(REFINE_PATH):
    print(f"✅ Loaded Refinement weights from {REFINE_PATH}")
    refine_net.load_state_dict(torch.load(REFINE_PATH, map_location=DEVICE))
refine_net.eval()

# =========================================================
# FULL PIPELINE LOGIC (A -> B -> C)
# =========================================================

def apply_gaussian_blur(mask_tensor, kernel_size=7, sigma=2.0):
    if kernel_size % 2 == 0: kernel_size += 1
    x_coord = torch.arange(kernel_size).float()
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*np.pi*variance)) * torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(mask_tensor.device)
    pad = kernel_size // 2
    blurred_mask = F.conv2d(mask_tensor, gaussian_kernel, padding=pad)
    return blurred_mask

def run_cnn_refinement_step(coarse_img, raw_mask_255, original_img):
    """
    Sử dụng mạng CNN khớp 100% logic Training:
    1. Predict refined_pred từ (coarse, 1-mask)
    2. Tạo mask_soft bằng Gaussian Blur (k=11, sigma=3.0)
    3. Blend: (1-mask_soft)*GT + mask_soft*refined_pred
    """
    h, w = original_img.shape[:2]
    
    # 1. Chuyển đổi sang Tensor [1, C, H, W] dải [0, 1]
    coarse_t = torch.from_numpy(coarse_img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    mask_t = torch.from_numpy(raw_mask_255).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
    orig_t = torch.from_numpy(original_img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

    # --- BƯỚC 1: CNN PREDICT ---
    # Giống training: refined_pred = refine_net(coarse_img, 1.0 - mask)
    mask_for_net = 1.0 - mask_t 
    
    # Resize về 256 để đưa vào mạng (vì mạng train ở size này)
    coarse_in = F.interpolate(coarse_t, size=(256, 256), mode='bilinear')
    mask_in = F.interpolate(mask_for_net, size=(256, 256), mode='nearest')

    with torch.no_grad():
        refined_pred_t = refine_net(coarse_in, mask_in)
        # Resize kết quả dự đoán ngược lại kích thước gốc của ảnh
        refined_pred_t = F.interpolate(refined_pred_t, size=(h, w), mode='bilinear')

    # --- BƯỚC 2: SOFTEN MASK ---
    # Giống training: mask_soft = gaussian_blur(mask, kernel_size=11, sigma=3.0)
    # Lưu ý: mask_t ở đây có vùng xóa là 1.0
    mask_soft = apply_gaussian_blur(mask_t, kernel_size=11, sigma=3.0)

    # --- BƯỚC 3: GHÉP OUTPUT (ALPHA BLENDING) ---
    # Giống training: final_output = (1.0 - mask_soft) * gt_img + mask_soft * refined_pred
    # Ở inference, gt_img chính là original_img
    final_t = (1.0 - mask_soft) * orig_t + mask_soft * refined_pred_t

    # 4. Chuyển về Numpy uint8 để hiển thị
    final_np = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    return final_np

def gradio_process(editor_data, text_prompt):
    if editor_data is None or editor_data["background"] is None:
        return None, None, None, "⚠️ Please upload an image."

    img_rgb = editor_data["background"]
    if img_rgb.shape[-1] == 4: 
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    # 1. Lấy mask vẽ tay (alpha channel)
    mask_manual = None
    if editor_data.get("layers"):
        mask_manual = editor_data["layers"][0][:,:,3]
        if not np.any(mask_manual): mask_manual = None

    try:
        # --- BƯỚC 1: FasterRCNN + SAM -> Raw Mask ---
        # Ta chỉ dùng pipeline để lấy raw_mask
        raw_mask, _, status = pipeline.process(img_rgb, mask_manual, text_prompt)
        if raw_mask is None: 
            return None, None, None, "❌ Không tìm thấy vật thể!"

        # --- BƯỚC 2: (A) MASK REFINE (Dilate + Blur) ---
        # Tạo mask mờ để SD và Blending dùng
        mask_alpha = refine_mask_iopaint(raw_mask, dilate_k=15, blur_k=21)

        # --- BƯỚC 3: TẠO COARSE IMAGE (Sử dụng LaMa có sẵn trong pipeline) ---
        # Lưu ý: Không khởi tạo lại LaMa ở đây (tốn RAM), dùng cái đã init trong pipeline
        print("🎨 Step 3: Generating Base Inpaint with LaMa...")
        coarse_img = pipeline.lama.inpaint(img_rgb, raw_mask)

        # --- BƯỚC 4: CNN REFINEMENT (Mạng bạn tự train 20 epoch) ---
        # Đầu vào là ảnh LaMa và Raw Mask. Trả về ảnh đã tút tát texture.
        print("🪄 Step 4: Refining Texture with Custom CNN...")
        refined_cnn = run_cnn_refinement_step(coarse_img, raw_mask, img_rgb)
        
        # --- BƯỚC 5: (B) SEMANTIC INPAINT (Stable Diffusion) ---
        # SD sẽ lấy ảnh đã được CNN sửa lỗi làm base để vẽ tiếp nội dung ngữ nghĩa
        # print("🚀 Step 5: Enhancing Semantics with Stable Diffusion...")
        # sd_prompt = f"{text_prompt}, high quality background, seamless" if text_prompt else "seamless background, high quality"
        
        # # SD inpaint nhận ảnh đầu vào là refined_cnn
        # final_sd = sd_painter.inpaint(refined_cnn, mask_alpha, prompt=sd_prompt)

        # --- BƯỚC 6: (C) POST-PROCESS (Histogram Match + Blending) ---
        # Bước này giúp ảnh SD "ăn" hoàn toàn vào màu sắc của ảnh gốc ban đầu
        print("✨ Step 6: Final Blending...")
        final_result = apply_post_process(img_rgb, refined_cnn, mask_alpha)

        # Trả về: Mask Alpha | Ảnh sau LaMa | Ảnh Cuối cùng (SD + PostProcess)
        return (mask_alpha * 255).astype(np.uint8), coarse_img, final_result, f"✅ Done! {status}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"❌ Error: {str(e)}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"❌ Error: {str(e)}"

# =========================================================
# UI DESIGN
# =========================================================
with gr.Blocks(title="Object Removal Pro", css=css) as demo:
    gr.Markdown("## 🪄 Professional Object Removal (A->B->C Pipeline)")

    with gr.Row():
        # Cột trái: Input
        with gr.Column():
            input_editor = gr.ImageEditor(
                label="Input & Draw Mask",
                type="numpy",
                # Cọ màu vàng (#FFC107) có độ trong suốt 0.5 (80 trong hexa)
                brush=gr.Brush(colors=["#FFC10780"], color_mode="fixed", default_size=35),
                height= 1000,
                sources=["upload", "clipboard"],
                elem_classes=["image-editor-container"],
                transforms=[] # Giữ nguyên tỷ lệ để tránh lệch tọa độ
            )
            text_prompt = gr.Textbox(label="Text Prompt (Optional)", placeholder="e.g. remove person...")
            btn_run = gr.Button("🚀 Run Pipeline", variant="primary")

        # Cột phải: Results
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("✨ Final Result"):
                    output_final = gr.Image(label="Refined & Blended")
                with gr.Tab("🖼️ SD Coarse"):
                    output_coarse = gr.Image(label="Stable Diffusion Output")
                with gr.Tab("🎭 Soft Mask"):
                    output_mask = gr.Image(label="Refined Mask (Alpha)", image_mode="L")
            
            status_text = gr.Label(label="Status")

    btn_run.click(
        fn=gradio_process,
        inputs=[input_editor, text_prompt],
        outputs=[output_mask, output_coarse, output_final, status_text],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(share=True, show_api=False)