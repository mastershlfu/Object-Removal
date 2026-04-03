import gradio as gr
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
# Thêm đường dẫn root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.fasterRCNN_SAM_LaMa import ObjectRemovalSystem
from src.refinement.network import RefinementUNet

# --- CONFIG ---
DEVICE = "cuda"
# Kiểm tra kỹ đường dẫn file .pth của bạn
RCNN_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/fasterrcnn_epoch_7.pth"
SAM_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
LAMA_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/lama/big-lama.pt"

REFINE_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/refinement/models/refinement_logs/refine_epoch_5.pth"
REPAINT_CONF_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/RePaint/confs/test_p256_thin.yml"
# --- INIT SYSTEM ---
pipeline = ObjectRemovalSystem(RCNN_PATH, SAM_PATH, REPAINT_CONF_PATH, device=DEVICE)

# khởi tạo mạng Refinement
refine_net = RefinementUNet(in_channels=4, out_channels=3).to(DEVICE)
if os.path.exists(REFINE_PATH):
    print(f"  Loading Refinement weights from {REFINE_PATH}")
    refine_net.load_state_dict(torch.load(REFINE_PATH, map_location=DEVICE))
refine_net.eval()


def apply_gaussian_blur(mask_tensor, kernel_size=7, sigma=2.0):
    # mask_tensor: [1, 1, H, W]
    if kernel_size % 2 == 0: kernel_size += 1
    
    # Tạo gaussian kernel
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
    
    # Padding để giữ nguyên size
    pad = kernel_size // 2
    blurred_mask = F.conv2d(mask_tensor, gaussian_kernel, padding=pad)
    return blurred_mask

def run_refinement(coarse_img_rgb, mask_255, original_img_rgb):
    """
    Hàm Inference khớp 100% với logic Training mới
    """
    h, w = original_img_rgb.shape[:2]
    device = DEVICE

    # 1. Chuyển đổi Tensor
    coarse_t = torch.from_numpy(coarse_img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    # mask_255: 255 là lỗ, 0 là nền
    mask_t = torch.from_numpy(mask_255).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    orig_t = torch.from_numpy(original_img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # --- ĐỒNG BỘ LOGIC TRAINING ---
    # Trong train.py bạn có dòng: mask = 1.0 - mask
    # Giả sử mask_t ban đầu 1.0 là lỗ, sau khi đảo 0.0 là lỗ.
    mask_for_net = 1.0 - mask_t 
    
    # 2. Resize về 256 (size mà mạng đã học)
    coarse_in = F.interpolate(coarse_t, size=(256, 256), mode='bilinear')
    mask_in = F.interpolate(mask_for_net, size=(256, 256), mode='nearest')

    with torch.no_grad():
        # CNN Predict
        refined_t = refine_net(coarse_in, mask_in)
        
        # Resize kết quả về size gốc
        refined_t = F.interpolate(refined_t, size=(h, w), mode='bilinear')

    # 3. Làm mềm biên Mask (Gaussian Blur) như trong Training
    # Lưu ý: mask_blur dùng để blend, ta dùng mask_t gốc (1.0 là lỗ)
    mask_blur = apply_gaussian_blur(mask_t, kernel_size=7, sigma=2.0)

    # 4. Mix kết quả (Pasting Logic)
    # Trong train bạn dùng: coarse_img * (1 - mask_blur) + refined_pred * mask_blur
    # Ở Inference, ta dùng original_img thay cho coarse_img để vùng ngoài nét nhất có thể
    final_t = orig_t * (1.0 - mask_blur) + refined_t * mask_blur

    # 5. Convert back
    final_res = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return final_res

# def run_refinement(coarse_img_rgb, mask_255, original_img_rgb):
#     """
#     Hàm chạy CNN Refinement và thực hiện Pasting Logic
#     """
#     h, w = original_img_rgb.shape[:2]
    
#     # Preprocess: Chuyển về Tensor [0, 1]
#     coarse_t = torch.from_numpy(coarse_img_rgb).permute(2, 0, 1).float() / 255.0
#     mask_t = torch.from_numpy(mask_255).float() / 255.0
#     mask_t = mask_t.unsqueeze(0) # [1, H, W]
#     orig_t = torch.from_numpy(original_img_rgb).permute(2, 0, 1).float() / 255.0
    
#     # Resize về 256 (nếu mạng train ở 256)
#     coarse_t_input = torch.nn.functional.interpolate(coarse_t.unsqueeze(0), size=(256, 256)).to(DEVICE)
#     mask_t_input = torch.nn.functional.interpolate(mask_t.unsqueeze(0), size=(256, 256)).to(DEVICE)
    
#     with torch.no_grad():
#         # CNN Predict
#         refined_t = refine_net(coarse_t_input, mask_t_input)
        
#         # Resize ngược lại kích thước ảnh gốc
#         refined_t = torch.nn.functional.interpolate(refined_t, size=(h, w))
#         refined_t = refined_t.squeeze(0).cpu()

#     # --- PASTING LOGIC (CỰC KỲ QUAN TRỌNG) ---
#     # Giữ nguyên pixel ảnh gốc bên ngoài, chỉ lấy pixel CNN vẽ bên trong mask
#     mask_binary = (mask_t > 0.5).float()
    
#     mask_t = mask_t.cpu()
#     orig_t = orig_t.cpu()
    
#     print("mask: ", mask_t.shape)
#     print("origin: ", orig_t.shape)
#     print("refined: ", refined_t.shape)
    
#     final_t = (1 - mask_binary) * coarse_t + mask_binary * refined_t
    
#     # Convert back to Numpy RGB
#     final_res = (final_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     return final_res


def load_image_from_gradio(img_data):
    """
    Hàm helper để đọc ảnh từ Gradio bất kể nó trả về cái gì (Filepath hay Numpy)
    """
    if img_data is None:
        return None
        
    # Trường hợp 1: Gradio trả về đường dẫn file (String)
    if isinstance(img_data, str):
        # Đọc ảnh bằng OpenCV, chuyển sang RGB
        img = cv2.imread(img_data)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Trường hợp 2: Gradio trả về Numpy array
    if isinstance(img_data, np.ndarray):
        if img_data.shape[-1] == 4:
            return cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        return img_data
        
    return None

# def gradio_process(input_dict, text_prompt):
#     if input_dict is None:
#         return None, None, "⚠️ Vui lòng upload ảnh!"

#     # 1. Đọc ảnh nền (Background)
#     # Gradio mới trả về dict: {'background': ..., 'layers': ..., 'composite': ...}
#     raw_background = input_dict.get("background")
#     image = load_image_from_gradio(raw_background)
    
#     if image is None:
#         return None, None, None, "❌ Lỗi đọc ảnh gốc!"

#     # 2. Xử lý lớp vẽ (Mask)
#     layers = input_dict.get("layers", [])
#     mask = None
    
#     if len(layers) > 0:
#         raw_layer = layers[0]
#         draw_layer = load_image_from_gradio(raw_layer)
        
#         if draw_layer is not None and np.any(draw_layer):
#             # Nếu layer có 4 kênh (RGBA), lấy kênh Alpha làm mask
#             if draw_layer.shape[-1] == 4:
#                 mask = draw_layer[:, :, 3]
#             # Nếu là RGB/BGR (3 kênh) nhưng nền đen nét trắng
#             elif draw_layer.shape[-1] == 3:
#                 mask = cv2.cvtColor(draw_layer, cv2.COLOR_RGB2GRAY)
            
#             # Đảm bảo binary mask
#             mask[mask > 0] = 255

#     # 3. Gọi Pipeline
#     # Pipeline của bạn nhận vào: (image_rgb_numpy, mask_numpy, text)
#     try:
#         mask_result, coarse_result, status = pipeline.process(image, mask, text_prompt)
        
#         if coarse_result is None:
#             return None, None, None, status
        
#         final_refined = run_refinement(coarse_result, mask_result, image)
            
#         return mask_result, coarse_result, final_refined, status
    
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return None, None, None, f"❌ Lỗi hệ thống: {str(e)}"

def gradio_process(input_dict, text_prompt):
    img_rgb = input_dict["background"]
    mask_manual = None
    if input_dict.get("layers"):
        mask_manual = input_dict["layers"][0][:,:,3] # Lấy nét vẽ tay nếu có

    # 1. Lấy Mask từ SAM và Ảnh Coarse từ RePaint
    # Bước này sẽ mất khoảng 5-10 giây tùy cấu hình RePaint của bạn
    mask_res, coarse_res, status = pipeline.process(img_rgb, mask_manual, text_prompt)
    
    if coarse_res is None: return None, None, None, status

    # 2. Chạy Refinement CNN để làm mượt các nhiễu của RePaint
    final_refined = run_refinement(coarse_res, mask_res, img_rgb)
    
    return mask_res, coarse_res, final_refined, status

# --- UI LAYOUT ---
with gr.Blocks(title="Object Removal") as demo:
    gr.Markdown("# 🪄 Object Removal")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_editor = gr.ImageEditor(
                label="Input Image",
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=20),
                interactive=True,
                sources=["upload", "clipboard"],
                transforms=[], # Tắt crop mặc định cho đỡ rối
            )
            text_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. person...")
            btn_run = gr.Button("🚀 Run", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("1. Final Refined"):
                    output_final = gr.Image(label="Refined Result")
                with gr.Tab("2. Coarse (LaMa)"):
                    output_coarse = gr.Image(label="Coarse Result")
                with gr.Tab("3. Mask"):
                    output_mask = gr.Image(label="SAM Mask", image_mode="L")
            
            status_text = gr.Label(label="Status")

    btn_run.click(
        fn=gradio_process,
        inputs=[input_editor, text_prompt],
        outputs=[output_mask, output_coarse, output_final, status_text],
        api_name=False
    )

if __name__ == "__main__":
    # allow_flagging="never" để tắt tính năng lưu log gây lỗi
    demo.launch(
    share=True,
    allowed_paths=["/home/ml4u/BKTeam/source/BaoNhi/Object-Removal"],
    show_api=False   # 🔥 FIX LỖI
)