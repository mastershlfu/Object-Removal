# End-to-End Object Removal & Inpainting Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![SAM](https://img.shields.io/badge/Meta-Segment_Anything-blue.svg)](https://github.com/facebookresearch/segment-anything)
[![SDXL](https://img.shields.io/badge/Diffusers-SDXL-purple.svg)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Đồ án môn học Computer Vision / Generative AI. Xây dựng một hệ thống hoàn chỉnh (pipeline) tự động phát hiện, trích xuất mask, xóa vật thể và tái tạo lại nền ảnh một cách chân thực bằng cách kết hợp YOLOv8, SAM, LaMa và Stable Diffusion XL.

---

## 📊 Dataset Overview

Tập dữ liệu tự thu thập phục vụ cho việc fine-tune mô hình YOLOv8m để nhận diện 12 lớp vật thể tùy chỉnh (phụ kiện, vật dụng thông thường).

| Metric           | Value |
|-----------------|------|
| Total Instances | 14,243 |
| Custom Classes  | 12 (Bracelets, Glasses, Necklaces, etc.) |
| COCO Classes    | 80 (Base YOLOv8x) |
| Source          | [Roboflow Custom Dataset](https://app.roboflow.com/dao-duong/my_dataset-8fpvk/1) |

---

## 📚 Project Modules & Phân công

| Phần | Nội dung | Người thực hiện | Role |
|------|----------|----------------|------|
| ⚙️ **Part 1** | Khởi tạo Project Skeleton, xây dựng pipeline End-to-End | **Hà Bảo Nhi** | Pipeline Architect |
| 🎯 **Part 2** | Parallel YOLO (v8x & v8m), trích xuất mask với SAM, refinement với SDXL| **Đào Quang Dương** | CV Engineer |
| 🪄 **Part 3** | LaMa Inpainting | **Hoàng Thị Hằng** | Generative AI Engineer |
| 📝 **Part 4** | Testing, tổng hợp kết quả, viết report & slide | **Đặng Châu Anh** | QA & Writer |

---

## 🗂️ Project Structure

```text
.
├── README.md
├── requirements.txt
├── remove_objects.py
├── src/
│   ├── inpaint/
│   │   └── lama.py
│   └── pipeline/
├── models/
│   ├── yolov8x.pt
│   ├── yolov8m_finetuned.pt
│   ├── sam_vit_h_4b8939.pth
│   └── lama/
│       └── big-lama.pt
├── data/
├── outputs/
└── .venv/
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd /path/to/Object-Removal

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Lưu ý:**  
Cần tải các file weights của YOLO, SAM, LaMa và SDXL và đặt đúng vào thư mục `models/`.

---

### 2. Run the Pipeline

```bash
python remove_objects.py
```

Kết quả sẽ được lưu tại thư mục `outputs/`.

---

## 🛠️ Tech Stack

| Category            | Tools & Libraries |
|--------------------|------------------|
| Language           | Python 3.8+ |
| Computer Vision    | OpenCV, PIL |
| Object Detection   | YOLOv8 (Ultralytics) |
| Segmentation       | Segment Anything (SAM) |
| Generative AI      | PyTorch, Diffusers (SDXL), LaMa |
| Hardware           | CUDA / GPU |

---

## 📜 License

Dự án phục vụ mục đích học tập và nghiên cứu.

- LaMa weights: saic-mdal  
- SAM: Meta Research  
- SDXL: HuggingFace Diffusers  

---

## 🔗 References

- YOLOv8 (Ultralytics)  
- Segment Anything Model (SAM)  
- LaMa Inpainting  
- Stable Diffusion XL (SDXL)  

---

**Last Updated:** April 2026