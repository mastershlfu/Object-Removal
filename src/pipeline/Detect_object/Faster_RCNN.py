import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class Faster_RCNN(nn.Module):
    def __init__(self, num_classes):
        super(Faster_RCNN, self).__init__()

        # 1. Khởi tạo Backbone (ResNet50 cắt 2 lớp cuối như bạn muốn)
        backbone_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Lấy đến layer4, bỏ avgpool và fc
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        
        # QUAN TRỌNG: FasterRCNN cần biết số channel đầu ra của backbone
        self.backbone.out_channels = 2048 

        # 2. Định nghĩa Anchor Generator (Cho RPN)
        # Các kích cỡ anchor phù hợp với ảnh 800x800
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # 3. Định nghĩa RoI Pooling (Dùng MultiScaleRoIAlign như bạn đã khai báo)
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'], # '0' là tên mặc định cho feature map từ Sequential
            output_size=7,
            sampling_ratio=2
        )

        # 4. Ghép tất cả vào khung FasterRCNN của torchvision
        # Khung này sẽ tự động xử lý RPN, Box Head, và tính Loss cho bạn
        self.model = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, targets=None):
        """
        images: List[Tensor] - Mỗi tensor là một ảnh [C, H, W]
        targets: List[Dict] - Chứa 'boxes' và 'labels'
        """
        # Khi train: trả về dict các loại loss
        # Khi eval: trả về list các dự đoán (boxes, scores, labels)
        return self.model(images, targets)