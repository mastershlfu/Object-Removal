import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class PSRoIPooling(nn.Module):
    def __init__(self, k=3, num_classes=21):
        super(PSRoIPooling, self).__init__()
        self.k = k 
        self.num_classes = num_classes

    def forward(self, feature_map, rois):
        N = rois.size(0)
        C = self.num_classes
        k = self.k
        _, _, H, W = feature_map.shape
        
        all_roi_features = []

        for i in range(N):
            roi = rois[i]
            batch_idx = roi[0].long()
            # Giữ các tọa độ ở dạng float để không làm mất grad_fn của feature_map khi slice
            x1, y1, x2, y2 = roi[1], roi[2], roi[3], roi[4]
            
            w = torch.clamp(x2 - x1, min=1.0)
            h = torch.clamp(y2 - y1, min=1.0)
            bin_w = w / k
            bin_h = h / k

            roi_bins = []
            for row in range(k):
                for col in range(k):
                    idx = (row * k + col)
                    start_c = idx * C
                    end_c = start_c + C
                    
                    # Tính toán index cẩn thận
                    y_start = torch.floor(y1 + row * bin_h).long().clamp(0, H-1)
                    y_end = torch.ceil(y1 + (row + 1) * bin_h).long().clamp(1, H)
                    x_start = torch.floor(x1 + col * bin_w).long().clamp(0, W-1)
                    x_end = torch.ceil(x1 + (col + 1) * bin_w).long().clamp(1, W)

                    # TRÁNH LỖI: Slice trực tiếp từ feature_map để giữ grad_fn
                    if y_end > y_start and x_end > x_start:
                        crop = feature_map[batch_idx:batch_idx+1, start_c:end_c, y_start:y_end, x_start:x_end]
                        bin_val = torch.mean(crop, dim=(2, 3), keepdim=True) # [1, C, 1, 1]
                    else:
                        bin_val = torch.zeros((1, C, 1, 1), device=feature_map.device)
                    
                    roi_bins.append(bin_val)
            
            # Ghép k*k bins thành [C, k, k]
            # Dùng reshape thay vì view để an toàn
            roi_feat = torch.cat(roi_bins, dim=2).reshape(C, k, k)
            all_roi_features.append(roi_feat)
        
        return torch.stack(all_roi_features, dim=0)

class R_FCN(nn.Module):
    def __init__(self, num_classes=1204, k=3):
        super(R_FCN, self).__init__()
        self.k = k
        self.num_classes = num_classes 
        
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Đảm bảo backbone được phép update
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.ps_conv_cls = nn.Conv2d(2048, k * k * self.num_classes, kernel_size=1)
        self.ps_conv_reg = nn.Conv2d(2048, k * k * 4, kernel_size=1)

        self.ps_roi_pool_cls = PSRoIPooling(k, self.num_classes)
        self.ps_roi_pool_reg = PSRoIPooling(k, 4)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)

        # 1. Đi qua backbone (grad_fn bắt đầu từ đây)
        base_feat = self.backbone(images) 

        # 2. Score maps
        score_maps = self.ps_conv_cls(base_feat) 
        reg_maps = self.ps_conv_reg(base_feat)

        if self.training and targets is not None:
            all_rois_list = []
            all_labels = []
            all_boxes = []

            for i, t in enumerate(targets):
                if t['boxes'].shape[0] > 0:
                    b_idx = torch.full((t['boxes'].size(0), 1), i, device=images.device)
                    all_rois_list.append(torch.cat([b_idx, t['boxes']], dim=1))
                    all_labels.append(t['labels'])
                    all_boxes.append(t['boxes'])
            
            if not all_rois_list:
                return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}

            rois = torch.cat(all_rois_list, dim=0)
            target_labels = torch.cat(all_labels, dim=0)
            target_boxes = torch.cat(all_boxes, dim=0)

            # 3. Pooling (Vẫn giữ grad_fn từ score_maps)
            pooled_cls = self.ps_roi_pool_cls(score_maps, rois)
            pooled_reg = self.ps_roi_pool_reg(reg_maps, rois)

            cls_score = torch.mean(pooled_cls, dim=(2, 3)) 
            bbox_pred = torch.mean(pooled_reg, dim=(2, 3)) 

            return {
                "loss_classifier": F.cross_entropy(cls_score, target_labels),
                "loss_box_reg": F.smooth_l1_loss(bbox_pred, target_boxes)
            }
            
        return score_maps