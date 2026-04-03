import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, ops

class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super(FPN, self).__init__()
        # Dùng weights thay cho pretrained
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.c2 = nn.Sequential(*list(resnet.children())[:5])  
        self.c3 = list(resnet.children())[5]              
        self.c4 = list(resnet.children())[6]  
        self.c5 = list(resnet.children())[7]         
        self.lat_c5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lat_c2 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c2 = self.c2(x); c3 = self.c3(c2); c4 = self.c4(c3); c5 = self.c5(c4)
        p5 = self.lat_c5(c5)
        p4 = F.interpolate(p5, scale_factor=2) + self.lat_c4(c4)
        p3 = F.interpolate(p4, scale_factor=2) + self.lat_c3(c3)
        p2 = F.interpolate(p3, scale_factor=2) + self.lat_c2(c2)
        return [self.smooth(p2), self.smooth(p3), self.smooth(p4), self.smooth(p5)]

class MaskRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MaskRCNNHead, self).__init__()
        self.num_classes = num_classes
        # Lấy stride 8 cho p3
        self.roi_align = ops.RoIAlign(output_size=(7, 7), spatial_scale=1.0/8.0, sampling_ratio=2)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU()
        )
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, (num_classes + 1) * 4)

    def forward(self, features, rois):
        # rois phải là List[Tensor] hoặc Tensor [N, 5]
        pooled_feat = self.roi_align(features, rois)
        x = pooled_feat.flatten(1)
        x = self.fc(x)
        return self.cls_score(x), self.bbox_pred(x)

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.backbone_fpn = FPN(out_channels=256)
        self.roi_head = MaskRCNNHead(256, num_classes)

    def forward(self, x, targets=None):
        if isinstance(x, list): x = torch.stack(x)
        fpn_features = self.backbone_fpn(x)
        p3 = fpn_features[1] 

        rois = [t['boxes'] for t in targets] if targets else None
        cls_scores, bbox_preds = self.roi_head(p3, rois)

        if self.training and targets:
            labels = torch.cat([t['labels'] for t in targets])
            gt_boxes = torch.cat([t['boxes'] for t in targets])
            
            # Loss Cls
            loss_cls = F.cross_entropy(cls_scores, labels)

            # Loss BBox: Trích xuất đúng 4 tọa độ của class tương ứng
            # bbox_preds: [N, (C+1)*4] -> [N, C+1, 4]
            N = cls_scores.shape[0]
            bbox_preds = bbox_preds.reshape(N, -1, 4)
            # Lấy index của các nhãn thực tế
            picked_bbox = bbox_preds[torch.arange(N), labels]
            loss_reg = F.smooth_l1_loss(picked_bbox, gt_boxes)

            return {"loss_classifier": loss_cls, "loss_box_reg": loss_reg}
        
        return cls_scores, bbox_preds