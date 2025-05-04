import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_instance_segmentation_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = MaskRCNN(backbone, num_classes=num_classes + 1)  # add 1 for background
    return model
"""
import torch
import torch.nn as nn
import torchvision

class TransformerMaskPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.flatten_size = 14 * 14  # standard pooled ROI mask size
        self.embed_dim = 256

        self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8),
            num_layers=2
        )
        self.output_proj = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        N, C, H, W = x.shape  # x: [N, C, H, W]
        x = self.input_proj(x)  # [N, embed_dim, H, W]
        x = x.flatten(2).permute(2, 0, 1)  # [HW, N, embed_dim]
        x = self.transformer(x)  # [HW, N, embed_dim]
        x = x.permute(1, 2, 0).view(N, self.embed_dim, H, W)  # [N, embed_dim, H, W]
        x = self.output_proj(x)  # [N, num_classes, H, W]
        return x

def get_instance_segmentation_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = MaskRCNN(backbone, num_classes=num_classes + 1)  # +1 for background

    # Replace the default mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = TransformerMaskPredictor(in_features_mask, num_classes + 1)

    return model
"""