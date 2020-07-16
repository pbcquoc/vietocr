import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class resnet_fpn(nn.Module):
    def __init__(self, backbone):
        backbone = resnet_fpn_backbone(backbone, pretrained=True, trainable_backbone_layers=5)

        self.backbone = backbone['0']

    def forward(self, x):
        conv = self.backbone(x)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)

        return conv

