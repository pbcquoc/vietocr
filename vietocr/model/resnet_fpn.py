import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class resnet_fpn(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        backbone = resnet_fpn_backbone(backbone, pretrained=True)
       
        for n, p in backbone.named_parameters():
            p.requires_grad= True

        self.backbone = backbone

    def forward(self, x):
        conv = self.backbone(x)['1']
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)

        return conv

