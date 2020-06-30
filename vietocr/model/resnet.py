import torch
from torch import nn
from torchvision.models import resnet50

class Resnet50(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(2, 1), padding=3,
                                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=1)
        self.conv = nn.Conv2d(2048, hidden, 1)
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        h = self.conv(x)
        output = h.flatten(2).permute(2, 0, 1)
        return output
