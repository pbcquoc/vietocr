import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck
from torch.hub import load_state_dict_from_url

resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

class Resnet50(nn.Module):
    def __init__(self, ss, hidden=256):
        super().__init__()
        backbone = resnet50(pretrained=False)
        backbone.inplanes = 64

        layers = [3, 4, 6, 3]
        block = Bottleneck
        backbone.conv1 = nn.Conv2d(3, backbone.inplanes, kernel_size=7, stride=ss[0], padding=3, bias=False)
        backbone.layer1 = backbone._make_layer(block, 64, layers[0], stride=ss[1])
        backbone.layer2 = backbone._make_layer(block, 128, layers[1], stride=ss[2])
        backbone.layer3 = backbone._make_layer(block, 256, layers[2], stride=ss[3])
        backbone.layer4 = backbone._make_layer(block, 512, layers[3], stride=ss[4])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=1)

                
        state_dict = load_state_dict_from_url(resnet50_url,
                                              progress=True)
        backbone.load_state_dict(state_dict)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.last_conv_1x1 = nn.Conv2d(2048, hidden, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        conv = self.last_conv_1x1(x)
#        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)

        return conv

