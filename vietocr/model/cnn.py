import torch
from torch import nn

import vietocr.model.vgg as vgg
from vietocr.model.resnet import Resnet50

class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        if backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'resnet':
            self.model = Resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
