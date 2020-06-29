import torch
from torch import nn

from vietocr.model.vgg import Vgg
from vietocr.model.resnet import Resnet50

class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        if backbone == 'vgg':
            self.model = Vgg(**kwargs)
        elif backbone == 'resnet':
            self.model = Resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)
