import torch
from torch import nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, ss, ks):
        super(CNN, self).__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        pool_idx = 0
        
        for i, layer in enumerate(self.cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):        
                self.cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
                
    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.cnn.features(x)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        return conv
