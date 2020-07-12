import torch
from torch import nn
from torchvision import models


class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden):
        super(Vgg, self).__init__()
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=True)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=True)

        pool_idx = 0
        
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):        
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
 
        self.features = cnn.features
               
    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        try:
            conv = self.features(x)
        except:
            print(x.shape)
            raise Exception('abc')
        conv = self.last_conv_1x1(conv)

#        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def vgg11_bn(ss, ks, hidden):
    return Vgg('vgg11_bn', ss, ks, hidden)

def vgg19_bn(ss, ks, hidden):
    return Vgg('vgg19_bn', ss, ks, hidden)
   
