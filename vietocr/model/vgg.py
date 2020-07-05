import torch
from torch import nn
from torchvision import models
from einops import rearrange

class Vgg(nn.Module):
    def __init__(self, ss, ks, hidden):
        super(Vgg, self).__init__()
        self.conv = nn.Conv2d(512, hidden, 1)

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
        conv = self.features(x)
        conv = self.conv(conv)

#        conv = conv.squeeze(2)
#        conv = conv.permute(2, 0, 1)  # [w, b, c]
        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.permute(-1, 0, 1)
        return conv
