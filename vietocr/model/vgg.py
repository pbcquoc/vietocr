import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter


class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden, dropout=0.1):
        super(Vgg, self).__init__()

        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=True)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=True)

        pool_idx = 0
        
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):        
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
 
#        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
#        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)
	
	return_layers = {'13': 'out_layer13', '26':'out_layer26', '39':'out_layer39', '52':'out_layer52'}
	self.features = model_with_multuple_layer = IntermediateLayerGetter(cnn.features, return_layers=return_layers)
	
	in_channels = [128, 256, 512, 512]
	self.last_conv1x1s = []

	for i, in_channel in enumerate(in_channels):
	    conv1x1 = nn.Conv2d(in_channel, 256, 1)    
	    self.last_conv1x1s.append(conv1x1)

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        features = self.features(x)
        outputs = [p for _, p in features.items()]

	
	scaled_outputs = []    
	for i in range(len(in_channels)):    
	    output = outputs[i]
	    output = self.last_conv1x1s[i](output)
	    output = torch.nn.functional.interpolate(output, scale_factor=2**i)
	    scaled_outputs.append(output)

        conv = torch.mean(torch.stack(scaled_outputs), dim=0)

#        conv = self.features(x)
        conv = self.dropout(conv)
#        conv = self.last_conv_1x1(conv)

#        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def vgg11_bn(ss, ks, hidden):
    return Vgg('vgg11_bn', ss, ks, hidden)

def vgg19_bn(ss, ks, hidden):
    return Vgg('vgg19_bn', ss, ks, hidden)
   
