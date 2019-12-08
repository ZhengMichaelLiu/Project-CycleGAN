#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F

# For discriminator networks, we use 70 x 70 PatchGAN. 
# Let Ck denote a 4 x 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. 
# After the last layer, we apply a convolution to produce a 1-dimensional output. 
# We do not use InstanceNorm for the first C64 layer. 
# We use leaky ReLUs with a slope of 0.2. 
# The discriminator architecture is: C64-C128-C256-C512

class Ck(nn.Module):
    def __init__(self, in_channels, out_channels, inst_norm):
        super(Ck, self).__init__()
        
        conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1)
        
        norm_layer = nn.InstanceNorm2d(num_features = out_channels)
        
        lrelu_layer = nn.LeakyReLU(0.2, True)
        
        if inst_norm == True:
            sequence = [conv_layer, norm_layer, lrelu_layer]
        else:
            sequence = [conv_layer, lrelu_layer]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        sequence = [Ck(input_channels, 64, False), Ck(64, 128, True), Ck(128, 256, True), Ck(256, 512, True)]
        sequence += [nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)]
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

