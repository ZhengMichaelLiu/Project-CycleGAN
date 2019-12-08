#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn

# We use 6 residual blocks for 128 x 128 training images, and 9 residual blocks for 256 x 256 or higher-resolution training images.
# Let c7s1-k denote a 7 x 7 Convolution - InstanceNorm - ReLU layer with k filters and stride 1. 
# dk denotes a 3 x 3 Convolution - InstanceNorm - ReLU layer with k filters and stride 2. 
# Reflection padding was used to reduce artifacts.
# Rk denotes a residual block that contains two 3 x 3 convolutional layers with the same number of filters on both layer. 
# uk denotes a 3 x 3 fractional-strided-Convolution - InstanceNorm - ReLU layer with k filters and stride 1 / 2.

# The network with 6 residual blocks consists of: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3
# The network with 9 residual blocks consists of: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3

class C7S1_k(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C7S1_k, self).__init__()
        
        reflect_padding = nn.ReflectionPad2d(3)
        conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        norm_layer = nn.InstanceNorm2d(num_features = out_channels)
        relu_layer = nn.ReLU(True)
        
        sequence = [reflect_padding, conv_layer, norm_layer, relu_layer]
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        return self.model(x)

class Dk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dk, self).__init__()
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = True)
        norm_layer = nn.InstanceNorm2d(out_channels)
        relu_layer = nn.ReLU(True)
        
        self.model = nn.Sequential(conv_layer, norm_layer, relu_layer)
    
    def forward(self, x):
        return self.model(x)

class Rk(nn.Module):
    def __init__(self, dim):
        super(Rk, self).__init__()
        self.conv_block = self.build_block(dim)

    def build_block(self, dim):
        sequence = []
        sequence += [nn.ReflectionPad2d(1)]
        sequence += [nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 0, bias = True)]
        sequence += [nn.InstanceNorm2d(num_features = dim), nn.ReLU(True)]
        sequence += [nn.ReflectionPad2d(1)]
        sequence += [nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 0, bias = True)]
        sequence += [nn.InstanceNorm2d(num_features = dim)]
        
        return nn.Sequential(*sequence)
    
    def forward(self, x):
        return x + self.conv_block(x)

class Uk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Uk, self).__init__()
        conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = True)
        norm_layer = nn.InstanceNorm2d(num_features = out_channels)
        relu_layer = nn.ReLU(True)
        
        self.model = nn.Sequential(conv_layer, norm_layer, relu_layer)
        
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_of_res_block):
        super(Generator, self).__init__()
        model = []
        model += [C7S1_k(input_channels, 64), Dk(64, 128), Dk(128, 256)]
        for i in range(num_of_res_block):
            model += [Rk(256)]
        model += [Uk(256, 128), Uk(128, 64)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels = 64, out_channels = output_channels, kernel_size = 7, padding = 0), nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
