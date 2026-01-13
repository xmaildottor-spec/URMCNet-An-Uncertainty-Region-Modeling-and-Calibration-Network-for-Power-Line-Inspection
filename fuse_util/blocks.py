import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    """
    Standard Convolution Block: Conv2d -> BatchNorm2d -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        
        # Weights Initialization (Xavier)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    Compresses spatial dimension to compute channel-wise importance.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP: Dimension reduction -> Dimension restoration
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    Compresses channel dimension to compute spatial importance.
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # Concatenate Max and Avg pooling results, then convolve
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Edge_Weight(nn.Module):
    """
    High-Frequency Extractor.
    Extracts edge/texture information by subtracting the global average pooling 
    (low-frequency) from the original feature.
    """
    def __init__(self, in_channels, out_channels):
        super(Edge_Weight, self).__init__()
        self.conv = ConvBnRelu(in_channels=in_channels, out_channels=out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a):
        # 1. Compute Global Average (Low Frequency) -> [B, C, 1, 1]
        b = torch.nn.functional.adaptive_avg_pool2d(a, (1, 1))
        
        # 2. Subtraction: Feature - Mean = High Frequency (Edges/Texture)
        edge_feature = a - b 

        c = self.conv(edge_feature)
        return self.sigmoid(c)

class ReverseAttention(nn.Module):
    """
    Reverse Attention Module.
    Highlights the background or regions NOT activated by the attention map.
    """
    def __init__(self, in_channels):
        super(ReverseAttention, self).__init__()
        self.conv_atten = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, input_feature):
        # Generate Attention Map (0~1)
        atten_map = torch.sigmoid(self.conv_atten(input_feature))
        
        # Generate Reverse Attention (1 - Map)
        reverse_atten_map = 1 - atten_map
        
        # Apply to input: suppress foreground, highlight background
        out = input_feature * reverse_atten_map
        return out

class SSA(nn.Module):
    """
    Shared Spatial Attention (SSA).
    Fuses two features spatially and re-weights both.
    """
    def __init__(self, in_channels):
        super(SSA, self).__init__()
        self.conv1 = ConvBnRelu(2*in_channels, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b):
        ab = torch.cat((a, b), dim=1)
        ab = self.conv1(ab)
        ab_w = self.sigmoid(ab)
        
        # Weight both inputs by the shared map and sum them
        a = a * ab_w
        b = b * ab_w
        return a + b 

class SCA(nn.Module):
    """
    Simple Channel Attention (SCA).
    Global pooling followed by 1x1 conv to re-weight channels.
    """
    def __init__(self, in_channels):
        super(SCA, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, a):
        a = torch.nn.functional.adaptive_avg_pool2d(a, (1, 1))
        out = self.conv1(a) 
        return out