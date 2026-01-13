import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    """
    Standard Convolution-BatchNorm-ReLU Block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TBR(nn.Module):
    """
    Transposed Conv - BatchNorm - ReLU Block.
    Used for learnable upsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(TBR, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv(x)

class TransConvBnLeakyRelu2d(nn.Module):
    """
    Transposed Conv with LeakyReLU.
    Often used in final output heads for smoother gradients.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(TransConvBnLeakyRelu2d, self).__init__()      
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)  
        
        # Initialization
        for m in self.modules():            
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()        
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)   
                               
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))  

class GeneralizedMeanPooling(nn.Module):
    """
    GEP: Generalized Mean Pooling.
    Adaptive pooling that acts as a continuum between Max and Average pooling.
    """
    def __init__(self, output_size=1, p=3):
        super(GeneralizedMeanPooling, self).__init__()
        self.output_size = output_size
        self.p = p

    def forward(self, x):
        # Clamp to avoid NaN in pow()
        x = x.clamp(min=1e-6)
        return F.adaptive_avg_pool2d(x.pow(self.p), self.output_size).pow(1. / self.p)

class BridgeBlock(nn.Module):
    """
    The 'Bridge' Block (Yellow 'B' in diagrams).
    Fuses Encoder Feature (F_en) and Decoder Feature (F_d).
    
    Logic: (F_en * F_d) + (F_en - F_d) -> CBR
    This combines multiplicative interaction (common features) with difference interaction (boundary refinement).
    """
    def __init__(self, in_channels):
        super(BridgeBlock, self).__init__()
        self.cbr = CBR(in_channels, in_channels)

    def forward(self, f_en, f_d):
        # Assuming inputs are already aligned in spatial dimensions
        prod = f_en * f_d
        diff = f_en - f_d
        summ = prod + diff
        return self.cbr(summ)