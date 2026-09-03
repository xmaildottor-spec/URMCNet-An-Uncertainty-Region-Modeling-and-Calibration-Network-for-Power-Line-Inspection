import warnings
warnings.filterwarnings("ignore")
import torch 
from torch import nn
import torch.nn.functional as F

#==================================================================================
class ConvBnrelu2d_3(nn.Module):
    # Convolution
    # Batch Normalization
    # ReLU
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        
class ConvBnrelu2d_1(nn.Module):
    # Convolution
    # Batch Normalization
    # ReLU
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class TransConvBnLeakyRelu2d(nn.Module):
    # Deconvolution (Transposed Convolution)
    # Batch Normalization
    # Leaky ReLU
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(TransConvBnLeakyRelu2d, self).__init__()      
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)  
        for m in self.modules():            
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()        
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)   
                               
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))  
#==================================================================================

class Decoder_base(nn.Module):         
    def __init__(self, channel_0=64, channel_1=256, channel_2=512, channel_3=1024, n_class=1):
        """
        Initialize the Primary Decoder based on the 4-layer primary feature architecture.
        channel_0: Number of channels for m0 (64)
        channel_1: Number of channels for m1 (256)
        channel_2: Number of channels for m2 (512)
        channel_3: Number of channels for m3 (1024) - Deepest feature layer F4
        """
        super(Decoder_base,self).__init__()
        
        # Initial convolution for processing the deepest features
        self.fuse = ConvBnrelu2d_3(channel_3, channel_3)
        
        # Stage 1: Decode channel_3 -> channel_2
        self.fusion_decoder0 = TransConvBnLeakyRelu2d(channel_3, channel_2)
        self.high1 = ConvBnrelu2d_1(channel_3, channel_2)
        self.fusion_conv1 = ConvBnrelu2d_3(channel_2, channel_2)
        
        # Stage 2: Decode channel_2 -> channel_1
        self.fusion_decoder1 = TransConvBnLeakyRelu2d(channel_2, channel_1)     
        self.high2 = ConvBnrelu2d_1(channel_3, channel_1)
        self.fusion_conv2 = ConvBnrelu2d_3(channel_1, channel_1)
        
        # Stage 3: Decode channel_1 -> channel_0
        self.fusion_decoder2 = TransConvBnLeakyRelu2d(channel_1, channel_0)      
        self.high3 = ConvBnrelu2d_1(channel_3, channel_0)
        self.fusion_conv3 = ConvBnrelu2d_3(channel_0, channel_0)
        
        # Final Upsample: channel_0 -> channel_0 (Restore to original resolution)
        self.fusion_decoder3 = TransConvBnLeakyRelu2d(channel_0, channel_0) 
        self.fusion_conv4 = ConvBnrelu2d_3(channel_0, channel_0)
        
        # Prediction Head
        self.fusion_conv5 = nn.Conv2d(channel_0, n_class, kernel_size=1, padding=0, stride=1, bias=False)
        nn.init.xavier_uniform_(self.fusion_conv5.weight.data)

    def forward(self, m0, m1, m2, m3):  
        # Feature input and upsampling addition corresponding to Figure 2(b)
        fusion = self.fuse(m3)  
        high = m3  
        
        # --- Stage 1: Fuse with m2 ---
        fusion = self.fusion_decoder0(fusion)  
        high1 = F.interpolate(self.high1(high), scale_factor=2, mode='bilinear', align_corners=True) 
        fusion = fusion + m2 + high1 
        fusion = self.fusion_conv1(fusion) 

        # --- Stage 2: Fuse with m1 ---
        fusion = self.fusion_decoder1(fusion)
        high2 = F.interpolate(self.high2(high), scale_factor=4, mode='bilinear', align_corners=True)
        fusion = fusion + m1 + high2 
        fusion = self.fusion_conv2(fusion)

        # --- Stage 3: Fuse with m0 ---
        fusion = self.fusion_decoder2(fusion)
        high3 = F.interpolate(self.high3(high), scale_factor=8, mode='bilinear', align_corners=True)
        fusion = fusion + m0 + high3
        fusion = self.fusion_conv3(fusion)

        # Retain the final feature prior to upsampling to feed into subsequent FP/FN encoders
        last_feature = fusion 

        # --- Final Stage: Restore to original image resolution and output prediction ---
        fusion = self.fusion_decoder3(fusion)
        fusion = self.fusion_conv4(fusion) 
        output = self.fusion_conv5(fusion)  

        return output, last_feature


if __name__=='__main__':
    # Simulate the output features of a 4-layer encoder (Batch=2)
    m0 = torch.rand(2, 64, 128, 128)
    m1 = torch.rand(2, 256, 64, 64)
    m2 = torch.rand(2, 512, 32, 32)
    m3 = torch.rand(2, 1024, 16, 16)
  
    # Instantiate the Primary Decoder (passing the corresponding channel dimensions)
    decoder_net = Decoder_base(channel_0=64, channel_1=256, channel_2=512, channel_3=1024, n_class=1)  
    
    # Forward pass
    pre, last_feat = decoder_net(m0, m1, m2, m3)

    print("Final Prediction Mask Shape:", pre.size())        # Expected: [2, 1, 256, 256]
    print("Low-level Feature Shape for Subsequent Branches (Last Feature):", last_feat.size()) # Expected: [2, 64, 128, 128]