import torch
import torch.nn as nn
# Import sub-modules and blocks
from blocks import ConvBnRelu, ChannelAttention, SpatialAttention
from FPRFM import FPRFM
from FNRFM import FNRFM

class PFCM(nn.Module):
    """
    Primary Feature Calibration Mechanism (PFCM).
    
    Functionality:
        This is the core calibration unit. It integrates the False-Positive Suppression (FPRFM)
        and False-Negative Compensation (FNRFM). It then applies a Cross-Spatial Attention 
        mechanism to fuse the calibrated features effectively.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(PFCM, self).__init__()

        # Sub-modules for FP suppression and FN compensation
        self.fprfm = FPRFM(in_channels)
        self.fnrfm = FNRFM(in_channels)

        # Attention mechanisms for FP branch
        self.ca_fp = ChannelAttention(in_planes=in_channels)
        self.sa_fp = SpatialAttention()

        # Attention mechanisms for FN branch
        self.ca_fn = ChannelAttention(in_planes=in_channels)
        self.sa_fn = SpatialAttention()

        # Refinement Convolutions
        self.cbr_fp = ConvBnRelu(in_channels=in_channels, out_channels=in_channels)
        self.cbr_fn = ConvBnRelu(in_channels=in_channels, out_channels=in_channels)

        # Final Fusion Layer (Concatenation reduction)
        self.fuse = ConvBnRelu(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, padding=0)
       
    def forward(self, m, fp, fn):
        """
        Args:
            m (torch.Tensor): Main feature map.
            fp (torch.Tensor): False Positive uncertainty features.
            fn (torch.Tensor): False Negative uncertainty features.
        
        Returns:
            torch.Tensor: Calibrated and fused feature map.
        """
        # 1. Individual Calibration
        feature_fp = self.fprfm(m, fp) 
        feature_fn = self.fnrfm(m, fn) 

        # 2. Compute Attention Weights
        feature_fp_ca_weight = self.ca_fp(feature_fp) 
        feature_fp_sa_weight = self.sa_fp(feature_fp) 

        feature_fn_ca_weight = self.ca_fn(feature_fn) 
        feature_fn_sa_weight = self.sa_fn(feature_fn)

        # 3. Apply Channel Attention
        feature_fp_ca = self.cbr_fp(feature_fp * feature_fp_ca_weight)
        feature_fn_ca = self.cbr_fn(feature_fn * feature_fn_ca_weight)

        # 4. Cross-Spatial Attention Interaction
        # Note: FP features are weighted by FN spatial weights, and vice versa.
        feature_fp_sa = feature_fp_ca * feature_fn_sa_weight
        feature_fn_sa = feature_fn_ca * feature_fp_sa_weight

        # 5. Residual Connection
        feature_fp = feature_fp_sa + feature_fp
        feature_fn = feature_fn_sa + feature_fn

        # 6. Final Fusion
        out = self.fuse(torch.cat((feature_fp, feature_fn), dim=1))

        return out 

# ==============================================================================
# Unit Test
# ==============================================================================
if __name__ == '__main__':
    # Simulating input tensors: [Batch, Channels, Height, Width]
    # Assuming m, fp, fn have the same dimensions for calibration
    channels = 256
    input_main = torch.rand(2, channels, 32, 32)
    input_fp   = torch.rand(2, channels, 32, 32)
    input_fn   = torch.rand(2, channels, 32, 32)
    
    model = PFCM(in_channels=channels)
    
    # Forward Pass
    final_out = model(input_main, input_fp, input_fn)
    
    print(f"Input Shape: {input_main.shape}")
    print(f"Output Shape: {final_out.size()}")