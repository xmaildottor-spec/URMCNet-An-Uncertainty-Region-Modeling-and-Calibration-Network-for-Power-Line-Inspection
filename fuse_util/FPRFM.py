import torch
import torch.nn as nn
from blocks import ConvBnRelu, ReverseAttention, SSA

class FPRFM(nn.Module):
    """
    False-Positive Region Suppression Module (FPRFM).
    
    Functionality:
        This module is designed to suppress False Positive (FP) regions in the 
        feature maps. It explicitly models the FP characteristics and uses 
        Reverse Attention to filter them out from the main feature stream.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(FPRFM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.ssa = SSA(in_channels=in_channels)
        self.conv = ConvBnRelu(in_channels=in_channels, out_channels=in_channels)
        self.neg = ReverseAttention(in_channels=in_channels)

    def forward(self, m, fp):
        """
        Args:
            m (torch.Tensor): Main feature map.
            fp (torch.Tensor): False Positive feature map (from uncertainty branch).
        """
        # 1. Extract high-frequency FP details (Deviation from mean)
        fp_gap = torch.nn.functional.adaptive_avg_pool2d(fp, (1, 1))
        fp_m = fp - fp_gap
        
        # 2. Enhance FP features
        fp_m_weight = self.sigmoid(fp_m)
        fp_m_en = fp * fp_m_weight

        # 3. Generate Reverse Attention on FP features (Focus on non-FP regions)
        fp_r = self.neg(fp_m)
        fp_r_weight = 1.0 - fp_m_weight

        # 4. Calibration
        # m1: Subtract the FP components from Main
        m1 = m - fp_m_en
        # m2: Weight Main by the reverse FP weight (suppression)
        m2 = m * fp_r_weight
        # m3: Shared Spatial Attention fusion
        m3 = self.ssa(m, fp_r)

        out = self.conv(m1 + m2 + m3)
        return out