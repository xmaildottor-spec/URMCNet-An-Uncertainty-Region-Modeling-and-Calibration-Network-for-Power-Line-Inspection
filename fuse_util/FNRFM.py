import torch
import torch.nn as nn
from blocks import ConvBnRelu, SCA, Edge_Weight

class FNRFM(nn.Module):
    """
    False-Negative Region Compensation Module (FNRFM).
    
    Functionality:
        This module aims to recover False Negative (FN) regions (missed detections).
        It highlights boundary/edge information in the FN features and injects 
        them back into the main feature stream.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(FNRFM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.cbr = ConvBnRelu(in_channels=in_channels, out_channels=in_channels)
        self.sca = SCA(in_channels=in_channels)
        self.edge = Edge_Weight(in_channels=in_channels, out_channels=in_channels)

    def forward(self, m, fn):
        """
        Args:
            m (torch.Tensor): Main feature map.
            fn (torch.Tensor): False Negative feature map (from uncertainty branch).
        """
        # 1. Extract edge weights from FN features and enhance FN
        fn_r = self.edge(fn) * fn
        
        # 2. Fuse Main and Enhanced FN
        mfn = m + fn_r
        mfn_weight = self.sigmoid(mfn)
        
        # 3. Refine the weighted combination
        m_en = self.cbr(m * mfn_weight + fn)
        
        # 4. Apply Simple Channel Attention based on FN global stats
        out = m_en * self.sca(fn)
        return out