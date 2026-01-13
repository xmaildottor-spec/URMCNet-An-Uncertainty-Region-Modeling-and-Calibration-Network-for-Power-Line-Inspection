import torch
import torch.nn as nn
from decoder_blocks import CBR, GeneralizedMeanPooling
from carafe import AHPFGenerator

class HFAM(nn.Module):
    """
    Hierarchical Feature Attention Module (HFAM).
    
    Structure:
    1. **Spatial Branch**: Extracts edges/details using AHPFGenerator.
       - Adds high-frequency details back to the feature map.
    2. **Channel Branch**: Computes channel importance weights.
       - Uses both GAP (Global Avg Pool) and GEP (Generalized Mean Pool).
    """
    def __init__(self, in_channels, reduction=16):
        super(HFAM, self).__init__()
        
        self.conv1x1 = CBR(in_channels, in_channels, 1, 1, 0)
        
        # Spatial Branch (High-Frequency Extraction)
        self.AHPFGenerator = AHPFGenerator(channels=in_channels, kernel_size=3)
        
        # Channel Branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gep = GeneralizedMeanPooling(1)
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv1x1(x)
        
        # --- Spatial Path ---
        # Extract details via AHPF and inject into features
        spatial_residual = self.AHPFGenerator(feat)
        spatial_out = feat + spatial_residual

        # --- Channel Path ---
        b, c, _, _ = feat.size()
        y_gap = self.gap(feat)
        y_gep = self.gep(feat)
        
        # Fuse pooling statistics
        y = y_gap + y_gep 
        
        # Compute weights
        weights = self.mlp(y).view(b, c, 1, 1)
        
        # --- Final Fusion ---
        out = spatial_out * weights
        
        return out