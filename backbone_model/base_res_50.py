"""
Two-Stream Fusion Network with ResNet50 Backbone
=======================================================

    model = Net(n_class=2)
    output = model(rgb_image, thermal_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple

# ==============================================================================
# Basic Building Blocks
# ==============================================================================

class ConvBnRelu3x3(nn.Module):
    """
    Standard Convolutional Block: 3x3 Conv -> BatchNorm -> ReLU.
    Used for feature refinement and smoothing.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ConvBnRelu1x1(nn.Module):
    """
    Pointwise Convolutional Block: 1x1 Conv -> BatchNorm -> ReLU.
    Used for channel reduction (bottleneck) and feature alignment.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


# ==============================================================================
# Backbone Encoder
# ==============================================================================

class ResNet50Encoder(nn.Module):
    """
    ResNet50 Feature Extractor.
    
    This class loads a pre-trained ResNet50 and extracts features from 
    intermediate layers (Layer 1-4) to provide multi-scale representations.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load weights: Use DEFAULT for best available pre-trained weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # Initial stem (Input -> 1/4 Scale)
        self.stem_conv = resnet.conv1      
        self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu
        self.stem_maxpool = resnet.maxpool 

        # Residual Layers
        self.layer1 = resnet.layer1   # Output: 256 channels, 1/4 scale
        self.layer2 = resnet.layer2   # Output: 512 channels, 1/8 scale
        self.layer3 = resnet.layer3   # Output: 1024 channels, 1/16 scale
        self.layer4 = resnet.layer4   # Output: 2048 channels, 1/32 scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input image [Batch, 3, H, W]

        Returns:
            Tuple containing features from Layer 1, 2, 3, and 4.
        """
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.stem_maxpool(x)

        feat_l1 = self.layer1(x)  # stride 4
        feat_l2 = self.layer2(feat_l1) # stride 8
        feat_l3 = self.layer3(feat_l2) # stride 16
        feat_l4 = self.layer4(feat_l3) # stride 32

        return feat_l1, feat_l2, feat_l3, feat_l4


# ==============================================================================
# Main Network: Net
# ==============================================================================

class Net(nn.Module):
    """
    Two-Stream Fusion Network (Net) for RGB-T Semantic Segmentation.
    """
    def __init__(self, n_class: int = 2):
        super().__init__()

        # 1. Dual Encoders
        self.rgb_encoder = ResNet50Encoder()
        self.tir_encoder = ResNet50Encoder()

        # 2. Channel Reduction Layers (1x1 Conv)
        # Reduces the channel dimensions of fused features to a uniform 512 size
        self.reduce_l1 = ConvBnRelu1x1(256, 512)
        self.reduce_l2 = ConvBnRelu1x1(512, 512)
        self.reduce_l3 = ConvBnRelu1x1(1024, 512)
        self.reduce_l4 = ConvBnRelu1x1(2048, 512)

        # 3. Feature Fusion Layers (3x3 Conv)
        # Refines features after addition and reduction
        self.fuse_l1 = ConvBnRelu3x3(512, 512)
        self.fuse_l2 = ConvBnRelu3x3(512, 512)
        self.fuse_l3 = ConvBnRelu3x3(512, 512)
        self.fuse_l4 = ConvBnRelu3x3(512, 512)

        # 4. Aggregation Decoder
        # Post-concatenation processing. Input channels = 512 * 4 = 2048
        self.decode_fusion = ConvBnRelu1x1(2048, 512)

        # 5. Classification Head
        self.head = nn.Conv2d(512, n_class, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RGB-T fusion.

        Args:
            rgb (torch.Tensor): RGB image [B, 3, H, W]
            tir (torch.Tensor): Thermal image [B, 1, H, W] or [B, 3, H, W]

        Returns:
            torch.Tensor: Segmentation logits [B, n_class, H, W]
        """
        # --- Pre-processing: Ensure Thermal is 3-channel for ResNet ---
        if tir.shape[1] == 1:
            tir = torch.cat([tir, tir, tir], dim=1)

        # --- Step 1: Feature Extraction (Dual Branch) ---
        # r_x: features from RGB stream, t_x: features from Thermal stream
        r1, r2, r3, r4 = self.rgb_encoder(rgb)
        t1, t2, t3, t4 = self.tir_encoder(tir)

        # --- Step 2: Additive Fusion & Channel Alignment ---
        # Fusion Strategy: Element-wise Sum -> 1x1 Reduce -> 3x3 Refine
        
        # Layer 1 (1/4 scale)
        f1 = self.fuse_l1(self.reduce_l1(r1 + t1))
        
        # Layer 2 (1/8 scale)
        f2 = self.fuse_l2(self.reduce_l2(r2 + t2))
        
        # Layer 3 (1/16 scale)
        f3 = self.fuse_l3(self.reduce_l3(r3 + t3))
        
        # Layer 4 (1/32 scale)
        f4 = self.fuse_l4(self.reduce_l4(r4 + t4))

        # --- Step 3: Multi-Scale Aggregation ---
        # Upsample all deeper features to the resolution of Layer 1 (1/4 scale)
        f2_up = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        f3_up = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f4_up = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)

        # Concatenate features: [B, 2048, H/4, W/4]
        fusion_stack = torch.cat([f1, f2_up, f3_up, f4_up], dim=1)
        
        # Refine fused volume
        fusion_out = self.decode_fusion(fusion_stack)

        # --- Step 4: Classification & Final Upsampling ---
        out = self.head(fusion_out)
        
        # Upsample from 1/4 to original resolution (4x)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out


# ==============================================================================
# Integrity Check
# ==============================================================================
if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Net on device: {device}")

    # Create dummy inputs
    batch_size = 2
    height, width = 256, 256
    num_classes = 2

    dummy_rgb = torch.randn(batch_size, 3, height, width).to(device)
    dummy_tir = torch.randn(batch_size, 1, height, width).to(device) # Testing 1-channel input handling

    # Initialize Model
    model = Net(n_class=num_classes).to(device)
    
    # Forward Pass
    with torch.no_grad():
        output = model(dummy_rgb, dummy_tir)

    # Verification
    print(f"Input RGB Shape: {dummy_rgb.shape}")
    print(f"Input TIR Shape: {dummy_tir.shape}")
    print(f"Output Shape:    {output.shape}")

    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, "Output shape mismatch!"
    print("Test Passed: Model forward pass is successful.")