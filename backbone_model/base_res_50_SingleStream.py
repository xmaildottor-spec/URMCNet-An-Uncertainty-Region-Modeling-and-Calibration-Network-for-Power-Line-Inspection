"""
Single-Stream TFNet (ResNet50 Backbone)
=======================================
Usage:
    model = SingleStreamNet(n_class=2)
    output = model(rgb_image)
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
    Used for channel projection (reducing or aligning feature dimensions).
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
    Extracts multi-scale features from an RGB input.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Use modern torchvision weights API
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # Initial Stem (Input -> 1/4 Scale)
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
        Returns features from 4 stages of ResNet.
        """
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.stem_maxpool(x)

        c2 = self.layer1(x)  # Stride 4
        c3 = self.layer2(c2) # Stride 8
        c4 = self.layer3(c3) # Stride 16
        c5 = self.layer4(c4) # Stride 32

        return c2, c3, c4, c5


# ==============================================================================
# Main Network: Single Stream TFNet
# ==============================================================================

class SingleStreamNet(nn.Module):
    """
    Single-Modality Baseline Network (RGB Only).
    """
    def __init__(self, n_class: int = 2):
        super().__init__()

        # 1. Encoder (RGB only)
        self.rgb_encoder = ResNet50Encoder(pretrained=True)

        # 2. Channel Projection (Reduce)
        # Projects variable channel sizes (256-2048) to a fixed 512 size.
        self.reduce_l1 = ConvBnRelu1x1(256, 512)
        self.reduce_l2 = ConvBnRelu1x1(512, 512)
        self.reduce_l3 = ConvBnRelu1x1(1024, 512)
        self.reduce_l4 = ConvBnRelu1x1(2048, 512)

        # 3. Feature Refinement (Fuse)
        # Smooths the features. Note: In the dual-stream version, this fuses RGB+T.
        # Here, it acts as a non-linear refinement layer for the single stream.
        self.refine_l1 = ConvBnRelu3x3(512, 512)
        self.refine_l2 = ConvBnRelu3x3(512, 512)
        self.refine_l3 = ConvBnRelu3x3(512, 512)
        self.refine_l4 = ConvBnRelu3x3(512, 512)

        # 4. Aggregation Decoder
        # Compresses the concatenated features (512*4 = 2048) back to 512.
        self.decode_fusion = ConvBnRelu1x1(2048, 512)

        # 5. Classification Head
        self.head = nn.Conv2d(512, n_class, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RGB image.
        
        Args:
            rgb (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Logits [B, n_class, H, W]
        """
        # --- Step 1: Feature Extraction ---
        # c2 (1/4), c3 (1/8), c4 (1/16), c5 (1/32)
        r2, r3, r4, r5 = self.rgb_encoder(rgb)

        # --- Step 2: Projection & Refinement ---
        f2 = self.refine_l1(self.reduce_l1(r2))
        f3 = self.refine_l2(self.reduce_l2(r3))
        f4 = self.refine_l3(self.reduce_l3(r4))
        f5 = self.refine_l4(self.reduce_l4(r5))

        # --- Step 3: Multi-Scale Aggregation ---
        # Upsample all deeper features to the resolution of f2 (1/4 scale)
        f3_up = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)
        f4_up = F.interpolate(f4, scale_factor=4, mode='bilinear', align_corners=True)
        f5_up = F.interpolate(f5, scale_factor=8, mode='bilinear', align_corners=True)

        # Concatenate: [B, 2048, H/4, W/4]
        fusion_stack = torch.cat([f2, f3_up, f4_up, f5_up], dim=1)
        
        # Decode
        fusion_out = self.decode_fusion(fusion_stack)

        # --- Step 4: Final Prediction ---
        out = self.head(fusion_out)
        
        # Upsample to original image resolution (x4)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out


# ==============================================================================
# Integrity Check
# ==============================================================================
if __name__ == "__main__":
    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SingleStreamNet on device: {device}")

    # Dummy input
    batch_size = 2
    height, width = 480, 640
    num_classes = 2
    dummy_input = torch.randn(batch_size, 3, height, width).to(device)

    # Init model
    model = SingleStreamNet(n_class=num_classes).to(device)

    # Forward pass
    try:
        output = model(dummy_input)
        print("\nTest Passed Successfully.")
        print(f"Input Shape:  {dummy_input.shape}")
        print(f"Output Shape: {output.shape}")
        
        expected_shape = (batch_size, num_classes, height, width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
    except Exception as e:
        print(f"\nTest Failed: {e}")