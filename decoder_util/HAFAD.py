"""
High-Frequency Aware Fusion Decoder
===========================================

Description:
    This module implements the decoding head for the semantic segmentation network.
    It employs a hierarchical fusion strategy to combine multi-scale features:
    
    1. **Stage 1 (Deep):** Fuses high-level semantic features (F4, F5).
    2. **Stage 2 (Middle):** Integrates mid-level features (F3) with deep features.
    3. **Stage 3 (Shallow):** Recovers fine-grained details using Spatial-Frequency 
       aware modules (HFAM) and Bridge Blocks on shallow features (F1, F2).

    The architecture uses a "U-Shape" connection where deep information acts as 
    context for refining shallow details.

Classes:
    - HFAFD: The complete, proposed decoder architecture (based on HFAFD_star).
    - HFAFD_Res: A residual variant (based on HFAFD_res).

Dependencies:
    - blocks.py (CBR, TBR, BridgeBlock, TransConvBnLeakyRelu2d)
    - hfam.py (HFAM)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom blocks
# Ensure blocks.py and hfam.py are in the same directory or python path
from decoder_blocks import CBR, TBR, BridgeBlock, TransConvBnLeakyRelu2d
from HFAM import HFAM

class HFAFD(nn.Module):
    """
    High-Frequency Aware Fusion Decoder (Standard Version).
    
    This is the primary implementation corresponding to the proposed method.
    It uses Transposed Convolutions (TBR) for learnable upsampling and 
    explicit bi-linear interpolation for feature alignment.
    
    Args:
        num_classes (int): Number of output segmentation classes.
        encoder_channels (list): Channel counts of the backbone features [F1...F5].
    """
    def __init__(self, num_classes=1, encoder_channels=[32, 32, 64, 128, 256]):
        super(HFAFD, self).__init__()
        
        # Unified decoder dimension (complexity control)
        self.dec_dim = 64 
        
        # ======================================================================
        # 1. Feature Projectors (Channel Alignment)
        # ======================================================================
        # Projects all encoder levels (F1-F5) to self.dec_dim
        self.params_c1 = CBR(encoder_channels[0], self.dec_dim, 1, 1, 0)
        self.params_c2 = CBR(encoder_channels[1], self.dec_dim, 1, 1, 0)
        self.params_c3 = CBR(encoder_channels[2], self.dec_dim, 1, 1, 0)
        self.params_c4 = CBR(encoder_channels[3], self.dec_dim, 1, 1, 0)
        self.params_c5 = CBR(encoder_channels[4], self.dec_dim, 1, 1, 0)

        # ======================================================================
        # Stage 1: Deep Semantics (Processing F5 & F4)
        # ======================================================================
        # Upsampler for F5 (Learnable)
        self.tbr_5 = TBR(self.dec_dim, self.dec_dim) 
        
        # Fusion block: Aggregates F5 and F4
        self.fuse_stage1 = CBR(self.dec_dim * 2, self.dec_dim)
        
        # Deep Supervision Head 2
        self.seg_head_aux2 = nn.Conv2d(self.dec_dim, num_classes, 1)

        # ======================================================================
        # Stage 2: Middle Semantics (Processing F3 & Stage 1 Output)
        # ======================================================================
        # Transition/Upsample previous stage output
        self.trans_stage2 = nn.Sequential(
            CBR(self.dec_dim, self.dec_dim),
            TBR(self.dec_dim, self.dec_dim)
        )
        
        # Fusion block: Aggregates Stage 1 and F3
        self.fuse_stage2 = CBR(self.dec_dim * 2, self.dec_dim)
        
        # Deep Supervision Head 1
        self.seg_head_aux1 = nn.Conv2d(self.dec_dim, num_classes, 1)

        # ======================================================================
        # Stage 3: Detail Recovery (Main Branch - Processing F1, F2)
        # ======================================================================
        # Global Context Projector
        self.context_stage3 = CBR(self.dec_dim, self.dec_dim)
        
        # Bridge Blocks: Fuse Encoder F1/F2 with Context
        self.bridge_1 = BridgeBlock(self.dec_dim)
        self.bridge_2 = BridgeBlock(self.dec_dim)
        
        # HFAM Blocks: Hierarchical Feature Attention
        self.hfam_1 = HFAM(self.dec_dim)
        self.hfam_2 = HFAM(self.dec_dim)
        
        # Final Fusion: Aggregates 5 distinct feature paths
        self.fuse_main = CBR(self.dec_dim * 5, self.dec_dim)
        
        # Main Segmentation Head (Final Upsampling)
        self.seg_head_main = nn.Sequential(
            CBR(self.dec_dim, self.dec_dim),
            TransConvBnLeakyRelu2d(self.dec_dim, self.dec_dim),
            nn.Conv2d(self.dec_dim, num_classes, 1)
        )

    def forward(self, features):
        """
        Forward pass of the decoder.
        
        Args:
            features (list): List of tensors [c1, c2, c3, c4, c5] from the backbone.
                             c1 is highest resolution (1/2), c5 is lowest (1/32).
        
        Returns:
            pred_main (Tensor): Final prediction mask.
            pred_aux1 (Tensor): Auxiliary prediction from Stage 2.
            pred_aux2 (Tensor): Auxiliary prediction from Stage 1.
        """
        # Unpack and Project
        c1, c2, c3, c4, c5 = features
        f1 = self.params_c1(c1)
        f2 = self.params_c2(c2)
        f3 = self.params_c3(c3)
        f4 = self.params_c4(c4)
        f5 = self.params_c5(c5)
        
        # ---------------------- Stage 1 ----------------------
        # Path: F5 -> TBR (Upsample)
        f5_tbr = self.tbr_5(f5) 
        
        # Interaction: F4 * Upsample(F5)
        f5_up_bilinear = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=True)
        interaction_45 = f4 * f5_up_bilinear
        
        # Fusion
        cat_1 = torch.cat([f5_up_bilinear, interaction_45], dim=1)
        out_stage1 = self.fuse_stage1(cat_1)
        
        # Aux Output 2
        pred_aux2 = self.seg_head_aux2(out_stage1)
        pred_aux2 = F.interpolate(pred_aux2, scale_factor=16, mode='bilinear', align_corners=True)

        # ---------------------- Stage 2 ----------------------
        # Path: Stage1 -> Upsample
        prev_feat_2 = self.trans_stage2(out_stage1)
        
        # Interaction: F3 * Upsample(F4)
        f4_up = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        interaction_3 = f3 * f4_up
        
        # Fusion
        cat_2 = torch.cat([prev_feat_2, interaction_3], dim=1)
        out_stage2 = self.fuse_stage2(cat_2)
        
        # Aux Output 1
        pred_aux1 = self.seg_head_aux1(out_stage2)
        pred_aux1 = F.interpolate(pred_aux1, scale_factor=8, mode='bilinear', align_corners=True)

        # ---------------------- Stage 3 (Main) ----------------------
        # Context from deep stages
        context_feat = self.context_stage3(out_stage2)
        
        # Branch 1 (F1): Bridge + HFAM
        ctx_up_1 = F.interpolate(context_feat, size=f1.shape[2:], mode='bilinear', align_corners=True)
        b1_out = self.bridge_1(f1, ctx_up_1)
        hfam1_out = self.hfam_1(f1)
        
        # Branch 2 (F2): Bridge + HFAM
        ctx_up_2 = F.interpolate(context_feat, size=f2.shape[2:], mode='bilinear', align_corners=True)
        b2_out = self.bridge_2(f2, ctx_up_2)
        
        hfam2_out = self.hfam_2(f2)
        
        # Align all Stage 3 features to F1 resolution (Highest Scale)
        b2_out_up = F.interpolate(b2_out, size=f1.shape[2:], mode='bilinear', align_corners=True)
        hfam2_up = F.interpolate(hfam2_out, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        
        # Feature Mixer
        mix_12 = f1 * f2_up
        
        # Final Concatenation
        final_cat = torch.cat([hfam1_out, b1_out, b2_out_up, hfam2_up, mix_12], dim=1)
        final_feat = self.fuse_main(final_cat)
        
        # Main Prediction
        pred_main = self.seg_head_main(final_feat)
     
        return pred_main, pred_aux1, pred_aux2


class HFAFD_Res(nn.Module):
    """
    High-Frequency Aware Fusion Decoder (Residual/Lite Variant).
    
    Description:
        A variant of HFAFD that uses standard Convolutions (CBR) instead of 
        Transposed Convolutions for the F5 branch, intended for ablation 
        studies or lighter compute requirements.
    """
    def __init__(self, num_classes=1, encoder_channels=[64, 256, 512, 1024, 2048]):
        super(HFAFD_Res, self).__init__()
        
        self.dec_dim = 64 
        
        # Channel Reducers
        self.params_c1 = CBR(encoder_channels[0], self.dec_dim, 1, 1, 0)
        self.params_c2 = CBR(encoder_channels[1], self.dec_dim, 1, 1, 0)
        self.params_c3 = CBR(encoder_channels[2], self.dec_dim, 1, 1, 0)
        self.params_c4 = CBR(encoder_channels[3], self.dec_dim, 1, 1, 0)
        self.params_c5 = CBR(encoder_channels[4], self.dec_dim, 1, 1, 0)

        # Stage 1: Uses CBR instead of TBR
        self.cbr_5 = CBR(self.dec_dim, self.dec_dim) 
        self.fuse_stage1 = CBR(self.dec_dim * 2, self.dec_dim)
        self.seg_head_aux2 = nn.Conv2d(self.dec_dim, num_classes, 1)

        # Stage 2
        self.trans_stage2 = nn.Sequential(
            CBR(self.dec_dim, self.dec_dim),
            TBR(self.dec_dim, self.dec_dim)
        )
        self.fuse_stage2 = CBR(self.dec_dim * 2, self.dec_dim)
        self.seg_head_aux1 = nn.Conv2d(self.dec_dim, num_classes, 1)

        # Stage 3
        self.context_stage3 = CBR(self.dec_dim, self.dec_dim)
        self.bridge_1 = BridgeBlock(self.dec_dim)
        self.bridge_2 = BridgeBlock(self.dec_dim)
        self.hfam_1 = HFAM(self.dec_dim)
        self.hfam_2 = HFAM(self.dec_dim)
        self.fuse_main = CBR(self.dec_dim * 5, self.dec_dim)
        
        self.seg_head_main = nn.Sequential(
            CBR(self.dec_dim, self.dec_dim),
            TransConvBnLeakyRelu2d(self.dec_dim, self.dec_dim),
            nn.Conv2d(self.dec_dim, num_classes, 1)
        )

    def forward(self, features):
        c1, c2, c3, c4, c5 = features
        f1 = self.params_c1(c1)
        f2 = self.params_c2(c2)
        f3 = self.params_c3(c3)
        f4 = self.params_c4(c4)
        f5 = self.params_c5(c5)
        
        # --- Stage 1 ---
        # Note: In HFAFD_Res, we assume implicit interaction logic 
        # but ensure safety with interpolation
        
        # Path: F5 -> CBR
        f5_processed = self.cbr_5(f5) 
        
        # Safety: Interpolate F5 to match F4 before interaction
        f5_up = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=True)
        f5_proc_up = F.interpolate(f5_processed, size=f4.shape[2:], mode='bilinear', align_corners=True)
        
        interaction_45 = f4 * f5_up
        
        cat_1 = torch.cat([f5_proc_up, interaction_45], dim=1)
        out_stage1 = self.fuse_stage1(cat_1)
        
        pred_aux2 = self.seg_head_aux2(out_stage1)
        pred_aux2 = F.interpolate(pred_aux2, scale_factor=16, mode='bilinear', align_corners=True)

        # --- Stage 2 ---
        prev_feat_2 = self.trans_stage2(out_stage1)
        f4_up = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        interaction_3 = f3 * f4_up
        cat_2 = torch.cat([prev_feat_2, interaction_3], dim=1)
        out_stage2 = self.fuse_stage2(cat_2)
        
        pred_aux1 = self.seg_head_aux1(out_stage2)
        pred_aux1 = F.interpolate(pred_aux1, scale_factor=8, mode='bilinear', align_corners=True)

        # --- Stage 3 ---
        context_feat = self.context_stage3(out_stage2)
        
        ctx_up_1 = F.interpolate(context_feat, size=f1.shape[2:], mode='bilinear', align_corners=True)
        b1_out = self.bridge_1(f1, ctx_up_1)
        hfam1_out = self.hfam_1(f1)
        
        ctx_up_2 = F.interpolate(context_feat, size=f2.shape[2:], mode='bilinear', align_corners=True)
        b2_out = self.bridge_2(f2, ctx_up_2)
        hfam2_out = self.hfam_2(f2)
        
        # Align to F1
        b2_out_up = F.interpolate(b2_out, size=f1.shape[2:], mode='bilinear', align_corners=True)
        hfam2_up = F.interpolate(hfam2_out, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        
        mix_12 = f1 * f2_up
        
        final_cat = torch.cat([hfam1_out, b1_out, b2_out_up, hfam2_up, mix_12], dim=1)
        final_feat = self.fuse_main(final_cat)
        
        pred_main = self.seg_head_main(final_feat)
     
        return pred_main, pred_aux1, pred_aux2


if __name__ == "__main__":
    # Integration Test
    print("Testing HFAFD Module...")
    
    # Dummy inputs representing encoder features [1/2 ... 1/32]
    # Channels: [32, 32, 64, 128, 256]
    img_size = 256
    x1 = torch.randn(2, 32, img_size//2, img_size//2)
    x2 = torch.randn(2, 32, img_size//4, img_size//4)
    x3 = torch.randn(2, 64, img_size//8, img_size//8)
    x4 = torch.randn(2, 128, img_size//16, img_size//16)
    x5 = torch.randn(2, 256, img_size//32, img_size//32)
    
    features = [x1, x2, x3, x4, x5]
    
    # 1. Test Standard HFAFD
    model_std = HFAFD(num_classes=1, encoder_channels=[32, 32, 64, 128, 256])
    p1, a1, a2 = model_std(features)
    print(f"[HFAFD] Main: {p1.shape}, Aux1: {a1.shape}, Aux2: {a2.shape}")
    
    # 2. Test Residual Variant
    model_res = HFAFD_Res(num_classes=1, encoder_channels=[32, 32, 64, 128, 256])
    p1, a1, a2 = model_res(features)
    print(f"[HFAFD_Res] Main: {p1.shape}, Aux1: {a1.shape}, Aux2: {a2.shape}")