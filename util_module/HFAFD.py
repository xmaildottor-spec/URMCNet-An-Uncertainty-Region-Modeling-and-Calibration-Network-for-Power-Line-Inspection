import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    """Standard Conv-BN-ReLU block with Dilation Support"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TransConvBnLeakyRelu2d(nn.Module):
    """Transposed Convolution for final segmentation head (stride=2)"""
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

class TBR(nn.Module):
    """Transposed Conv - BN - ReLU (Used for upsampling features, stride=2)"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(TBR, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv(x)

class GeneralizedMeanPooling(nn.Module):
    """GEP: Generalized Mean Pooling (Adaptive to Max or Avg based on parameter p)"""
    def __init__(self, output_size=1, p=3):
        super(GeneralizedMeanPooling, self).__init__()
        self.output_size = output_size
        self.p = p

    def forward(self, x):
        x = x.clamp(min=1e-6)
        return F.adaptive_avg_pool2d(x.pow(self.p), self.output_size).pow(1. / self.p)

def carafe(x, normed_mask, kernel_size, group=1, up=1):
    """CARAFE Operator: Corresponds to the (*) Convolve operation in the network architecture"""
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    
    assert m_h == up * h
    assert m_w == up * w
    
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    return res

class AHPFGenerator(nn.Module):
    """High-Pass Filter Generator (AHPF)"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        
        self.content_encoder = nn.Conv2d(channels, kernel_size ** 2, kernel_size=3, padding=1, groups=1)
        self.register_buffer('hamming', torch.ones(1, 1, kernel_size, kernel_size))

    def kernel_normalizer(self, mask):
        n, mask_c, h, w = mask.size()
        mask = mask.view(n, 1, -1, h, w)
        mask = F.softmax(mask, dim=2)
        mask = mask.view(n, 1, self.kernel_size, self.kernel_size, h, w)
        mask = mask * self.hamming.view(1, 1, self.kernel_size, self.kernel_size, 1, 1)
        mask = mask.view(n, 1, -1, h, w)
        mask /= mask.sum(dim=2, keepdim=True)
        mask = mask.view(n, -1, h, w)
        return mask

    def forward(self, x):
        mask_raw = self.content_encoder(x)
        mask_lp = self.kernel_normalizer(mask_raw)
        
        feat_lp = carafe(x, mask_lp, self.kernel_size)
        feat_hp = x - feat_lp
        
        return feat_hp

class HFAM(nn.Module):
    """High-Frequency Awareness Module (HFAM)"""
    def __init__(self, in_channels, reduction=16):
        super(HFAM, self).__init__()
        self.conv1x1 = CBR(in_channels, in_channels, 1, 1, 0)
        self.AHPFGenerator = AHPFGenerator(channels=in_channels, kernel_size=3)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gep = GeneralizedMeanPooling(1)
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv1x1(x)
        
        spatial_out = self.AHPFGenerator(feat)
        spatial_out = feat + spatial_out

        b, c, _, _ = feat.size()
        y_gap = self.gap(feat)
        y_gep = self.gep(feat)
        
        w_gap = self.mlp(y_gap)
        w_gep = self.mlp(y_gep)
        weights = self.sigmoid(w_gap + w_gep).view(b, c, 1, 1)
        
        out = spatial_out * weights
        return out

class BridgeBlock(nn.Module):
    """Bridge Block for feature interaction"""
    def __init__(self, in_channels):
        super(BridgeBlock, self).__init__()
        self.cbr = CBR(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f_en, f_d):
        prod = f_en * f_d
        diff = f_en - f_d
        summ = prod + diff
        return self.cbr(summ)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP)"""
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.aspp2 = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.aspp3 = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.aspp4 = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.out_conv = CBR(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out_conv(x_cat)

class HFAFD(nn.Module):
    
    def __init__(self, num_classes=1, encoder_channels=[32, 32, 64, 128]):
        super(HFAFD, self).__init__()
        
        self.dec_dim = 64 
        
        # Unified dimension mapping
        self.params_c1 = CBR(encoder_channels[0], self.dec_dim, 1, 1, 0)
        self.params_c2 = CBR(encoder_channels[1], self.dec_dim, 1, 1, 0)
        self.params_c3 = CBR(encoder_channels[2], self.dec_dim, 1, 1, 0)
        self.params_c4 = CBR(encoder_channels[3], self.dec_dim, 1, 1, 0)

        # Generate Stage 5 features: Downsample and apply ASPP for deep context extraction
        self.down_c4_to_f5 = CBR(encoder_channels[3], self.dec_dim, kernel_size=3, stride=2, padding=1)
        self.aspp_f5 = ASPP(self.dec_dim, self.dec_dim)

        # --- Stage 1 (Right): Processing F5 & F4 ---
        self.tbr_5 = TBR(self.dec_dim, self.dec_dim) # Upsample F5 (H/32 -> H/16)
        self.fuse_stage1 = CBR(self.dec_dim * 2, self.dec_dim)
        self.seg_head_aux2 = nn.Conv2d(self.dec_dim, num_classes, 1)

        # --- Stage 2 (Middle): Processing F_d1 (Stage1 output) & F3 ---
        self.tbr_d1 = TBR(self.dec_dim, self.dec_dim) # Upsample F_d1 (H/16 -> H/8)
        self.fuse_stage2 = CBR(self.dec_dim * 2, self.dec_dim)
        self.seg_head_aux1 = nn.Conv2d(self.dec_dim, num_classes, 1)

        # --- Stage 3 (Left - Main): Processing F_d2, F1, F2 ---
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
        c1, c2, c3, c4 = features
        
        # Extract F1 to F4 features (resolutions: H/2, H/4, H/8, H/16 respectively)
        f1 = self.params_c1(c1)
        f2 = self.params_c2(c2)
        f3 = self.params_c3(c3)
        f4 = self.params_c4(c4)
        
        # Generate F5 feature (resolution: H/32)
        f5 = self.down_c4_to_f5(c4)
        f5 = self.aspp_f5(f5)
        
        # ==================== Stage 1: Equations (18)-(19) ====================
        f5_up = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=True)
        inter_4 = f5_up * f4                             # Corresponds to UP_2(F_en_5) ⊙ F_en_4
        prev_f5 = self.tbr_5(f5)                         # Corresponds to TBR(F_en_5)
        
        out_stage1 = self.fuse_stage1(torch.cat([prev_f5, inter_4], dim=1)) # Output F_d_1 (H/16)
        
        pred_aux2 = self.seg_head_aux2(out_stage1)
        pred_aux2 = F.interpolate(pred_aux2, scale_factor=16, mode='bilinear', align_corners=True)

        # ==================== Stage 2: Equations (20)-(21) ====================
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=True)
        inter_3 = f4_up * f3                             # Corresponds to UP_2(F_en_4) ⊙ F_en_3
        prev_Fd1 = self.tbr_d1(out_stage1)               # Corresponds to TBR(F_d_1)
        
        out_stage2 = self.fuse_stage2(torch.cat([prev_Fd1, inter_3], dim=1)) # Output F_d_2 (H/8)
        
        pred_aux1 = self.seg_head_aux1(out_stage2)
        pred_aux1 = F.interpolate(pred_aux1, scale_factor=8, mode='bilinear', align_corners=True)

        # ==================== Stage 3 (Main): Equations (22)-(34) ====================
        context_feat = self.context_stage3(out_stage2)   # Extract shared semantic features
        
        # Bridge interaction extraction
        ctx_up_1 = F.interpolate(context_feat, size=f1.shape[2:], mode='bilinear', align_corners=True) 
        b1_out = self.bridge_1(f1, ctx_up_1)             # Corresponds to F_com^1
        
        ctx_up_2 = F.interpolate(context_feat, size=f2.shape[2:], mode='bilinear', align_corners=True) 
        b2_out = self.bridge_2(f2, ctx_up_2) 
        b2_out = F.interpolate(b2_out, size=f1.shape[2:], mode='bilinear', align_corners=True) # Corresponds to F_com^2
        
        # High-frequency feature enhancement
        hfam1_out = self.hfam_1(f1)                      # Corresponds to F_out_c_1
        hfam2_out = self.hfam_2(f2) 
        hfam2_up = F.interpolate(hfam2_out, size=f1.shape[2:], mode='bilinear', align_corners=True) # Corresponds to F_out_c_2
        
        # Shallow and deep feature crossover
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True) 
        mix_12 = f1 * f2_up                              # Corresponds to F_share
        
        # Fusion output
        final_cat = torch.cat([hfam1_out, b1_out, b2_out, hfam2_up, mix_12], dim=1)
        final_feat = self.fuse_main(final_cat)           # Output F_fuse (H/2)
        
        pred_main = self.seg_head_main(final_feat)       # Output final predicted segmentation map (H)
     
        return pred_main, pred_aux1, pred_aux2


if __name__ == "__main__":
    img_size = 256
    x1 = torch.randn(2, 32, img_size//2, img_size//2)
    x2 = torch.randn(2, 32, img_size//4, img_size//4)
    x3 = torch.randn(2, 64, img_size//8, img_size//8)
    x4 = torch.randn(2, 128, img_size//16, img_size//16)
    
    features = [x1, x2, x3, x4]
    
    # Instantiate the latest HFAFD model
    model = HFAFD(num_classes=1, encoder_channels=[32, 32, 64, 128])
    
    p_main, p_aux1, p_aux2 = model(features)
    
    print("Main Output Shape:", p_main.shape)  # Expected: [2, 1, 256, 256]
    print("Aux1 Output Shape:", p_aux1.shape)  # Expected: [2, 1, 256, 256]
    print("Aux2 Output Shape:", p_aux2.shape)  # Expected: [2, 1, 256, 256]