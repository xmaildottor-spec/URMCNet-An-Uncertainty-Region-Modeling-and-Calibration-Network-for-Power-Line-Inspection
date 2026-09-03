import warnings
warnings.filterwarnings("ignore")
import torch 
from torch import nn
import torch.nn.functional as F

class ConvBnrelu2d_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
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
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class OppositeValue(nn.Module):
    """Tensor negation operation R(*)"""
    def forward(self, x):
        return -x

# ==================== Basic CBAM Attention Components ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Ensure the number of intermediate channels is at least 1
        inner_channels = max(in_planes // ratio, 1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, inner_channels, 1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

# ==================== Simplified Spatial and Channel Attention ====================
class SSA(nn.Module):
    """Simplified Spatial Attention (SSA) Module"""
    def __init__(self, in_channels):
        super(SSA, self).__init__()
        self.conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, a, b):
        ab = torch.cat((a, b), dim=1)
        ab_w = self.sigmoid(self.conv1(ab))
        return (a * ab_w) + (b * ab_w) 

class SCA(nn.Module):
    """Simplified Channel Attention (SCA) Module - Equation (13)"""
    def __init__(self, in_channels):
        super(SCA, self).__init__()
        # Utilize Global Max Pooling (GMP)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, a):
        x = self.gmp(a)
        return self.mlp(x)

class HFEB(nn.Module):
    """High Frequency Enhancement Block (HFEB) - Equations (9) and (10)"""
    def __init__(self):
        super(HFEB, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(x - self.gap(x))
        return w * x

# ==================== Core Modules: FPRSM and FNRCM ====================
class FPRSM(nn.Module):
    """False-Positive Region Suppression Module"""
    def __init__(self, in_channels):
        super(FPRSM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.ssa = SSA(in_channels=in_channels)
        self.neg = OppositeValue()
        # Equation (8): Concatenate three branches, then reduce dimensions using 1x1 convolution
        self.cbr_out = ConvBnrelu2d_1(in_channels=3 * in_channels, out_channels=in_channels)
        
    def forward(self, m, fp):
        # 1. Extract high-frequency FP information
        fp_h = fp - self.gap(fp)
        w_fp_h = self.sigmoid(fp_h)
        fp_en = w_fp_h * fp
        
        # 2. Branch a: Suppress FP regions
        m_a = m - fp_en
        
        # 3. Branch b: Enhance non-false-positive regions
        fp_r = self.neg(fp_h)
        w_fp_r = self.sigmoid(fp_r)
        m_b = w_fp_r * m
        
        # 4. Branch c: SSA enhancement
        m_c = self.ssa(m, fp_r)
        
        # 5. Feature fusion
        out = torch.cat([m_a, m_b, m_c], dim=1)
        out = self.cbr_out(out)
        return out 

class FNRCM(nn.Module):
    """False-Negative Region Compensation Module"""
    def __init__(self, in_channels):
        super(FNRCM, self).__init__()
        self.hfeb = HFEB()
        self.sigmoid = nn.Sigmoid()
        self.cbr_m = ConvBnrelu2d_3(in_channels=in_channels, out_channels=in_channels)
        self.sca = SCA(in_channels=in_channels)
        # Final CBR for output, Equation (14)
        self.cbr_out = ConvBnrelu2d_3(in_channels=in_channels, out_channels=in_channels)
        
    def forward(self, m, fn):
        # 1. Enhance high-frequency FN regions
        fn_en = self.hfeb(fn)
        
        # 2. Generate compensation weights and apply to primary features
        m_weight = self.sigmoid(m + fn_en)
        m_compensated = self.cbr_m((m_weight * m) + fn)
        
        # 3. Apply SCA channel weighting and output
        fn_c_weight = self.sca(fn)
        out = self.cbr_out(fn_c_weight * m_compensated)
        return out 

# ==================== Primary Feature Calibration Mechanism (PFCM) ====================
class PFCM(nn.Module):
    """Primary Feature Calibration Mechanism (Includes MA Module logic)"""
    def __init__(self, in_channels):
        super(PFCM, self).__init__()

        self.fprsm = FPRSM(in_channels)
        self.fnrcm = FNRCM(in_channels)

        self.ca_fp = ChannelAttention(in_planes=in_channels)
        self.sa_fp = SpatialAttention()

        self.ca_fn = ChannelAttention(in_planes=in_channels)
        self.sa_fn = SpatialAttention()

        self.cbr_fp = ConvBnrelu2d_3(in_channels=in_channels, out_channels=in_channels)
        self.cbr_fn = ConvBnrelu2d_3(in_channels=in_channels, out_channels=in_channels)

        # Equation (17): Final fusion using Concat followed by 1x1 Conv
        self.fuse = ConvBnrelu2d_1(in_channels=2 * in_channels, out_channels=in_channels)
       
    def forward(self, m, fp, fn):
        # 1. Obtain suppressed and compensated features
        f_fp = self.fprsm(m, fp) 
        f_fn = self.fnrcm(m, fn) 

        # 2. Extract respective channel and spatial attention weights
        ca_weight_fp = self.ca_fp(f_fp) 
        sa_weight_fp = self.sa_fp(f_fp) 

        ca_weight_fn = self.ca_fn(f_fn) 
        sa_weight_fn = self.sa_fn(f_fn)

        # 3. Multi-dimensional Cross-Attention Fusion
        # Top branch: FN combines its own CA and FP's SA
        fn_branch = self.cbr_fn(f_fn * ca_weight_fn)
        fn_att = (fn_branch * sa_weight_fp) + f_fn
        
        # Bottom branch: FP combines its own CA and FN's SA
        fp_branch = self.cbr_fp(f_fp * ca_weight_fp)
        fp_att = (fp_branch * sa_weight_fn) + f_fp

        # 4. Concatenate and reduce dimensions for output
        out = self.fuse(torch.cat([fp_att, fn_att], dim=1))
        return out 

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__ == '__main__':
    
    # Construct simulated input features: Batch=2, Channel=64, H=256, W=256
    # Note: In the same network layer, the dimensions of the primary feature (m), 
    # false-positive feature (fp), and false-negative feature (fn) are identical.
    input_m  = torch.rand(2, 64, 256, 256)
    input_fp = torch.rand(2, 64, 256, 256)
    input_fn = torch.rand(2, 64, 256, 256)
    
    # PFCM initialization only requires in_channels
    test_net = PFCM(in_channels=64)

    final_out = test_net(input_m, input_fp, input_fn)
    print("Calibrated feature output shape (Expected: [2, 64, 256, 256]):", final_out.size())