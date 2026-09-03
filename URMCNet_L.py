import warnings
warnings.filterwarnings("ignore")
from builtins import print
import torch 
from torch import nn
import torch.nn.functional as F

# Please ensure the corresponding tool modules support 4-level feature inputs (32, 32, 64, 128)
from <YOUR_MODULE_PATH>.HFAFD import HFAFD
from <YOUR_MODULE_PATH>.decoder_star import Decoder_base
from <YOUR_MODULE_PATH>.PFCM import PFCM
from <YOUR_MODULE_PATH>.starnet import starnet_s2 

# --------------------- Utility Functions ---------------------
class ConvBnrelu2d_3(nn.Module):
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

class OppositeValue(nn.Module):
    """
    Numerical Inversion Operation.
    Corresponds to R(*) in the theoretical derivation and the '(R) Opposite value' 
    module in the proposed network architecture diagram.
    """
    def forward(self, x):
        return -x

# --------------------- Main Feature Extractor (Encoder_main) ---------------------
class Encoder_main(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder_main, self).__init__()
        
        self.starnet = starnet_s2(pretrained=pretrained)
        self.encoder0 = self.starnet.stem          # stride = 2  (32)
        self.encoder1 = self.starnet.stages[0]     # stride = 4  (32)
        self.encoder2 = self.starnet.stages[1]     # stride = 8  (64)
        self.encoder3 = self.starnet.stages[2]     # stride = 16 (128)

        # Release memory for unused layers (Remove stage 4 to comply with the 4-layer multi-scale configuration)
        del self.starnet.stages[3]

        # Strictly follow the Conv -> CNNBlock sequence; Conv channels must align with StarNet input channels
        self.conv0 = ConvBnrelu2d_3(in_channels=3, out_channels=3)       # Image
        self.conv1 = ConvBnrelu2d_3(in_channels=32, out_channels=32)     # encoder1 input
        self.conv2 = ConvBnrelu2d_3(in_channels=32, out_channels=32)     # encoder2 input
        self.conv3 = ConvBnrelu2d_3(in_channels=64, out_channels=64)     # encoder3 input

    def forward(self, input_feature):
        # --- Stage 0 ---
        feat0 = self.conv0(input_feature) 
        out0 = self.encoder0(feat0)
       
        # --- Stage 1 ---
        feat1 = self.conv1(out0) 
        out1 = self.encoder1(feat1)
      
        # --- Stage 2 ---
        feat2 = self.conv2(out1)
        out2 = self.encoder2(feat2)
     
        # --- Stage 3 ---
        feat3 = self.conv3(out2)
        out3 = self.encoder3(feat3)
      
        return out0, out1, out2, out3

# --------------------- False Positive Extractor (Encoder_FP) ---------------------
class Encoder_FP(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder_FP,self).__init__()

        self.starnet = starnet_s2(pretrained=pretrained)
        
        # [Core Modification] Change stride=2 to stride=1
        # The input 'last_feature' is already H/2 in size, so no further downsampling 
        # is needed to ensure spatial alignment with m0.
        self.stem_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.stem_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.encoder0 = nn.Sequential(
            self.stem_conv, 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )
        nn.init.constant_(self.encoder0[1].weight, 1)
        nn.init.constant_(self.encoder0[1].bias, 0)
        
        self.encoder1 = self.starnet.stages[0]     
        self.encoder2 = self.starnet.stages[1]     
        self.encoder3 = self.starnet.stages[2]     
        
        del self.starnet.stem
        del self.starnet.stages[3]

        self.conv0 = ConvBnrelu2d_3(in_channels=32, out_channels=32)
        self.conv1 = ConvBnrelu2d_3(in_channels=32+32, out_channels=32)
        self.conv2 = ConvBnrelu2d_3(in_channels=32+32, out_channels=32)
        self.conv3 = ConvBnrelu2d_3(in_channels=64+64, out_channels=64)

        self.neg = OppositeValue()
        self.neg0 = OppositeValue()
        self.neg1 = OppositeValue()
        self.neg2 = OppositeValue()

    def forward(self, input_feature, m0, m1, m2):
        input_feature = self.neg(input_feature)
        m0 = self.neg0(m0) 
        m1 = self.neg1(m1) 
        m2 = self.neg2(m2) 

        # --- Stage 0 ---
        c0 = self.conv0(input_feature) 
        out0 = self.encoder0(c0)

        # --- Stage 1 ---
        c1 = self.conv1(torch.cat([out0, m0], dim=1))
        out1 = self.encoder1(c1)

        # --- Stage 2 ---
        c2 = self.conv2(torch.cat([out1, m1], dim=1))
        out2 = self.encoder2(c2)

        # --- Stage 3 ---
        c3 = self.conv3(torch.cat([out2, m2], dim=1))
        out3 = self.encoder3(c3)

        return out0, out1, out2, out3

# --------------------- False Negative Extractor (Encoder_FN) ---------------------
class Encoder_FN(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder_FN,self).__init__()

        self.starnet = starnet_s2(pretrained=pretrained)
        
        # [Core Modification] Similarly, change stride=1 to prevent resolution mismatch with the backbone
        self.stem_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.stem_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.encoder0 = nn.Sequential(
            self.stem_conv, 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )
        nn.init.constant_(self.encoder0[1].weight, 1)
        nn.init.constant_(self.encoder0[1].bias, 0)
        
        self.encoder1 = self.starnet.stages[0]     
        self.encoder2 = self.starnet.stages[1]     
        self.encoder3 = self.starnet.stages[2]     
        
        del self.starnet.stem
        del self.starnet.stages[3]

        self.conv0 = ConvBnrelu2d_3(in_channels=32, out_channels=32)
        self.conv1 = ConvBnrelu2d_3(in_channels=32+32, out_channels=32)
        self.conv2 = ConvBnrelu2d_3(in_channels=32+32, out_channels=32)
        self.conv3 = ConvBnrelu2d_3(in_channels=64+64, out_channels=64)

    def forward(self, input_feature, m0, m1, m2):
        # --- Stage 0 ---
        c0 = self.conv0(input_feature) 
        out0 = self.encoder0(c0)

        # --- Stage 1 ---
        c1 = self.conv1(torch.cat([out0, m0], dim=1))
        out1 = self.encoder1(c1)

        # --- Stage 2 ---
        c2 = self.conv2(torch.cat([out1, m1], dim=1))
        out2 = self.encoder2(c2)

        # --- Stage 3 ---
        c3 = self.conv3(torch.cat([out2, m2], dim=1))
        out3 = self.encoder3(c3)
      
        return out0, out1, out2, out3

# --------------------- Auxiliary Encoder Manager (Encoder_Aux) ---------------------
class Encoder_Aux(nn.Module):
    def __init__(self, n_class):
        super(Encoder_Aux,self).__init__()

        self.Encoder_FP = Encoder_FP(pretrained=True)
        self.Encoder_FN = Encoder_FN(pretrained=True)
        self.Encoder_main = Encoder_main(pretrained=True)

        self.Decoder_FP = Decoder_base(32, 32, 64, 128, n_class=n_class)
        self.Decoder_FN = Decoder_base(32, 32, 64, 128, n_class=n_class)
        self.Decoder_main = Decoder_base(32, 32, 64, 128, n_class=n_class)

    def forward(self, input_feature):        
     
        m0, m1, m2, m3 = self.Encoder_main(input_feature) 

        pre_main, last_feature = self.Decoder_main(m0, m1, m2, m3)

        fp0, fp1, fp2, fp3 = self.Encoder_FP(last_feature, m0, m1, m2)  
        fn0, fn1, fn2, fn3 = self.Encoder_FN(last_feature, m0, m1, m2)  
     
        pre_fp, _ = self.Decoder_FP(fp0, fp1, fp2, fp3)
        pre_fn, _ = self.Decoder_FN(fn0, fn1, fn2, fn3)

        return pre_main, pre_fp, pre_fn, m0, m1, m2, m3, fp0, fp1, fp2, fp3, fn0, fn1, fn2, fn3

# --------------------- Main Network (URMCNet / TFNet) ---------------------
class TFNet(nn.Module):
    def __init__(self, n_class):
        super(TFNet,self).__init__()

        self.Encoder_Aux = Encoder_Aux(n_class=n_class)

        self.Decoder_final = HFAFD(num_classes=n_class, encoder_channels=[32, 32, 64, 128])

        self.PFCM_0 = PFCM(in_channels=32)
        self.PFCM_1 = PFCM(in_channels=32)
        self.PFCM_2 = PFCM(in_channels=64)
        self.PFCM_3 = PFCM(in_channels=128)
        
    def forward(self, input_feature):        

        pre_main, pre_fp, pre_fn, m0, m1, m2, m3, fp0, fp1, fp2, fp3, fn0, fn1, fn2, fn3 = self.Encoder_Aux(input_feature)             
        
        sam0 = self.PFCM_0(m0, fp0, fn0)
        sam1 = self.PFCM_1(m1, fp1, fn1)
        sam2 = self.PFCM_2(m2, fp2, fn2)
        sam3 = self.PFCM_3(m3, fp3, fn3)

        out_features = [sam0, sam1, sam2, sam3]

        pre_final, pre_aux0, pre_aux1 = self.Decoder_final(out_features)
       
        return pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn

if __name__=='__main__':
    # Forward propagation topology verification test
    input_rgb = torch.rand(2, 3, 256, 256)
    model = TFNet(n_class=1)
    
    pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn = model(input_rgb)
    print("Final prediction output shape:", pre_final.size())
    print("Auxiliary prediction 0 output shape:", pre_aux0.size())
    print("Auxiliary prediction 1 output shape:", pre_aux1.size())
    print("Main branch prediction output shape:", pre_main.size())
    print("FN branch prediction output shape:", pre_fn.size())
    print("FP branch prediction output shape:", pre_fp.size())