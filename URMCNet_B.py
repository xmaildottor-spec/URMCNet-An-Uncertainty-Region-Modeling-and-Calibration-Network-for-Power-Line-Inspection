import warnings
warnings.filterwarnings("ignore")
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models 

# ------------------------------------------------------------------------------
# Placeholder paths: Update '<YOUR_MODULE_PATH>' with your actual directory structure 
# before deploying this repository for multi-modal segmentation tasks.
# Ensure the corresponding tool modules support 4-level feature inputs (64, 256, 512, 1024).
# ------------------------------------------------------------------------------
from <YOUR_MODULE_PATH>.HFAFD import HFAFD
from <YOUR_MODULE_PATH>.decoder_res import Decoder_base
from <YOUR_MODULE_PATH>.PFCM import PFCM

# --------------------- Utility Modules ---------------------
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
    def __init__(self):
        super(Encoder_main, self).__init__()
        
        resnet1 = models.resnet50(pretrained=True)
        
        # --- CNNBlocks Definition ---
        self.encoder0 = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)

        # Replace MaxPool with a stride-2 convolution to preserve high-frequency fine-grained information
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace=True)
        self.encoder1 = nn.Sequential(
            self.layer1_conv, 
            self.layer1_bn, 
            self.layer1_relu, 
            resnet1.layer1)
        
        for m in self.encoder0.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3

        # Release memory for unused layers (retaining strictly the 4-layer structure)
        del resnet1.layer4
        del resnet1.avgpool
        del resnet1.fc

        # --- Conv Module Definition (Dimensions adjusted to fit the preceding structure) ---
        # Strictly follow the Conv -> CNNBlock sequence; Conv channels must align with CNNBlock input channels
        self.conv0 = ConvBnrelu2d_3(in_channels=3, out_channels=3)       # Corresponds to encoder0 input (Image)
        self.conv1 = ConvBnrelu2d_3(in_channels=64, out_channels=64)     # Corresponds to encoder1 input
        self.conv2 = ConvBnrelu2d_3(in_channels=256, out_channels=256)   # Corresponds to encoder2 input
        self.conv3 = ConvBnrelu2d_3(in_channels=512, out_channels=512)   # Corresponds to encoder3 input

    def forward(self, input_feature):
        # --- Stage 0 ---
        # Logical alignment: Image -> Conv -> CNNBlock -> F_pr^1
        feat0 = self.conv0(input_feature) 
        out0 = self.encoder0(feat0)       # Output dimension: 64
       
        # --- Stage 1 ---
        # Logical alignment: F_pr^1 -> Conv -> CNNBlock -> F_pr^2
        feat1 = self.conv1(out0) 
        out1 = self.encoder1(feat1)       # Output dimension: 256
      
        # --- Stage 2 ---
        # Logical alignment: F_pr^2 -> Conv -> CNNBlock -> F_pr^3
        feat2 = self.conv2(out1)
        out2 = self.encoder2(feat2)       # Output dimension: 512
     
        # --- Stage 3 ---
        # Logical alignment: F_pr^3 -> Conv -> CNNBlock -> F_pr^4
        feat3 = self.conv3(out2)
        out3 = self.encoder3(feat3)       # Output dimension: 1024
      
        # Final output of multi-level main features (F_pr^1 to F_pr^4)
        # Channels (64, 256, 512, 1024) maintain perfect compatibility
        return out0, out1, out2, out3

# --------------------- False Positive Extractor (Encoder_FP) ---------------------
class Encoder_FP(nn.Module):
    def __init__(self):
        super(Encoder_FP,self).__init__()

        resnet1 = models.resnet50(pretrained=True)    
        
        # [Core Modification] Change stride=2 to stride=1
        # Since the input 'last_feature' is already H/2 in size, no further downsampling 
        # is needed to ensure spatial alignment with m0.
        self.stem_conv = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False)
        nn.init.kaiming_normal_(self.stem_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.encoder0 = nn.Sequential(
            self.stem_conv, 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        nn.init.constant_(self.encoder0[1].weight, 1)
        nn.init.constant_(self.encoder0[1].bias, 0)
        
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace=True)
        self.encoder1 = nn.Sequential(
            self.layer1_conv, 
            self.layer1_bn, 
            self.layer1_relu, 
            resnet1.layer1)

        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        
        del resnet1.layer4
        del resnet1.avgpool
        del resnet1.fc
        del resnet1.conv1
        del resnet1.bn1

        self.conv0 = ConvBnrelu2d_3(in_channels=64, out_channels=64)       
        self.conv1 = ConvBnrelu2d_3(in_channels=64+64, out_channels=64)    
        self.conv2 = ConvBnrelu2d_3(in_channels=256+256, out_channels=256) 
        self.conv3 = ConvBnrelu2d_3(in_channels=512+512, out_channels=512) 

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
    def __init__(self):
        super(Encoder_FN,self).__init__()

        resnet1 = models.resnet50(pretrained=True)
        
        # [Core Modification] Similarly, change stride=1 to prevent resolution mismatch with the backbone
        self.stem_conv = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False)
        nn.init.kaiming_normal_(self.stem_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.encoder0 = nn.Sequential(
            self.stem_conv, 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        nn.init.constant_(self.encoder0[1].weight, 1)
        nn.init.constant_(self.encoder0[1].bias, 0)
        
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace=True)
        self.encoder1 = nn.Sequential(
            self.layer1_conv, 
            self.layer1_bn, 
            self.layer1_relu, 
            resnet1.layer1)

        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        
        del resnet1.layer4
        del resnet1.avgpool
        del resnet1.fc
        del resnet1.conv1
        del resnet1.bn1

        self.conv0 = ConvBnrelu2d_3(in_channels=64, out_channels=64)
        self.conv1 = ConvBnrelu2d_3(in_channels=64+64, out_channels=64)
        self.conv2 = ConvBnrelu2d_3(in_channels=256+256, out_channels=256)
        self.conv3 = ConvBnrelu2d_3(in_channels=512+512, out_channels=512)

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

        self.Encoder_FP = Encoder_FP()
        self.Encoder_FN = Encoder_FN()
        self.Encoder_main = Encoder_main()

        self.Decoder_FP = Decoder_base(64, 256, 512, 1024, n_class=n_class)
        self.Decoder_FN = Decoder_base(64, 256, 512, 1024, n_class=n_class)
        self.Decoder_main = Decoder_base(64, 256, 512, 1024, n_class=n_class)

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

        self.Decoder_final = HFAFD(num_classes=n_class, encoder_channels=[64, 256, 512, 1024])

        self.PFCM_0 = PFCM(in_channels=64)
        self.PFCM_1 = PFCM(in_channels=256)
        self.PFCM_2 = PFCM(in_channels=512)
        self.PFCM_3 = PFCM(in_channels=1024)
        
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
    # Architecture topology verification test
    input_rgb = torch.rand(2, 3, 256, 256)
    model = TFNet(n_class=1)
    
    pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn = model(input_rgb)
    print("Final prediction output shape (HFAFD Output):", pre_final.size())
    print("Main branch primary prediction output shape:", pre_main.size())
    print("FN branch prediction output shape:", pre_fn.size())
    print("FP branch prediction output shape:", pre_fp.size())