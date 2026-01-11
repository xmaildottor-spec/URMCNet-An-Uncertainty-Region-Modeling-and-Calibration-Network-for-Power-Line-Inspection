import torch
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
import torch
from model.segformer import mit_b1
from my_base_block import ConvBnrelu2d_3, ConvBnrelu2d_1

class TFNet(nn.Module):
    def __init__(self, n_class, mode=True):
        super(TFNet, self).__init__()
        self.model1 = mit_b1()
        self.model2 = mit_b1()
        self.training = mode

        state_dict = torch.load(r'xxx.pth', map_location='cpu',weights_only=True)
        
        if self.training:
            self.model1.load_state_dict(state_dict, strict=False)
            self.model1.load_state_dict(state_dict, strict=False)
        self.fuse1 = ConvBnrelu2d_3(64, 64)
        self.fuse2 = ConvBnrelu2d_3(128, 128)
        self.fuse3 = ConvBnrelu2d_3(320, 320)
        self.fuse4 = ConvBnrelu2d_3(512, 512)
        
        self.c1 = ConvBnrelu2d_1(64, 512)
        self.c2 = ConvBnrelu2d_1(128, 512)
        self.c3 = ConvBnrelu2d_1(320, 512)
        
        self.defuse = ConvBnrelu2d_1(2048, 512)
        
        self.fusion_conv = nn.Conv2d(512, n_class, kernel_size=1, padding=0, stride=1,bias=False)
        nn.init.xavier_uniform_(self.fusion_conv.weight.data)
        
    def forward(self, input_RGB,input_T):
        input_T = torch.cat((input_T,input_T,input_T),dim=1)           
        out_rgb = self.model1(input_RGB)
        r1,r2,r3,r4 = out_rgb
        out_t = self.model2(input_T)
        t1,t2,t3,t4 = out_t
        
        m1 = self.fuse1(r1 + t1)
        m2 = self.fuse2(r2 + t2)
        m3 = self.fuse3(r3 + t3)
        m4 = self.fuse4(r4 + t4)

        m1 = self.c1(m1)
        m2 = self.c2(m2)
        m3 = self.c3(m3)
        
        m2 = F.interpolate(m2, scale_factor=2, mode='bilinear',align_corners=True)
        m3 = F.interpolate(m3, scale_factor=4, mode='bilinear',align_corners=True)
        m4 = F.interpolate(m4, scale_factor=8, mode='bilinear',align_corners=True)
        
        fusion = self.defuse(torch.cat([m1, m2, m3, m4], dim=1))
        fusion = self.fusion_conv(fusion)

        return F.interpolate(fusion, scale_factor=4, mode='bilinear',align_corners=True)

if __name__ == "__main__":
    input_rgb = torch.randn(1, 3, 480, 640)
    input_t = torch.randn(1, 3, 480, 640)
    model = TFNet(n_class=2)
    out = model(input_rgb, input_t)
    print("Output shape:", out.shape)

