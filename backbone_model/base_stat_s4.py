import torch
import torch.nn as nn 
import torch.nn.functional as F
from model.starnet import starnet_s4 
from my_base_block import ConvBnrelu2d_3, ConvBnrelu2d_1


class TFNet(nn.Module):
    def __init__(self, n_class, mode=True, pretrained=True):
        super(TFNet, self).__init__()
        self.training = mode

        self.model1 = starnet_s4(pretrained=pretrained)
        self.model2 = starnet_s4(pretrained=pretrained)

        self.fuse1 = ConvBnrelu2d_3(24, 24)
        self.fuse2 = ConvBnrelu2d_3(48, 48)
        self.fuse3 = ConvBnrelu2d_3(96, 96)
        self.fuse4 = ConvBnrelu2d_3(192, 192)

        self.c1 = ConvBnrelu2d_1(24, 512)
        self.c2 = ConvBnrelu2d_1(48, 512)
        self.c3 = ConvBnrelu2d_1(96, 512)
        self.c4 = ConvBnrelu2d_1(192, 512)

        self.defuse = ConvBnrelu2d_1(2048, 512)

        self.fusion_conv = nn.Conv2d(512, n_class, kernel_size=1, padding=0, stride=1, bias=False)
        nn.init.xavier_uniform_(self.fusion_conv.weight.data)

    def forward(self, input_RGB, input_T):
        input_T = torch.cat((input_T, input_T, input_T), dim=1)

        r1, r2, r3, r4 = self.model1(input_RGB)
        t1, t2, t3, t4 = self.model2(input_T)

        print(r1.size())
        print(r2.size())
        print(r3.size())
        print(r4.size())
        exit()

        m1 = self.fuse1(r1 + t1)   
        m2 = self.fuse2(r2 + t2)   
        m3 = self.fuse3(r3 + t3) 
        m4 = self.fuse4(r4 + t4)  

        m1 = self.c1(m1)
        m2 = self.c2(m2)
        m3 = self.c3(m3)
        m4 = self.c4(m4)

        m2 = F.interpolate(m2, scale_factor=2, mode='bilinear', align_corners=True)
        m3 = F.interpolate(m3, scale_factor=4, mode='bilinear', align_corners=True)
        m4 = F.interpolate(m4, scale_factor=8, mode='bilinear', align_corners=True)

        fusion = self.defuse(torch.cat([m1, m2, m3, m4], dim=1))
        fusion = self.fusion_conv(fusion)

        return F.interpolate(fusion, scale_factor=4, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    input_rgb = torch.randn(1, 3, 256, 256)
    input_t = torch.randn(1, 1, 256, 256)
    model = TFNet(n_class=2, mode=True, pretrained=True)
    out = model(input_rgb, input_t)
    print("Output shape:", out.shape)