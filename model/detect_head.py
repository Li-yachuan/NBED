import torch
from torch import nn
import torch.nn.functional as F


class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """

    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


class CSAM_head(nn.Module):
    def __init__(self, channels):
        super(CSAM_head, self).__init__()

        modulelst = []
        for num in channels:
            modulelst.append(nn.Sequential(CSAM(num), nn.Conv2d(num, 1, 1)))

        self.modulelst = nn.ModuleList(modulelst)

        self.final = nn.Sequential(nn.Conv2d(5, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        _, _, H, W = feats[0].size()
        for i in range(len(feats)):
            feats[i] = F.interpolate(self.modulelst[i](feats[i]), (H, W), mode="bilinear")

        return self.final(torch.cat(feats, dim=1))


class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class CDCM_head(nn.Module):
    def __init__(self, channels):
        super(CDCM_head, self).__init__()
        self.channels = channels
        modulelst = []
        for num in channels:
            modulelst.append(
                nn.Sequential(
                    CDCM(num, num),
                    nn.Conv2d(num, 1, 1))
            )

        self.modulelst = nn.ModuleList(modulelst)

        self.final = nn.Sequential(nn.Conv2d(5, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        _, _, H, W = feats[0].size()
        for i in range(len(feats)):
            feats[i] = F.interpolate(self.modulelst[i](feats[i]), (H, W), mode="bilinear")
        return self.final(torch.cat(feats, dim=1))


class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        return ((x * attn).sum(1)).unsqueeze(1)


class CoFusion_head(nn.Module):
    def __init__(self, channels):
        super(CoFusion_head, self).__init__()

        modulelst = []
        for num in channels:
            modulelst.append(nn.Sequential(
                nn.Conv2d(num, 1, 1),
                nn.Sigmoid())
            )

            self.modulelst = nn.ModuleList(modulelst)

            self.final = CoFusion(5, 5)

    def forward(self, feats):
        _, _, H, W = feats[0].size()
        for i in range(len(feats)):
            feats[i] = F.interpolate(self.modulelst[i](feats[i]), (H, W), mode="bilinear")
        return self.final(torch.cat(feats, dim=1))

class Fusion_head(nn.Module):
    def __init__(self, channels):
        super(Fusion_head, self).__init__()

        modulelst = []
        for num in channels:
            modulelst.append(nn.Sequential(
                nn.Conv2d(num, 1, 1),
                nn.Sigmoid())
            )

            self.modulelst = nn.ModuleList(modulelst)

            self.final = nn.Sequential(nn.Conv2d(5,1,3,padding=1),
                                       nn.Sigmoid()
            )

    def forward(self, feats):
        _, _, H, W = feats[0].size()
        for i in range(len(feats)):
            feats[i] = F.interpolate(self.modulelst[i](feats[i]), (H, W), mode="bilinear")
        return self.final(torch.cat(feats, dim=1))


class Default_head(nn.Module):
    def __init__(self, channels):
        super(Default_head, self).__init__()

        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feats):
        return self.final(feats[0])
