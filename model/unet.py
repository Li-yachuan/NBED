from torch import nn
import torch
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.InstanceNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):

        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode="bilinear")
            # x = F.interpolate(x, size=skip.size()[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()

        self.depth = len(encoder_channels)
        convs = dict()

        for d in range(self.depth - 1):
            if d == self.depth - 2:
                convs["conv{}".format(d)] = DecoderBlock(encoder_channels[d + 1],
                                                         encoder_channels[d],
                                                         decoder_channels[d])
            else:
                convs["conv{}".format(d)] = DecoderBlock(decoder_channels[d + 1],
                                                         encoder_channels[d],
                                                         decoder_channels[d])

        self.convs = nn.ModuleDict(convs)

        # self.final = nn.Sequential(
        #     nn.Conv2d(decoder_channels[0], 1, 3, padding=1),
        #     nn.Sigmoid())

    def forward(self, features):

        for d in range(self.depth - 2, -1, -1):
            features[d] = self.convs["conv{}".format(d)](features[d + 1], features[d])

        return features
        # return self.final(features[0])


class Identity(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()
        convs = []
        for ec, dc in zip(encoder_channels, decoder_channels):
            convs.append(nn.Conv2d(ec, dc, 1))
        self.convs = nn.ModuleList(convs)

    def forward(self, features):
        return [c(f) for f, c in zip(features, self.convs)]
