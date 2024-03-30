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

        # encoder_channels: 96,192,384,576
        # decoder_channel:  32,64, 128,256

        # self.conv11 = DecoderBlock(encoder_channels[1], encoder_channels[0], decoder_channels[0])
        # self.conv21 = DecoderBlock(encoder_channels[2], encoder_channels[1], decoder_channels[1])
        # self.conv31 = DecoderBlock(encoder_channels[3], encoder_channels[2], decoder_channels[2])
        #
        # self.conv12 = DecoderBlock(decoder_channels[1], decoder_channels[0], decoder_channels[0])
        # self.conv22 = DecoderBlock(decoder_channels[2], decoder_channels[1], decoder_channels[1])
        #
        # self.conv13 = DecoderBlock(decoder_channels[1], decoder_channels[0], decoder_channels[0])
        self.width = len(encoder_channels)
        convs = dict()
        for w in range(self.width - 1):
            for d in range(self.width - w - 1):
                if w == 0:
                    convs["conv{}_{}".format(d, w)] = DecoderBlock(encoder_channels[d + 1],
                                                                   encoder_channels[d],
                                                                   decoder_channels[d])
                else:
                    convs["conv{}_{}".format(d, w)] = DecoderBlock(decoder_channels[d + 1],
                                                                   decoder_channels[d],
                                                                   decoder_channels[d])


        self.convs = nn.ModuleDict(convs)

        # self.final = nn.Sequential(
        #     nn.Conv2d(decoder_channels[0], 1, 3, padding=1),
        #     nn.Sigmoid())

    def forward(self, features):

        # features[0] = self.conv11(features[1], features[0])
        # features[1] = self.conv21(features[2], features[1])
        # features[2] = self.conv31(features[3], features[2])
        #
        # features[0] = self.conv12(features[1], features[0])
        # features[1] = self.conv22(features[2], features[1])
        #
        # features[0] = self.conv13(features[1], features[0])
        for w in range(self.width - 1):
            for d in range(self.width - w - 1):
                features[d] = self.convs["conv{}_{}".format(d, w)](features[d + 1], features[d])
        # return self.final(features[0])
        return features
