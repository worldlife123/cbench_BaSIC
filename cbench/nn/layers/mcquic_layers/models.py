import torch
from torch import nn

from .convs import conv3x3, pixelShuffle3x3
from .blocks import ResidualBlock, ResidualBlockWithStride, ResidualBlockShuffle, AttentionBlock

class Encoder(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
            # convs.conv3x3(3, channel),
            conv3x3(3, channel, 2),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockWithStride(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockWithStride(channel, channel, groups=1),
            ResidualBlock(channel, channel, groups=1)
        )


class Decoder(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockShuffle(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockShuffle(channel, channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            pixelShuffle3x3(channel, 3, 2)
        )


class LatentStageEncoder(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
                ResidualBlockWithStride(channel, channel, groups=1),
                # GroupSwishConv2D(channel, 3, groups=1),
                ResidualBlock(channel, channel, groups=1),
                AttentionBlock(channel, groups=1),
            )
class QuantizationHead(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
                ResidualBlock(channel, channel, groups=1),
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel)
                # convs.conv1x1(channel, channel, groups=1)
                # GroupSwishConv2D(channel, channel, groups=1)
            )


class LatentHead(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
                ResidualBlock(channel, channel, groups=1),
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel)
                # convs.conv1x1(channel, channel, groups=1)
            )


class RestoreHead(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
                AttentionBlock(channel, groups=1),
                ResidualBlock(channel, channel, groups=1),
                ResidualBlockShuffle(channel, channel, groups=1)
            )


class DequantizationHead(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel),
                ResidualBlock(channel, channel, groups=1),
            )


class SideHead(nn.Sequential):
    def __init__(self, channel):
        return super().__init__(
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel),
                ResidualBlock(channel, channel, groups=1),
            )
