import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv

from ..layers.slimmable_layers import BaseSlimmableLayer, DynamicConv2d, DynamicGDN

class BasicHyperpriorModule(nn.Module):
    def _channelwise_mul(self, input, gain):
        return (input.view(input.shape[0], input.shape[1], -1) * gain.unsqueeze(0).unsqueeze(-1)).view_as(input)

    def _forward(self, input, **kwargs):
        raise NotImplementedError()

    def forward(self, input, in_channel_gains=None, out_channel_gains=None, **kwargs):
        if in_channel_gains is not None:
            input = self._channelwise_mul(input, in_channel_gains)
        output = self._forward(input, **kwargs)
        if out_channel_gains is not None:
            output = self._channelwise_mul(output, out_channel_gains)
        return output

class HyperpriorAnalysisModel(BasicHyperpriorModule):
    def __init__(self, N, M, in_channels=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            conv(in_channels, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

    def _forward(self, x):
        return self.model(x)


class HyperpriorSynthesisModel(BasicHyperpriorModule):
    def __init__(self, N, M, out_channels=3, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.N = N
        self.M = M

        self.model =  nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_channels),
        )

    def _forward(self, x):
        return self.model(x)


class HyperpriorHyperAnalysisModel(BasicHyperpriorModule):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

    def _forward(self, x):
        return self.model(x)


class HyperpriorHyperSynthesisModel(BasicHyperpriorModule):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def _forward(self, x):
        return self.model(x)


class MeanScaleHyperpriorHyperAnalysisModel(BasicHyperpriorModule):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

    def _forward(self, x):
        return self.model(x)


class MeanScaleHyperpriorHyperSynthesisModel(BasicHyperpriorModule):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.N = N
        self.M = M

        self.model = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def _forward(self, x):
        return self.model(x)


class SlimmableHyperpriorAnalysisModel(BaseSlimmableLayer):
    def __init__(self, N_list, M_list, in_channels=3, **kwargs):
        super().__init__(M_list)
        self.in_channels = in_channels
        self.N_list = N_list
        self.M_list = M_list
        assert len(M_list) == len(N_list)

        self.model = nn.Sequential(
            DynamicConv2d(in_channels, N_list),
            DynamicGDN(N_list),
            DynamicConv2d(max(N_list), N_list),
            DynamicGDN(N_list),
            DynamicConv2d(max(N_list), N_list),
            DynamicGDN(N_list),
            DynamicConv2d(max(N_list), M_list),
        )

    def on_active_channels_idx_updated(self, value):
        for layer in self.model.modules():
            if isinstance(layer, BaseSlimmableLayer):
                layer.set_dynamic_parameter_value("active_channels_idx", value)

    def _forward(self, x):
        return self.model(x)


class SlimmableHyperpriorSynthesisModel(BaseSlimmableLayer):
    def __init__(self, N_list, M_list, out_channels=3, **kwargs):
        super().__init__(N_list)
        self.out_channels = out_channels
        self.N_list = N_list
        self.M_list = M_list
        assert len(M_list) == len(N_list)

        self.model =  nn.Sequential(
            DynamicConv2d(max(M_list), N_list, transposed=True),
            DynamicGDN(N_list, inverse=True),
            DynamicConv2d(max(N_list), N_list, transposed=True),
            DynamicGDN(N_list, inverse=True),
            DynamicConv2d(max(N_list), N_list, transposed=True),
            DynamicGDN(N_list, inverse=True),
            DynamicConv2d(max(N_list), [out_channels] * len(N_list), transposed=True),
        )

    def on_active_channels_idx_updated(self, value):
        for layer in self.model.modules():
            if isinstance(layer, BaseSlimmableLayer):
                layer.set_dynamic_parameter_value("active_channels_idx", value)

    def _forward(self, x):
        return self.model(x)

