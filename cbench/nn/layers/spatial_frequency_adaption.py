# Implementation from https://arxiv.org/pdf/2407.09853
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from .basic import BasicNNLayer

class SpatialModulationAdaptor(BasicNNLayer):
    def __init__(self, in_channels, mid_channels=None, **kwargs):
        super().__init__(in_channels, **kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels

        self.s_down1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.s_down2 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.s_dw = nn.Conv2d(mid_channels, mid_channels, 5, 1, 2, groups=mid_channels)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(mid_channels, in_channels, 1, 1, 0)

    def forward(self, x, **kwargs):
        s_modulate = self.s_up(self.s_relu(self.s_dw(self.s_down1(x)) * self.s_down2(x)))
        return s_modulate


class FrequencyModulationAdaptor(BasicNNLayer):
    def __init__(self, in_channels, mid_channels=None, **kwargs):
        super().__init__(in_channels, **kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels

        self.f_down = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(mid_channels, in_channels, 1, 1, 0)
        self.f_dw = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels)
        self.f_inter = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0)

    def forward(self, x, **kwargs):
        y = torch.fft.rfftn(self.f_down(x), dim=(-2,-1), norm="backward")
        y_amp = torch.abs(y)
        y_phs = torch.atan2(y.imag, y.real) # torch.angle(y) is not derivative in torch 1.7.1?
        y_amp_modulation = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * torch.sigmoid(y_amp_modulation)
        y_real = y_amp * torch.cos(y_phs)
        y_img = y_amp * torch.sin(y_phs)
        y = torch.complex(y_real, y_img)
        y = torch.fft.irfftn(y, s=x.shape[-2:], norm="backward")

        f_modulate = self.f_up(self.f_relu2(y))
        return f_modulate
    
class SpatialFrequencyModulationAdaptor(BasicNNLayer):
    def __init__(self, in_channels, mid_channels=None, factor=1., **kwargs):
        super().__init__(in_channels, **kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels
        self.factor = factor

        self.fma = FrequencyModulationAdaptor(in_channels, mid_channels=mid_channels, **kwargs)
        self.sma = SpatialModulationAdaptor(in_channels, mid_channels=mid_channels, **kwargs)

    def forward(self, x, factor=None, **kwargs):
        if factor is None: factor = self.factor
        if factor == 0:
            return x
        f_modulate = self.fma(x)
        s_modulate = self.sma(x)
        x_tilde = x + (f_modulate + s_modulate) * self.factor
        return x_tilde
