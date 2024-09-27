import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Any, Sequence, Union, Optional
import random
import numpy as np

from .basic import BasicNNLayer

from ..base import NNTrainableModule, DynamicNNTrainableModule
from cbench.codecs.base import VariableComplexityCodecInterface

# from compressai.layers import GDN
from compressai.ops.parametrizers import NonNegativeParametrizer

# from thop.vision.basic_hooks import count_convNd, count_relu
# from thop.vision.calc_func import calculate_conv2d_flops # This func does not count bias as MAC, we reimplement it!

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


class BaseSlimmableLayer(DynamicNNTrainableModule, VariableComplexityCodecInterface):
    def __init__(self, channels_list : List[int], **kwargs):
        super().__init__()
        self.channels_list = channels_list
        if self.channels_list is not None:
            self.register_dynamic_parameter("active_channels_idx", torch.zeros(1) + len(channels_list) - 1, self.on_active_channels_idx_updated)

    def set_complex_level(self, level, *args, **kwargs) -> None:
        self.set_dynamic_parameter_value("active_channels_idx", torch.zeros(1) + level)
    
    @property
    def num_complex_levels(self) -> int:
        return len(self.channels_list)
    
    def on_active_channels_idx_updated(self, value):
        pass

    # TODO: should implement this in DynamicNNTrainableModule!
    # @property
    # def active_channels_idx(self):
    #     return self.get_dynamic_parameter_value("active_channels_idx")
    
    # @active_channels_idx.setter
    # def active_channels_idx(self, value):
    #     return self.set_dynamic_parameter_value("active_channels_idx", value)

    def get_active_channels(self) -> int:
        if self.channels_list is not None:
            return self.channels_list[self.get_dynamic_parameter_value("active_channels_idx")]
        else:
            return 0

    # def get_max_channels(self) -> Dict[str, int]:
    #     if self.channels_list is not None:
    #         return self.channels_list[-1]
    #     else:
    #         return 0

# TODO: dynamic kernel_size
class DynamicConv2d(BaseSlimmableLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=2, padding=None, dilation=1, bias=True, transposed=False, conv_compability=False, **kwargs
    ):

        if isinstance(in_channels, Sequence):
            max_in_channels = max(in_channels)
        else:
            max_in_channels = in_channels

        if isinstance(out_channels, Sequence):
            out_channels_list = list(out_channels)
            max_out_channels = max(out_channels_list)
        else:
            out_channels_list = [out_channels]
            max_out_channels = out_channels

        super().__init__(out_channels_list)
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.bias = bias
        self.transposed = transposed
        self.conv_compability = conv_compability

        if transposed:
            conv_module = nn.ConvTranspose2d(
                self.max_in_channels,
                self.max_out_channels,
                self.kernel_size,
                stride=self.stride,
                output_padding=self.stride-1,
                bias=bias,
            )
        else:
            conv_module = nn.Conv2d(
                self.max_in_channels,
                self.max_out_channels,
                self.kernel_size,
                stride=self.stride,
                bias=bias,
            )

        if self.conv_compability:
            self.weight = conv_module.weight
            if bias:
                self.bias = conv_module.bias
            else:
                self.register_parameter('bias', None)
        else:
            self.conv = conv_module

        self.active_out_channel = self.max_out_channels

        # flops counter
        self.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)
        self.register_forward_hook(count_dynamic_convNd)

    def get_current_flops(self, input=None):
        return self.total_ops.squeeze().clone()
    
    def get_nn_complexity(self, input=None, metric=None):
        # TODO: implement other metrics. By default we return total_ops regardless of metrics
        return self.total_ops.squeeze().clone()

    def on_active_channels_idx_updated(self, value):
        self.active_out_channel = self.channels_list[int(value)]

    def get_active_filter(self, out_channel, in_channel):
        return self.get_weight()[:out_channel, :in_channel, :, :]
    
    def get_weight(self):
        if self.conv_compability:
            return self.weight
        else:
            return self.conv.weight

    def get_bias(self):
        if self.conv_compability:
            return self.bias
        else:
            return self.conv.bias

    def forward(self, x, out_channel=None):
        # clear ops counter
        self.total_ops.data.fill_(0)

        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        if self.transposed:
            filters = self.get_active_filter(in_channel, out_channel).contiguous()
        else:
            filters = self.get_active_filter(out_channel, in_channel).contiguous()
        bias = self.get_bias()
        if self.bias is not None:
            bias = bias[:out_channel] 

        padding = get_same_padding(self.kernel_size) if self.padding is None else self.padding
        # filters = (
        #     self.conv.weight_standardization(filters)
        #     if isinstance(self.conv, MyConv2d)
        #     else filters
        # )
        if self.transposed:
            output_padding = self.stride - 1 # TODO:
            y = F.conv_transpose2d(x, filters, bias, self.stride, padding, output_padding, 1, self.dilation)
        else:
            y = F.conv2d(x, filters, bias, self.stride, padding, self.dilation, 1)
        return y

# reimplement of thop.vision.calc_func import calculate_conv2d_flops, adding bias into MAC count
def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    in_kernel_ops = (in_c // g) * np.prod(kernel_size[2:])
    if bias:
        in_kernel_ops += 1
    return np.prod(output_size) * in_kernel_ops

def count_dynamic_convNd(m: DynamicConv2d, x, y: torch.Tensor):
    x = x[0]

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.get_weight().shape),
        groups = 1,
        bias = m.get_bias() is not None,
    )


class DynamicGDN(BaseSlimmableLayer):

    def __init__(
        self,
        channels_list : List[int],
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        **kwargs
    ):
        super().__init__(channels_list=channels_list, **kwargs)
        in_channels = max(channels_list)

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

        self.deform_dim = len(self.channels_list)

        self.gamma_scales = nn.Parameter(torch.ones(self.deform_dim))
        self.gamma_scales_reparam = NonNegativeParametrizer()
        self.gamma_biases = nn.Parameter(torch.zeros(self.deform_dim))
        self.gamma_biases_reparam = NonNegativeParametrizer()
        self.beta_scales = nn.Parameter(torch.ones(self.deform_dim))
        self.beta_scales_reparam = NonNegativeParametrizer()
        self.beta_biases = nn.Parameter(torch.zeros(self.deform_dim))
        self.beta_biases_reparam = NonNegativeParametrizer()

        # flops counter
        self.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)
        self.register_forward_hook(count_gdn)

    def get_current_flops(self, input=None):
        return self.total_ops.squeeze().clone()

    def get_nn_complexity(self, input=None, metric=None):
        # TODO: implement other metrics. By default we return total_ops regardless of metrics
        return self.total_ops.squeeze().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clear ops counter
        self.total_ops.data.fill_(0)

        _, C, _, _ = x.size()

        if self.channels_list is None:
            idx = C-1
        else:
            idx = int(self.get_dynamic_parameter_value("active_channels_idx"))
            assert self.channels_list[idx] == C

        beta = self.beta_scales_reparam(self.beta_scales[idx]) * self.beta_reparam(self.beta[:C]) + self.beta_biases_reparam(self.beta_biases[idx])
        gamma = self.gamma_scales_reparam(self.gamma_scales[idx]) * self.gamma_reparam(self.gamma[:C, :C]) + self.gamma_biases_reparam(self.gamma_biases[idx])
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

def count_gdn(m: DynamicGDN, x, y: torch.Tensor):
    x = x[0]

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = [x.shape[1], x.shape[1], 1, 1],
        groups = m.groups if hasattr(m, "groups") else 1,
        bias=True,
    )
    # TODO: should sqrt/pow counted as flops?


# From CompressAI
def dynamic_conv3x3(in_ch: Union[int, List[int]], out_ch: Union[int, List[int]], stride: int = 1, **kwargs) -> nn.Module:
    """3x3 convolution with padding."""
    in_ch = max(in_ch) if isinstance(in_ch, list) else in_ch
    out_ch = out_ch if isinstance(out_ch, list) else [out_ch]
    return DynamicConv2d(in_ch, out_ch, kernel_size=3, stride=stride, **kwargs)


def dynamic_subpel_conv3x3(in_ch: Union[int, List[int]], out_ch: Union[int, List[int]], r: int = 1, **kwargs) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    in_ch = max(in_ch) if isinstance(in_ch, list) else in_ch
    out_ch = out_ch if isinstance(out_ch, list) else [out_ch]
    out_ch = [ch * r**2 for ch in out_ch]
    return nn.Sequential(
        DynamicConv2d(in_ch, out_ch, kernel_size=3, stride=1, **kwargs), nn.PixelShuffle(r)
    )


def dynamic_conv1x1(in_ch: Union[int, List[int]], out_ch: Union[int, List[int]], stride: int = 1, **kwargs) -> nn.Module:
    """1x1 convolution."""
    in_ch = max(in_ch) if isinstance(in_ch, list) else in_ch
    out_ch = out_ch if isinstance(out_ch, list) else [out_ch]
    return DynamicConv2d(in_ch, out_ch, kernel_size=1, stride=stride, **kwargs)


class DynamicResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, **kwargs):
        super().__init__()
        self.conv1 = dynamic_conv3x3(in_ch, out_ch, stride=stride, **kwargs)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = dynamic_conv3x3(out_ch, out_ch, **kwargs)
        self.gdn = DynamicGDN(out_ch, **kwargs)
        if stride != 1 or in_ch != out_ch:
            self.skip = dynamic_conv1x1(in_ch, out_ch, stride=stride, **kwargs)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class DynamicResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: List[int], upsample: int = 2, **kwargs):
        super().__init__()
        self.subpel_conv = dynamic_subpel_conv3x3(in_ch, out_ch, upsample, **kwargs)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = dynamic_conv3x3(out_ch, out_ch, **kwargs)
        self.igdn = DynamicGDN(out_ch, inverse=True)
        self.upsample = dynamic_subpel_conv3x3(in_ch, out_ch, upsample, **kwargs)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class DynamicResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: List[int], **kwargs):
        super().__init__()
        self.conv1 = dynamic_conv3x3(in_ch, out_ch, **kwargs)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = dynamic_conv3x3(out_ch, out_ch, **kwargs)
        if in_ch != out_ch:
            self.skip = dynamic_conv1x1(in_ch, out_ch, **kwargs)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class DynamicResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: List[int], mid_ch : Optional[List[int]] = None, skip=False, **kwargs):
        super().__init__()
        # mid_ch = min(in_ch, out_ch) // 2 if mid_ch is None else mid_ch
        mid_ch = [c//2 for c in out_ch] if mid_ch is None else mid_ch
        self.conv1 = dynamic_conv1x1(in_ch, mid_ch, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = dynamic_conv3x3(mid_ch, mid_ch, **kwargs)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = dynamic_conv1x1(mid_ch, out_ch, **kwargs)
        self.skip = dynamic_conv1x1(in_ch, out_ch, **kwargs) if skip else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity


class DynamicAttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: List[int], **kwargs):
        super().__init__()

        N_half = [n // 2 for n in N]

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    dynamic_conv1x1(N, N_half, **kwargs),
                    nn.ReLU(inplace=True),
                    dynamic_conv3x3(N_half, N_half, **kwargs),
                    nn.ReLU(inplace=True),
                    dynamic_conv1x1(N_half, N, **kwargs),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            dynamic_conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


# from https://github.com/Zhichen-Zhang/ELFIC-Image-Compression
class DSConv2d(nn.Conv2d, BaseSlimmableLayer):
    def __init__(self,
                 in_channels_list,  # 输入通道数量的动态变化列表
                 out_channels_list,  # 输出通道数量的动态变化列表
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 cat_factor=1):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]  # 转成list or tuple 的数据格式
        super(DSConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)  # 对继承的nn.conv2d 使用输入的参数进行初始化~
        BaseSlimmableLayer.__init__(self, out_channels_list)
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2  # 计算same padding 应该填充的宽度
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        # self.active_out_channel = out_channels_list[-1]
        self.cat_factor = cat_factor

    def forward(self, x):
        self.running_inc = x.size(1)
        self.running_outc = self.get_active_channels()
        if self.cat_factor == 1:
            weight = self.weight[:self.running_outc, :self.running_inc]
        else:
            self.running_inc = x.size(1) // self.cat_factor
            self.weight_chunk = self.weight.chunk(self.cat_factor, dim=1)
            weight = [i[:self.running_outc, :self.running_inc] for i in self.weight_chunk]
            weight = torch.cat(weight, dim=1)

        bias = self.bias[:self.running_outc] if self.bias is not None else None
        self.running_groups = 1 if self.groups == 1 else self.running_outc

        return F.conv2d(x,
                        weight,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.running_groups)


class DSTransposeConv2d(nn.ConvTranspose2d, BaseSlimmableLayer):
    def __init__(self,
                 in_channels_list,  # 输入通道数量的动态变化列表
                 out_channels_list,  # 输出通道数量的动态变化列表
                 kernel_size,
                 stride=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 output_padding=1):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]  # 转成list or tuple 的数据格式
        super(DSTransposeConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            output_padding=output_padding)  # 对继承的nn.conv2d 使用输入的参数进行初始化~
        BaseSlimmableLayer.__init__(self, out_channels_list)
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2  # 计算same padding 应该填充的宽度
        self.padding = (padding, padding)
        self.output_padding = output_padding
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        # self.active_out_channel = out_channels_list[-1]

    def forward(self, x, *args, **kwargs):
        self.running_inc = x.size(1)
        self.running_outc = self.get_active_channels()
        weight = self.weight[:self.running_inc, :self.running_outc]  # 卷积和反卷积的卷积核其对应的输入通道是相反的。
        bias = self.bias[:self.running_outc] if self.bias is not None else None
        self.running_groups = 1 if self.groups == 1 else self.running_outc
        return F.conv_transpose2d(x,
                                  weight,
                                  bias,
                                  self.stride,
                                  self.padding,
                                  self.output_padding,
                                  self.groups)

class ELFICBasicBottleneckDenseBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride, dilation=1,
                 output_padding=1,
                 bias=True, Tconv=False):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation

        # Basic 2D convolution
        if Tconv:
            self.conv = DSTransposeConv2d(in_channels_list,
                                      out_channels_list,
                                      kernel_size=kernel_size,
                                      stride=(stride, stride),
                                      bias=bias,
                                      transposed=True,
                                      output_padding=(output_padding, output_padding))
            self.tconv = True
        else:
            self.conv = DSConv2d(in_channels_list,
                                 out_channels_list,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 dilation=(dilation, dilation),
                                 bias=bias)
            self.tconv = False


        self.block_1 = nn.ModuleList([
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(3, 3),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d([channel * 3 for channel in out_channels_list],
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias,
                     cat_factor=3),
        ]
        )

        self.block_2 = nn.ModuleList([
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(3, 3),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d([channel * 3 for channel in out_channels_list],
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias,
                     cat_factor=3),
        ]
        )

        self.block_3 = nn.ModuleList([
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(3, 3),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d([channel * 3 for channel in out_channels_list],
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias,
                     cat_factor=3),
        ]
        )

        self.global_fusion = DSConv2d([channel * 3 for channel in out_channels_list],
                                      out_channels_list,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      dilation=(1, 1),
                                      bias=bias,
                                      cat_factor=3)

        # self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.conv(x)
        main = x
        # self.active_out_channel = self.conv.active_out_channel
        # self.set_active_channels()

        x_1 = F.relu(self.block_1[0](x), inplace=True)
        x_2 = F.relu(self.block_1[1](x_1), inplace=True)
        x_3 = F.relu(self.block_1[2](x_2), inplace=True)
        identity_1 = self.block_1[3](torch.cat([x_1, x_2, x_3], dim=1))
        x = x + identity_1

        x_1 = F.relu(self.block_2[0](x), inplace=True)
        x_2 = F.relu(self.block_2[1](x_1), inplace=True)
        x_3 = F.relu(self.block_2[2](x_2), inplace=True)
        identity_2 = self.block_2[3](torch.cat([x_1, x_2, x_3], dim=1))
        x = x + identity_2

        x_1 = F.relu(self.block_3[0](x), inplace=True)
        x_2 = F.relu(self.block_3[1](x_1), inplace=True)
        x_3 = F.relu(self.block_3[2](x_2), inplace=True)
        identity_3 = self.block_3[3](torch.cat([x_1, x_2, x_3], dim=1))
        #
        x = main + self.global_fusion(torch.cat([identity_1, identity_2, identity_3], dim=1))
        return x


# Implementation from https://arxiv.org/pdf/2407.09853
class DynamicSpatialModulationAdaptor(BasicNNLayer):
    def __init__(self, in_channels, mid_channels=None, **kwargs):
        super().__init__(in_channels, **kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels

        self.s_down1 = DynamicConv2d(in_channels, mid_channels, 1, 1, 0)
        self.s_down2 = DynamicConv2d(in_channels, mid_channels, 1, 1, 0)
        self.s_dw = DynamicConv2d(mid_channels, mid_channels, 5, 1, 2, groups=mid_channels)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = DynamicConv2d(mid_channels, in_channels, 1, 1, 0)

    def forward(self, x, **kwargs):
        s_modulate = self.s_up(self.s_relu(self.s_dw(self.s_down1(x)) * self.s_down2(x)))
        return s_modulate


class DynamicFrequencyModulationAdaptor(BasicNNLayer):
    def __init__(self, in_channels, mid_channels=None, **kwargs):
        super().__init__(in_channels, **kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels

        self.f_down = DynamicConv2d(in_channels, mid_channels, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = DynamicConv2d(mid_channels, in_channels, 1, 1, 0)
        self.f_dw = DynamicConv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels)
        self.f_inter = DynamicConv2d(mid_channels, mid_channels, 1, 1, 0)

    # See https://discuss.pytorch.org/t/how-to-avoid-nan-output-from-atan2-during-backward-pass/176890/7
    def _stable_atan2(self, a, b):
        epsilon = 1e-10 
        # near_zeros = a < epsilon
        # a = a * (near_zeros.logical_not())
        # a = a + (near_zeros * epsilon)
        a = torch.where(a>=0, a+epsilon, a-epsilon)
        return torch.atan2(b, a)
    
    def forward(self, x, **kwargs):
        y = torch.fft.rfftn(self.f_down(x), dim=(-2,-1), norm="backward")
        y_amp = torch.abs(y)
        y_phs = self._stable_atan2(y.imag, y.real) # torch.angle(y) is not derivative in torch 1.7.1?
        y_amp_modulation = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * torch.sigmoid(y_amp_modulation)
        y_real = y_amp * torch.cos(y_phs)
        y_img = y_amp * torch.sin(y_phs)
        y = torch.complex(y_real, y_img)
        y = torch.fft.irfftn(y, s=x.shape[-2:], norm="backward")

        f_modulate = self.f_up(self.f_relu2(y))
        return f_modulate


class DynamicSpatialFrequencyModulationAdaptor(nn.Module):
    def __init__(self, in_channels, mid_channels=None, factor=1., **kwargs):
        super().__init__(in_channels, **kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels
        self.factor = factor

        self.fma = DynamicFrequencyModulationAdaptor(in_channels, mid_channels=mid_channels, **kwargs)
        self.sma = DynamicSpatialModulationAdaptor(in_channels, mid_channels=mid_channels, **kwargs)

    def forward(self, x, factor=None, **kwargs):
        if factor is None: factor = self.factor
        if factor == 0:
            return x
        f_modulate = self.fma(x)
        s_modulate = self.sma(x)
        x_tilde = x + (f_modulate + s_modulate) * self.factor
        return x_tilde


class GroupedDynamicSpatialFrequencyModulationAdaptor(DynamicNNTrainableModule):
    def __init__(self, in_channels, mid_channels=None, factor=1., num_modulators=1, **kwargs):
        super().__init__(**kwargs)

        mid_channels = mid_channels if mid_channels is not None else in_channels
        self.factor = factor

        self.fma_modules = nn.ModuleList([DynamicFrequencyModulationAdaptor(in_channels, mid_channels=mid_channels, **kwargs) for _ in range(num_modulators)])
        self.sma_modules = nn.ModuleList([DynamicSpatialModulationAdaptor(in_channels, mid_channels=mid_channels, **kwargs) for _ in range(num_modulators)])

        self.register_dynamic_parameter("sfma_idx", torch.zeros(1))

    def forward(self, x, idx=None, **kwargs):
        if idx is None: idx = int(self.get_dynamic_parameter_value("sfma_idx").item())
        if idx <= 0: return x # no mod for idx<=0
        f_modulate = self.fma_modules[idx-1](x)
        s_modulate = self.sma_modules[idx-1](x)
        x_tilde = x + (f_modulate + s_modulate) * self.factor
        return x_tilde