import math
import struct
import io
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv
from compressai.ans import BufferedRansEncoder, RansDecoder

from .base import PriorCoder
from cbench.codecs.base import VariableRateCodecInterface
from cbench.nn.base import NNTrainableModule
from cbench.nn.models.google import HyperpriorHyperSynthesisModel, HyperpriorHyperAnalysisModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def read_body(fd, segments=1):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        batch = []
        for seglen in read_uints(fd, segments):
            batch.append(read_bytes(fd, seglen))
        lstrings.append(batch)

    return lstrings, shape


def write_body(fd, shape, out_strings, segments=1):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        assert len(s) == segments
        bytes_cnt += write_uints(fd, [len(seg) for seg in s])
        for segidx in range(segments):
            bytes_cnt += write_bytes(fd, s[segidx])
    return bytes_cnt


class CompressAIEntropyBottleneckPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self, entropy_bottleneck_channels=256, 
                 eps=1e-7, 
                 use_inner_aux_opt=False, 
                 use_bit_rate_loss=True,
                 freeze_params=False,
                 training_output_straight_through=False,
                 **kwargs):
        super().__init__()
        NNTrainableModule.__init__(self)

        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.eps = eps
        self.use_inner_aux_opt = use_inner_aux_opt
        self.use_bit_rate_loss = use_bit_rate_loss
        self.freeze_params = freeze_params
        self.training_output_straight_through = training_output_straight_through
        # self.update_state()

        if self.freeze_params:
            for param in self.parameters():
                param.requires_grad = False
        else:
            # use aux optimizer for quantiles
            aux_params = []
            for name, param in self.entropy_bottleneck.named_parameters():
                if param.requires_grad and name.endswith("quantiles"):
                    aux_params.append(param)
            if self.use_inner_aux_opt:
                self.aux_opt = optim.Adam(aux_params, lr=1e-3)
            else:
                # mark params for external aux optimizer
                for param in aux_params:
                    param.aux_id = 0

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    # def load_state_dict(self, state_dict):
    #     # Dynamically update the entropy bottleneck buffers related to the CDFs
    #     update_registered_buffers(
    #         self.entropy_bottleneck,
    #         "entropy_bottleneck",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def _optimize_aux_params(self):
        if not self.freeze_params:
            loss_aux = self.aux_loss()
            if self.use_inner_aux_opt:
                self.aux_opt.zero_grad()
                loss_aux.backward()
                self.aux_opt.step()
                self.update_cache("moniter_dict",
                    loss_aux = loss_aux,
                )
            else:
                self.update_cache("loss_dict", 
                    loss_aux = loss_aux,
                )

    def _channelwise_mul(self, input, gain):
        return (input.view(input.shape[0], input.shape[1], -1) * gain.unsqueeze(0).unsqueeze(-1)).view_as(input)

    def forward(self, input, *args, channel_gains=None, channel_gains_inv=None, **kwargs):
        if channel_gains is not None:
            input = self._channelwise_mul(input, channel_gains)
        y_hat, y_likelihoods = self.entropy_bottleneck(input)
        if channel_gains_inv is not None:
            y_hat = self._channelwise_mul(y_hat, channel_gains_inv)

        entropy = -torch.log(y_likelihoods).sum() 
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / input.shape[0] # normalize by batch size
            )
            self._optimize_aux_params()
        self.update_cache("metric_dict",
            prior_entropy = entropy / input.shape[0], # normalize by batch size
        )

        if self.training_output_straight_through:
            input_quant =  self.entropy_bottleneck.quantize(
                input, "dequantize", self.entropy_bottleneck._get_medians()
            )
            y_hat = input_quant + input - input.detach()

        return y_hat

    def encode(self, input, *args, channel_gains=None, channel_gains_inv=None, **kwargs) -> bytes:
        if channel_gains is not None:
            input = self._channelwise_mul(input, channel_gains)
        y_strings = self.entropy_bottleneck.compress(input)
        with io.BytesIO() as bio:
            write_body(bio, input.size()[-2:], [[string] for string in y_strings])
            return bio.getvalue()

    def decode(self, byte_string, *args, channel_gains=None, channel_gains_inv=None, **kwargs):
        with io.BytesIO(byte_string) as bio:
            strings, shape = read_body(bio)
            # assert isinstance(strings, list) and len(strings) == 1
            y_hat = self.entropy_bottleneck.decompress([string[0] for string in strings], shape)
            if channel_gains_inv is not None:
                y_hat = self._channelwise_mul(y_hat, channel_gains_inv)
            return y_hat

    def update_state(self, *args, **kwargs) -> None:
        return self.update(*args, **kwargs)


class CompressAISlimmableEntropyBottleneckPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self, entropy_bottleneck_channels_list=[256], 
                 eps=1e-7, 
                 use_inner_aux_opt=False, 
                 use_bit_rate_loss=True,
                 freeze_params=False,
                 **kwargs):
        super().__init__()
        NNTrainableModule.__init__(self)

        self.entropy_bottleneck_channels_list = entropy_bottleneck_channels_list

        entropy_bottlenecks = []
        for entropy_bottleneck_channels in entropy_bottleneck_channels_list:
            entropy_bottlenecks.append(
                CompressAIEntropyBottleneckPriorCoder(entropy_bottleneck_channels, 
                                                      eps=eps,
                                                      use_inner_aux_opt=use_inner_aux_opt,
                                                      use_bit_rate_loss=use_bit_rate_loss,
                                                      freeze_params=freeze_params,
                                                      **kwargs)
            )
        self.entropy_bottlenecks = nn.ModuleList(entropy_bottlenecks)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, force=False):
        for entropy_bottleneck in self.entropy_bottlenecks:
            entropy_bottleneck.update(force=force)

    # def load_state_dict(self, state_dict):
    #     # Dynamically update the entropy bottleneck buffers related to the CDFs
    #     update_registered_buffers(
    #         self.entropy_bottleneck,
    #         "entropy_bottleneck",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def forward(self, input, *args, slim_level=None, **kwargs):
        input_channels = input.shape[1]
        module_idx = self.entropy_bottleneck_channels_list.index(input_channels) if slim_level is None else slim_level
        return self.entropy_bottlenecks[module_idx](input, *args, **kwargs)

    def encode(self, input, *args, slim_level=None, **kwargs) -> bytes:
        # TODO: add slim_level to bytes if slim_level is None
        input_channels = input.shape[1]
        module_idx = self.entropy_bottleneck_channels_list.index(input_channels)  if slim_level is None else slim_level
        return self.entropy_bottlenecks[module_idx].encode(input, *args, **kwargs)

    def decode(self, byte_string, *args, slim_level=None, **kwargs):
        # TODO: add slim_level to bytes if slim_level is None
        module_idx = self.entropy_bottleneck_channels_list.index(input.shape[1]) if slim_level is None else slim_level
        return self.entropy_bottlenecks[module_idx].decode(byte_string, *args, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        return self.update(*args, **kwargs)


class CompressAIGaussianConditionalCoder(PriorCoder, NNTrainableModule):
    def __init__(self, use_bit_rate_loss=True, training_output_straight_through=False, **kwargs):
        super().__init__(**kwargs)
        NNTrainableModule.__init__(self)
        self.gaussian_conditional = GaussianConditional(None)
        self.use_bit_rate_loss = use_bit_rate_loss
        self.training_output_straight_through = training_output_straight_through

    def _channelwise_mul(self, input, gain):
        return (input.view(input.shape[0], input.shape[1], -1) * gain.unsqueeze(0).unsqueeze(-1)).view_as(input)

    def forward(self, y, *args, prior=None, channel_gains=None, channel_gains_inv=None, **kwargs):
        if channel_gains is not None:
            y = self._channelwise_mul(y, channel_gains)
        y_hat, y_likelihoods = self.gaussian_conditional(y, prior[..., :y.shape[-2], :y.shape[-1]])
        if channel_gains_inv is not None:
            y_hat = self._channelwise_mul(y_hat, channel_gains_inv)

        entropy = -torch.log(y_likelihoods).sum()
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / y.shape[0] # normalize by batch size
            )
        self.update_cache("metric_dict",
            prior_entropy = entropy / y.shape[0], # normalize by batch size
        )

        if self.training_output_straight_through:
            y_quant =  self.gaussian_conditional.quantize(
                y, "dequantize"
            )
            y_hat = y_quant + y - y.detach()
        return y_hat

    def encode(self, y, *args, prior=None, channel_gains=None, channel_gains_inv=None, **kwargs):
        indexes = self.gaussian_conditional.build_indexes(prior[..., :y.shape[-2], :y.shape[-1]])
        if channel_gains is not None:
            y = self._channelwise_mul(y, channel_gains)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        with io.BytesIO() as bio:
            write_body(bio, y.size()[-2:], [[string] for string in y_strings])
            return bio.getvalue()

    def decode(self, byte_string, *args, prior=None, channel_gains=None, channel_gains_inv=None, **kwargs):
        with io.BytesIO(byte_string) as bio:
            strings, shape = read_body(bio)
            indexes = self.gaussian_conditional.build_indexes(prior[..., :shape[-2], :shape[-1]])
            y_hat = self.gaussian_conditional.decompress([string[0] for string in strings], indexes)
            if channel_gains_inv is not None:
                y_hat = self._channelwise_mul(y_hat, channel_gains_inv)
            return y_hat

    def update_state(self, *args, **kwargs) -> None:
        self.gaussian_conditional.update_scale_table(get_scale_table())
        return super().update_state(*args, **kwargs)


class CompressAIScaleHyperpriorCoder(CompressAIEntropyBottleneckPriorCoder, VariableRateCodecInterface):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=192, 
                num_vr_gains=1,
                ha_use_y_vr_gains=True,
                **kwargs):
        entropy_bottleneck_channels = kwargs.pop('entropy_bottleneck_channels')
        entropy_bottleneck_channels = N
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels, **kwargs)

        self.h_a = HyperpriorHyperAnalysisModel(N, M)

        self.h_s = HyperpriorHyperSynthesisModel(N, M)

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.num_vr_gains = int(num_vr_gains)
        self.ha_use_y_vr_gains = ha_use_y_vr_gains
        if self.num_vr_gains > 1:
            self.y_vr_gains = nn.Parameter(torch.ones(self.num_vr_gains, M))
            self.y_vr_gains_inv = nn.Parameter(torch.ones(self.num_vr_gains, M))
            self.z_vr_gains = nn.Parameter(torch.ones(self.num_vr_gains, N))
            self.z_vr_gains_inv = nn.Parameter(torch.ones(self.num_vr_gains, N))
            self.active_vr_level = 0

    def set_rate_level(self, level, *args, **kwargs) -> bytes:
        if self.num_vr_gains > 1:
            assert level >= 0 and level < self.num_vr_gains
            self.active_vr_level = level
    
    @property
    def num_rate_levels(self):
        return self.num_vr_gains

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def _channelwise_mul(self, input, gain):
        return (input.view(input.shape[0], input.shape[1], -1) * gain.unsqueeze(0).unsqueeze(-1)).view_as(input)

    def forward(self, y, *args, **kwargs):
        if self.num_vr_gains > 1 and self.ha_use_y_vr_gains:
            y = self._channelwise_mul(y, self.y_vr_gains[self.active_vr_level])
        z = self.h_a(torch.abs(y))
        if self.num_vr_gains > 1:
            z = self._channelwise_mul(z, self.z_vr_gains[self.active_vr_level])
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.num_vr_gains > 1:
            z_hat = self._channelwise_mul(z_hat, self.z_vr_gains_inv[self.active_vr_level])
        scales_hat = self.h_s(z_hat)
        if self.num_vr_gains > 1 and not self.ha_use_y_vr_gains:
            y = self._channelwise_mul(y, self.y_vr_gains[self.active_vr_level])
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        if self.num_vr_gains > 1:
            y_hat = self._channelwise_mul(y_hat, self.y_vr_gains_inv[self.active_vr_level])

        entropy = (-torch.log(y_likelihoods).sum() - torch.log(z_likelihoods).sum())
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / y.shape[0] # normalize by batch size
            )
            self._optimize_aux_params()
        self.update_cache("metric_dict",
            prior_entropy = entropy / y.shape[0], # normalize by batch size
        )
        return y_hat


    def encode(self, y, *args, **kwargs):
        if self.num_vr_gains > 1 and self.ha_use_y_vr_gains:
            y = self._channelwise_mul(y, self.y_vr_gains[self.active_vr_level])
        z = self.h_a(torch.abs(y))
        if self.num_vr_gains > 1:
            z = self._channelwise_mul(z, self.z_vr_gains[self.active_vr_level])

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        if self.num_vr_gains > 1:
            z_hat = self._channelwise_mul(z_hat, self.z_vr_gains_inv[self.active_vr_level])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        if self.num_vr_gains > 1 and not self.ha_use_y_vr_gains:
            y = self._channelwise_mul(y, self.y_vr_gains[self.active_vr_level])
        y_strings = self.gaussian_conditional.compress(y, indexes)
        with io.BytesIO() as bio:
            write_body(bio, z.size()[-2:], [[y_string, z_string] for y_string, z_string in zip(y_strings, z_strings)], segments=2)
            return bio.getvalue()

    def decode(self, byte_string, *args, **kwargs):
        with io.BytesIO(byte_string) as bio:
            strings, shape = read_body(bio, segments=2)
            # assert isinstance(strings, list) and len(strings) == 2
            z_hat = self.entropy_bottleneck.decompress([string[1] for string in strings], shape)
            if self.num_vr_gains > 1:
                z_hat = self._channelwise_mul(z_hat, self.z_vr_gains_inv[self.active_vr_level])
            scales_hat = self.h_s(z_hat)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat = self.gaussian_conditional.decompress([string[0] for string in strings], indexes, z_hat.dtype)
            if self.num_vr_gains > 1:
                y_hat = self._channelwise_mul(y_hat, self.y_vr_gains_inv[self.active_vr_level])
            return y_hat

    def update_state(self, *args, **kwargs) -> None:
        self.gaussian_conditional.update_scale_table(get_scale_table())
        return super().update_state(*args, **kwargs)


class CompressAIMeanScaleHyperpriorCoder(CompressAIScaleHyperpriorCoder):
    def __init__(self, N=128, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, y, *args, **kwargs):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        entropy = (-torch.log(y_likelihoods).sum() - torch.log(z_likelihoods).sum())
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / y.shape[0] # normalize by batch size
            )
            self._optimize_aux_params()
        self.update_cache("metric_dict",
            prior_entropy = entropy / y.shape[0], # normalize by batch size
        )
        return y_hat


    # def encode(self, y):
    #     z = self.h_a(y)

    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    #     gaussian_params = self.h_s(z_hat)
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
    #     return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    # def decode(self, strings, shape):
    #     assert isinstance(strings, list) and len(strings) == 2
    #     z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    #     gaussian_params = self.h_s(z_hat)
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_hat = self.gaussian_conditional.decompress(
    #         strings[0], indexes, means=means_hat
    #     )
    #     return {"x_hat": x_hat}


class CompressAIJointAutoregressiveCoder(CompressAIMeanScaleHyperpriorCoder):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, y, *args, **kwargs):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        entropy = (-torch.log(y_likelihoods).sum() - torch.log(z_likelihoods).sum())
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / y.shape[0] # normalize by batch size
            )
            self._optimize_aux_params()
        self.update_cache("metric_dict",
            prior_entropy = entropy / y.shape[0], # normalize by batch size
        )

        if self.training_output_straight_through:
            y_quant =  self.gaussian_conditional.quantize(
                y, "dequantize"#, means_hat
            )
            y_hat = y_quant + y - y.detach()

        return y_hat
        
    # def compress(self, y):
    #     if next(self.parameters()).device != torch.device("cpu"):
    #         warnings.warn(
    #             "Inference on GPU is not recommended for the autoregressive "
    #             "models (the entropy coder is run sequentially on CPU).",
    #             stacklevel=2,
    #         )

    #     z = self.h_a(y)

    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    #     params = self.h_s(z_hat)

    #     s = 4  # scaling factor between z and y
    #     kernel_size = 5  # context prediction kernel size
    #     padding = (kernel_size - 1) // 2

    #     y_height = z_hat.size(2) * s
    #     y_width = z_hat.size(3) * s

    #     y_hat = F.pad(y, (padding, padding, padding, padding))

    #     y_strings = []
    #     for i in range(y.size(0)):
    #         string = self._compress_ar(
    #             y_hat[i : i + 1],
    #             params[i : i + 1],
    #             y_height,
    #             y_width,
    #             kernel_size,
    #             padding,
    #         )
    #         y_strings.append(string)

    #     return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    # def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
    #     cdf = self.gaussian_conditional.quantized_cdf.tolist()
    #     cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
    #     offsets = self.gaussian_conditional.offset.tolist()

    #     encoder = BufferedRansEncoder()
    #     symbols_list = []
    #     indexes_list = []

    #     # Warning, this is slow...
    #     # TODO: profile the calls to the bindings...
    #     masked_weight = self.context_prediction.weight * self.context_prediction.mask
    #     for h in range(height):
    #         for w in range(width):
    #             y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
    #             ctx_p = F.conv2d(
    #                 y_crop,
    #                 masked_weight,
    #                 bias=self.context_prediction.bias,
    #             )

    #             # 1x1 conv for the entropy parameters prediction network, so
    #             # we only keep the elements in the "center"
    #             p = params[:, :, h : h + 1, w : w + 1]
    #             gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
    #             gaussian_params = gaussian_params.squeeze(3).squeeze(2)
    #             scales_hat, means_hat = gaussian_params.chunk(2, 1)

    #             indexes = self.gaussian_conditional.build_indexes(scales_hat)

    #             y_crop = y_crop[:, :, padding, padding]
    #             y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
    #             y_hat[:, :, h + padding, w + padding] = y_q + means_hat

    #             symbols_list.extend(y_q.squeeze().tolist())
    #             indexes_list.extend(indexes.squeeze().tolist())

    #     encoder.encode_with_indexes(
    #         symbols_list, indexes_list, cdf, cdf_lengths, offsets
    #     )

    #     string = encoder.flush()
    #     return string

    # def decompress(self, strings, shape):
    #     assert isinstance(strings, list) and len(strings) == 2

    #     if next(self.parameters()).device != torch.device("cpu"):
    #         warnings.warn(
    #             "Inference on GPU is not recommended for the autoregressive "
    #             "models (the entropy coder is run sequentially on CPU).",
    #             stacklevel=2,
    #         )

    #     # FIXME: we don't respect the default entropy coder and directly call the
    #     # range ANS decoder

    #     z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    #     params = self.h_s(z_hat)

    #     s = 4  # scaling factor between z and y
    #     kernel_size = 5  # context prediction kernel size
    #     padding = (kernel_size - 1) // 2

    #     y_height = z_hat.size(2) * s
    #     y_width = z_hat.size(3) * s

    #     # initialize y_hat to zeros, and pad it so we can directly work with
    #     # sub-tensors of size (N, C, kernel size, kernel_size)
    #     y_hat = torch.zeros(
    #         (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
    #         device=z_hat.device,
    #     )

    #     for i, y_string in enumerate(strings[0]):
    #         self._decompress_ar(
    #             y_string,
    #             y_hat[i : i + 1],
    #             params[i : i + 1],
    #             y_height,
    #             y_width,
    #             kernel_size,
    #             padding,
    #         )

    #     y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
    #     return y_hat

    # def _decompress_ar(
    #     self, y_string, y_hat, params, height, width, kernel_size, padding
    # ):
    #     cdf = self.gaussian_conditional.quantized_cdf.tolist()
    #     cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
    #     offsets = self.gaussian_conditional.offset.tolist()

    #     decoder = RansDecoder()
    #     decoder.set_stream(y_string)

    #     # Warning: this is slow due to the auto-regressive nature of the
    #     # decoding... See more recent publication where they use an
    #     # auto-regressive module on chunks of channels for faster decoding...
    #     for h in range(height):
    #         for w in range(width):
    #             # only perform the 5x5 convolution on a cropped tensor
    #             # centered in (h, w)
    #             y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
    #             ctx_p = F.conv2d(
    #                 y_crop,
    #                 self.context_prediction.weight,
    #                 bias=self.context_prediction.bias,
    #             )
    #             # 1x1 conv for the entropy parameters prediction network, so
    #             # we only keep the elements in the "center"
    #             p = params[:, :, h : h + 1, w : w + 1]
    #             gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
    #             scales_hat, means_hat = gaussian_params.chunk(2, 1)

    #             indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #             rv = decoder.decode_stream(
    #                 indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
    #             )
    #             rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
    #             rv = self.gaussian_conditional.dequantize(rv, means_hat)

    #             hp = h + padding
    #             wp = w + padding
    #             y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


class CompressAIJointAutoregressiveGaussianConditionalCoder(CompressAIGaussianConditionalCoder):
    def __init__(self, in_channels=192, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        M = self.in_channels

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

    def forward(self, y, *args, prior=None, **kwargs):
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((prior[..., :y.shape[-2], :y.shape[-1]], ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        entropy = -torch.log(y_likelihoods).sum()
        if self.training:
            # NOTE: we follow most works using bits as rate loss
            loss_rate = (entropy / math.log(2)) if self.use_bit_rate_loss else entropy
            self.update_cache("loss_dict", 
                loss_rate = loss_rate / y.shape[0] # normalize by batch size
            )
        self.update_cache("metric_dict",
            prior_entropy = entropy / y.shape[0], # normalize by batch size
        )
        return y_hat
        
    def encode(self, y, *args, prior=None, **kwargs):
        # if next(self.parameters()).device != torch.device("cpu"):
        #     warnings.warn(
        #         "Inference on GPU is not recommended for the autoregressive "
        #         "models (the entropy coder is run sequentially on CPU).",
        #         stacklevel=2,
        #     )

        params = prior

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = prior.shape[-2]
        y_width = prior.shape[-1]

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # y_strings = []
        # for i in range(y.size(0)):
        #     string = self._compress_ar(
        #         y_hat[i : i + 1],
        #         params[i : i + 1],
        #         y_height,
        #         y_width,
        #         kernel_size,
        #         padding,
        #     )
        #     y_strings.append(string)

        y_strings = self._compress_ar(
            y_hat,
            params,
            y_height,
            y_width,
            kernel_size,
            padding,
        )

        return y_strings

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decode(self, byte_string, *args, prior=None, **kwargs):
        # assert isinstance(byte_string, list)

        # if next(self.parameters()).device != torch.device("cpu"):
        #     warnings.warn(
        #         "Inference on GPU is not recommended for the autoregressive "
        #         "models (the entropy coder is run sequentially on CPU).",
        #         stacklevel=2,
        #     )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        params = prior

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = prior.shape[-2]
        y_width = prior.shape[-1]

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (prior.size(0), self.in_channels, y_height + 2 * padding, y_width + 2 * padding),
            device=prior.device,
        )

        # for i, y_string in enumerate(byte_string):
        #     self._decompress_ar(
        #         y_string,
        #         y_hat[i : i + 1],
        #         params[i : i + 1],
        #         y_height,
        #         y_width,
        #         kernel_size,
        #         padding,
        #     )

        self._decompress_ar(
            byte_string,
            y_hat,
            params,
            y_height,
            y_width,
            kernel_size,
            padding,
        )
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(p.shape[0], -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
