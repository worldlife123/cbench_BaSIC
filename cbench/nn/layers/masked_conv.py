# from compressai
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

__all__ = [
    "MaskedConv2d", "MaskedConv3d"
]

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B", "Checkerboard"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        if mask_type in ("A", "B"):
            self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
            self.mask[:, :, h // 2 + 1 :] = 0
        else:
            # checkerboard
            for i in range(h):
                for j in range(w):
                    if (i+j) % 2 == 0:
                        self.mask[:, :, i, j] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class MaskedConv3d(nn.Conv3d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, c, h, w = self.mask.size()
        self.mask[:, :, c // 2, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, c // 2, h // 2 + 1 :] = 0
        self.mask[:, :, c // 2 + 1 :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

import time
# TODO: confirm if this implementation has the same flops as nn.Conv2d (suppose to be a bit larger because of input mask)
class TopoGroupDynamicMaskConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 dynamic_channel_groups : int = 1, # TODO: deprecate?
                 allow_same_topogroup_conv=False, 
                 allow_continuous_topo_groups=False,
                 continuous_topo_groups_training_use_uniform_noise=False,
                 continuous_topo_groups_smooth_func="st",
                 detach_context_model=False,
                 **kwargs):
        kwargs.update(groups=1) # override group convolution (is this needed?)
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.dynamic_channel_groups = dynamic_channel_groups
        self.allow_same_topogroup_conv = allow_same_topogroup_conv
        self.allow_continuous_topo_groups = allow_continuous_topo_groups
        self.continuous_topo_groups_training_use_uniform_noise = continuous_topo_groups_training_use_uniform_noise
        self.continuous_topo_groups_smooth_func = continuous_topo_groups_smooth_func
        self.detach_context_model = detach_context_model
        self._meta = dict()
        # self._dynamic_kernel_weight = None # support for nn.Sequential
        # self._dynamic_kernel_bias = None # support for nn.Sequential

    def set_topo_groups(self, topo_groups : torch.Tensor):
        self._meta.update(_topo_groups=topo_groups)

    def set_channel_group_mask(self, channel_group_mask : torch.LongTensor):
        self._meta.update(_channel_group_mask=channel_group_mask)

    def set_dynamic_kernel(self, dynamic_kernel_weight, dynamic_kernel_bias):
        # self._dynamic_kernel_weight = dynamic_kernel_weight
        # self._dynamic_kernel_bias = dynamic_kernel_bias
        self._meta.update(_dynamic_kernel_weight=dynamic_kernel_weight, _dynamic_kernel_bias=dynamic_kernel_bias)

    # TODO: define input_mask for iterative maskconv
    def forward(self, input: torch.Tensor, topo_groups : torch.Tensor = None, 
                input_mask : torch.BoolTensor = None, 
                channel_group_mask : torch.BoolTensor = None, 
                dynamic_kernel_weight : torch.Tensor = None, 
                dynamic_kernel_bias : torch.Tensor = None, 
                **kwargs) -> torch.Tensor:
        batch_size = input.shape[0]
        input_unfold_group = F.unfold(input, self.kernel_size, padding=self.padding).unsqueeze(1)
        if topo_groups is None:
            topo_groups = self._meta.get("_topo_groups")
        if channel_group_mask is None:
            channel_group_mask = self._meta.get("_channel_group_mask")
        if dynamic_kernel_weight is None:
            dynamic_kernel_weight = self._meta.get("_dynamic_kernel_weight") # self._dynamic_kernel_weight
        if dynamic_kernel_bias is None:
            dynamic_kernel_bias = self._meta.get("_dynamic_kernel_bias") # self._dynamic_kernel_bias
        # start_time = time.time()
        if topo_groups is not None:
            topo_groups_batch_size = topo_groups.shape[0]
            input_channel_groups = topo_groups.shape[1]
            # assert topo_groups.shape[1] == dynamic_channel_groups
            # Note: as padding is zero, we make padded values the largest topogroup so that they are excluded from maskconv
            topo_groups = topo_groups.type_as(input)
            topo_groups_offset = topo_groups - topo_groups.max().ceil() - 1
            # [batch_size, self.channel_groups, self.in_channels // input_channel_groups * self.kernel_size[0] * self.kernel_size[1], spatial_size]
            # if self.training and self.allow_continuous_topo_groups and self.continuous_topo_groups_training_use_uniform_noise:
            #     topo_groups_offset = topo_groups_offset + torch.empty_like(topo_groups_offset).uniform_(-0.5, 0.5)
            topo_groups_center_group = topo_groups_offset.reshape(topo_groups_batch_size, topo_groups.shape[1], 1, -1)
            topo_groups_2d_unfold_group = F.unfold(topo_groups_offset, self.kernel_size, padding=self.padding).unsqueeze(1)
                # .repeat(1, 1, self.channel_groups * self.kernel_size[0] * self.kernel_size[1], 1).contiguous()
            if self.training and self.allow_continuous_topo_groups:
                topo_groups_2d_unfold_mask_group_hard = (topo_groups_center_group.round() - topo_groups_2d_unfold_group.round() + 1) \
                    if self.allow_same_topogroup_conv else (topo_groups_center_group.round() - topo_groups_2d_unfold_group.round())
                topo_groups_2d_unfold_mask_group = (topo_groups_center_group - topo_groups_2d_unfold_group + 1) \
                    if self.allow_same_topogroup_conv else (topo_groups_center_group - topo_groups_2d_unfold_group)
                if self.training and self.continuous_topo_groups_training_use_uniform_noise:
                    topo_groups_2d_unfold_mask_group = topo_groups_2d_unfold_mask_group + torch.empty_like(topo_groups_2d_unfold_mask_group).uniform_(-0.5, 0.5)
                topo_groups_2d_unfold_mask_group_hard = topo_groups_2d_unfold_mask_group_hard > 0.5
                if self.continuous_topo_groups_smooth_func == "hard":
                    topo_groups_2d_unfold_mask_group = topo_groups_2d_unfold_mask_group_hard
                elif self.continuous_topo_groups_smooth_func == "st":
                    topo_groups_2d_unfold_mask_group = topo_groups_2d_unfold_mask_group_hard + topo_groups_2d_unfold_mask_group - topo_groups_2d_unfold_mask_group.detach()
                elif self.continuous_topo_groups_smooth_func == "st-hardtanh":
                    topo_groups_2d_unfold_mask_group = F.hardtanh(topo_groups_2d_unfold_mask_group, 0., 1., inplace=True)
                    topo_groups_2d_unfold_mask_group = topo_groups_2d_unfold_mask_group_hard + topo_groups_2d_unfold_mask_group - topo_groups_2d_unfold_mask_group.detach()
                elif self.continuous_topo_groups_smooth_func == "st-sigmoid":
                    topo_groups_2d_unfold_mask_group = F.sigmoid(topo_groups_2d_unfold_mask_group - 0.5)
                    topo_groups_2d_unfold_mask_group = topo_groups_2d_unfold_mask_group_hard + topo_groups_2d_unfold_mask_group - topo_groups_2d_unfold_mask_group.detach()
                elif self.continuous_topo_groups_smooth_func == "reinmax-hardtanh":
                    topo_groups_2d_unfold_mask_group_pi0 = F.hardtanh(topo_groups_2d_unfold_mask_group, 0., 1., inplace=True)
                    topo_groups_2d_unfold_mask_group_pi1 = F.hardtanh(
                        (torch.log((topo_groups_2d_unfold_mask_group_pi0 + topo_groups_2d_unfold_mask_group_hard) / 2) - topo_groups_2d_unfold_mask_group).detach() + topo_groups_2d_unfold_mask_group,
                        0., 1., inplace=True)
                    topo_groups_2d_unfold_mask_group = 2 * topo_groups_2d_unfold_mask_group_pi1 - topo_groups_2d_unfold_mask_group_pi0 / 2
                    topo_groups_2d_unfold_mask_group = topo_groups_2d_unfold_mask_group_hard + topo_groups_2d_unfold_mask_group - topo_groups_2d_unfold_mask_group.detach()
                elif self.continuous_topo_groups_smooth_func == "hardtanh":
                    topo_groups_2d_unfold_mask_group = F.hardtanh(topo_groups_2d_unfold_mask_group, 0., 1., inplace=True)
                elif self.continuous_topo_groups_smooth_func == "hardtanh-0.5":
                    topo_groups_2d_unfold_mask_group = F.hardtanh((topo_groups_2d_unfold_mask_group-0.5)*2, 0., 1., inplace=True)
                elif self.continuous_topo_groups_smooth_func == "tanh5":
                    topo_groups_2d_unfold_mask_group = F.tanh(F.relu(topo_groups_2d_unfold_mask_group-0.5)*5)
                elif self.continuous_topo_groups_smooth_func == "tanh20":
                    topo_groups_2d_unfold_mask_group = F.tanh(F.relu(topo_groups_2d_unfold_mask_group-0.5)*20)
                else:
                    raise NotImplementedError(f"Unknown self.continuous_topo_groups_smooth_func {self.continuous_topo_groups_smooth_func}.")
            else:
                topo_groups_2d_unfold_mask_group = (topo_groups_2d_unfold_group <= topo_groups_center_group) \
                    if self.allow_same_topogroup_conv else (topo_groups_2d_unfold_group < topo_groups_center_group)
            topo_groups_2d_unfold_input_mask_group = topo_groups_2d_unfold_mask_group.reshape(topo_groups_batch_size, input_channel_groups, input_channel_groups, self.kernel_size[0] * self.kernel_size[1], -1)\
                .repeat(1, 1, 1, self.in_channels // input_channel_groups, 1)\
                .reshape(topo_groups_batch_size, input_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1], -1)
            # channel group mask
            if channel_group_mask is not None:
                topo_groups_2d_unfold_input_mask_group = topo_groups_2d_unfold_input_mask_group[:, channel_group_mask]
            # assert topo_groups_2d_unfold_input_mask_group.shape[1] == self.dynamic_channel_groups,\
            #     f"channel_group_mask {channel_group_mask} invalid! Should produce {self.dynamic_channel_groups} channel groups!"
            dynamic_channel_groups = topo_groups_2d_unfold_input_mask_group.shape[1]
            # align batch size
            if topo_groups_batch_size != 1 and topo_groups_batch_size != batch_size:
                topo_groups_2d_unfold_input_mask_group = topo_groups_2d_unfold_input_mask_group.repeat(batch_size // topo_groups_batch_size, 1, 1, 1)
            # print("mask_pre", time.time() - start_time)
            # start_time = time.time()
            # [batch_size, input_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1], spatial_size]
            input_conv_masked = input_unfold_group * topo_groups_2d_unfold_input_mask_group
            # print("mask_mul", time.time() - start_time)
            # start_time = time.time()
        else:
            dynamic_channel_groups = self.dynamic_channel_groups
            input_conv_masked = input_unfold_group
        # [batch_size, dynamic_channel_groups, self.out_channels // dynamic_channel_groups, spatial_size]
        if dynamic_kernel_weight is not None:
            if topo_groups_batch_size != 1 and topo_groups_batch_size != batch_size:
                dynamic_kernel_weight = dynamic_kernel_weight.repeat(batch_size // topo_groups_batch_size, 1, 1, 1, 1)
            if self.detach_context_model:
                dynamic_kernel_weight = dynamic_kernel_weight.detach()
            dynamic_kernel_weight = dynamic_kernel_weight.reshape(dynamic_kernel_weight.shape[0], dynamic_channel_groups, self.out_channels // dynamic_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1], -1)
            # NOTE: spatial dynamic (cost much more memory!)
            if dynamic_kernel_weight.shape[-1] != 1:
                output_unfold_group = (dynamic_kernel_weight.reshape(dynamic_kernel_weight.shape[0], dynamic_channel_groups, self.out_channels // dynamic_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1], -1)\
                                    * input_conv_masked.unsqueeze(2)).sum(-2)
                # output_unfold_group = dynamic_kernel_weight.reshape(dynamic_kernel_weight.shape[0], dynamic_channel_groups, self.out_channels // dynamic_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1], -1).movedim(-1,1)\
                #                       .matmul(input_conv_masked.movedim(-1,1).unsqueeze(-1)).movedim(1, -1)
            else:
                output_unfold_group = dynamic_kernel_weight.reshape(dynamic_kernel_weight.shape[0], dynamic_channel_groups, self.out_channels // dynamic_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1])\
                    .matmul(input_conv_masked)
        else:
            weight = self.weight
            if self.detach_context_model:
                weight = weight.detach()
            output_unfold_group = weight.reshape(1, dynamic_channel_groups, self.out_channels // dynamic_channel_groups, self.in_channels * self.kernel_size[0] * self.kernel_size[1])\
                .matmul(input_conv_masked)
        if dynamic_kernel_bias is not None:
            if topo_groups_batch_size != 1 and topo_groups_batch_size != batch_size:
                dynamic_kernel_bias = dynamic_kernel_bias.repeat(batch_size // topo_groups_batch_size, 1, 1, 1)
            if self.detach_context_model:
                dynamic_kernel_bias = dynamic_kernel_bias.detach()
            dynamic_kernel_bias = dynamic_kernel_bias.reshape(dynamic_kernel_bias.shape[0], dynamic_channel_groups, self.out_channels // dynamic_channel_groups, -1)
            output_unfold_group = output_unfold_group + dynamic_kernel_bias
        elif self.bias is not None:
            bias = self.bias
            if self.detach_context_model:
                bias = bias.detach()
            output_unfold_group = output_unfold_group + bias.reshape(1, dynamic_channel_groups, self.out_channels // dynamic_channel_groups, 1)
        output = output_unfold_group.reshape(batch_size, self.out_channels, *input.shape[2:])
        # print("conv", time.time() - start_time)
        # start_time = time.time()
        return output


class TopoGroupDynamicMaskConv2dContextModel(nn.Module):
    def __init__(self, in_channels=192, out_channels=384, kernel_size=5,
                 use_param_merger=True,
                 param_merger_in_channels=None,
                 param_merger_mid_channels_list=None,
                 param_merger_kernel_size=1,
                 allow_continuous_topo_groups=False,
                 continuous_topo_groups_smooth_func="st",
                 detach_context_model=False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.allow_continuous_topo_groups = allow_continuous_topo_groups
        self.continuous_topo_groups_smooth_func = continuous_topo_groups_smooth_func
        self.detach_context_model = detach_context_model

        self.context_prediction = TopoGroupDynamicMaskConv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, 
                                                             allow_continuous_topo_groups=self.allow_continuous_topo_groups,
                                                             continuous_topo_groups_smooth_func=self.continuous_topo_groups_smooth_func,
                                                             detach_context_model=self.detach_context_model, **kwargs)

        self.use_param_merger = use_param_merger
        if self.use_param_merger:
            self.param_merger_out_channels = self.out_channels
            self.param_merger_in_channels = self.param_merger_out_channels * 2 if param_merger_in_channels is None else param_merger_in_channels
            self.param_merger_mid_channels_list = [self.param_merger_out_channels * 5 // 3, self.param_merger_out_channels * 4 // 3] \
                if param_merger_mid_channels_list is None else param_merger_mid_channels_list
            self.param_merger_in = TopoGroupDynamicMaskConv2d(self.param_merger_in_channels, self.param_merger_mid_channels_list[0], param_merger_kernel_size, 
                                                              allow_same_topogroup_conv=True,
                                                              allow_continuous_topo_groups=self.allow_continuous_topo_groups,
                                                              continuous_topo_groups_smooth_func=self.continuous_topo_groups_smooth_func,
                                                              detach_context_model=self.detach_context_model, **kwargs)
            param_merger_out = []
            for i in range(len(self.param_merger_mid_channels_list)-1):
                param_merger_out.extend([
                    nn.LeakyReLU(inplace=True),
                    TopoGroupDynamicMaskConv2d(self.param_merger_mid_channels_list[i], self.param_merger_mid_channels_list[i+1], param_merger_kernel_size,
                                            allow_same_topogroup_conv=True,
                                            allow_continuous_topo_groups=self.allow_continuous_topo_groups,
                                            continuous_topo_groups_smooth_func=self.continuous_topo_groups_smooth_func,
                                            detach_context_model=self.detach_context_model, **kwargs),
                ])
            self.param_merger_out = nn.Sequential(
                *param_merger_out,
                nn.LeakyReLU(inplace=True),
                TopoGroupDynamicMaskConv2d(self.param_merger_mid_channels_list[-1], self.param_merger_out_channels, param_merger_kernel_size,
                                           allow_same_topogroup_conv=True,
                                           allow_continuous_topo_groups=self.allow_continuous_topo_groups,
                                           continuous_topo_groups_smooth_func=self.continuous_topo_groups_smooth_func,
                                           detach_context_model=self.detach_context_model, **kwargs),
            )

    def _merge_prior_params(self, pgm_params : torch.Tensor, pgm : torch.LongTensor, prior_params : torch.Tensor = None) -> torch.Tensor:
        if self.use_param_merger:
            concat_params = torch.cat([pgm_params, prior_params], dim=1)
            concat_pgms = torch.cat([pgm, torch.zeros_like(pgm) - 1], dim=1)# assign -1 topo group for prior
            channel_group_mask = [True] * pgm.shape[1] + [False] * pgm.shape[1]
            merged_params = self.param_merger_in(concat_params, concat_pgms, channel_group_mask=channel_group_mask)
            for layer in self.param_merger_out:
                if isinstance(layer, TopoGroupDynamicMaskConv2d):
                    layer.set_topo_groups(pgm)
            merged_params = self.param_merger_out(merged_params)
        else:
            # TODO: dist specific add?
            merged_params = pgm_params + prior_params if prior_params is not None else pgm_params
        return merged_params

    def forward(self, input, pgm, prior=None):
        pgm_params = self.context_prediction(input, pgm)
        merged_params = self._merge_prior_params(pgm_params, pgm=pgm, prior_params=prior)
        return merged_params