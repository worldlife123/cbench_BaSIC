import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Dict, List, Any
import math
import random, functools
import numpy as np

from ..base import NNTrainableModule, DynamicNNTrainableModule
from .slimmable_layers import BaseSlimmableLayer, DynamicConv2d, DynamicGDN, \
    DynamicResidualBlock, DynamicResidualBlockUpsample, DynamicResidualBlockWithStride, DynamicResidualBottleneckBlock, DynamicAttentionBlock,\
    dynamic_conv3x3, dynamic_subpel_conv3x3, GroupedDynamicSpatialFrequencyModulationAdaptor
from cbench.codecs.base import VariableComplexityCodecInterface

# from compressai.layers import GDN
from compressai.ops.parametrizers import NonNegativeParametrizer

from thop.vision.basic_hooks import count_convNd, count_relu
from thop.vision.calc_func import calculate_conv2d_flops


class GroupGDN(NNTrainableModule):
    def __init__(
        self,
        in_channels: int,
        groups: int = 1,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.groups = groups

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels // self.groups).repeat(self.groups, 1)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C // self.groups, 1, 1)
        norm = F.conv2d(x**2, gamma, beta, groups=self.groups)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


def count_gdn(m: GroupGDN, x, y: torch.Tensor):
    x = x[0]

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.gamma.shape) + [1, 1],
        groups = m.groups if hasattr(m, "groups") else 1,
    )

# Bipartite PGM
class BasePGMLayer(DynamicNNTrainableModule):
    def __init__(self, in_nodes, out_nodes, *args, 
                 bottleneck_nodes=None,
                 pgm_logits_format="bernoulli", # bernoulli, categorical
                 default_pgm_init_logits=0.0,
                 default_pgm_limit=0.0,
                 pgm_num_categories=1,
                 freeze_pgm=False,
                 enable_dynamic_pgm_limit=False,
                 enable_default_input_mask=False,
                 **kwargs):
        super().__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        # self.num_edges = in_nodes * out_nodes
        self.bottleneck_nodes = bottleneck_nodes

        self.pgm_logits_format = pgm_logits_format
        self.pgm_num_categories = pgm_num_categories
        self.freeze_pgm = freeze_pgm

        if self.pgm_logits_format == "bernoulli":
            default_pgm = torch.zeros(in_nodes, out_nodes if bottleneck_nodes is None else bottleneck_nodes) + default_pgm_init_logits
            # default_pgm[:] = -default_pgm_init_logits
            # default_pgm[0, 0] = default_pgm_init_logits
            default_agg_pgm = torch.zeros(out_nodes if bottleneck_nodes is None else bottleneck_nodes) + default_pgm_init_logits
        elif self.pgm_logits_format == "categorical":
            default_pgm = torch.zeros(in_nodes, out_nodes if bottleneck_nodes is None else bottleneck_nodes, self.pgm_num_categories) + torch.as_tensor(default_pgm_init_logits)
            default_agg_pgm = torch.zeros(out_nodes if bottleneck_nodes is None else bottleneck_nodes, self.pgm_num_categories) + torch.as_tensor(default_pgm_init_logits) # TODO: different categories for aggregator
        else:
            raise NotImplementedError(f"Unknown pgm_logits_format {self.pgm_logits_format}")

        self.default_pgm = nn.Parameter(default_pgm)

        # if bottleneck_nodes is not None:
        #     default_pgm_bout = torch.zeros(out_nodes, bottleneck_nodes) + default_pgm_init_logits
        #     self.pgm_bout = nn.Parameter(default_pgm_bout)

        self.default_agg_pgm = nn.Parameter(default_agg_pgm)

        self.enable_default_input_mask = enable_default_input_mask
        if enable_default_input_mask:
            self.default_input_mask = nn.Parameter(torch.zeros(in_nodes) + default_pgm_init_logits)

        self.enable_dynamic_pgm_limit = enable_dynamic_pgm_limit
        if enable_dynamic_pgm_limit:
            self.register_dynamic_parameter("pgm_limit", torch.zeros(1) + default_pgm_limit)
        else:
            self.pgm_limit = default_pgm_limit

        self.pgm_model = self.build_pgm_model()

    def build_pgm_model(self) -> nn.Module:
        raise NotImplementedError()

    def get_pgm_weights(self, pgm_logits):
        if self.pgm_logits_format == "bernoulli":
            if self.training and not self.freeze_pgm:
                # sample from pgm
                if self.enable_dynamic_pgm_limit:
                    rand_limit = torch.rand(1).clamp(min=0.01, max=0.99).item()
                    # pgm_logits = pgm_logits - math.log(rand_limit / (1 - rand_limit)) # inverse sigmoid
                    # solve (1 / 1 + e^(-pgm_logits) + e^pgm_limit)).mean() == rand(0, 1) (hard!)
                    # pgm_limit = torch.log(torch.sigmoid(pgm_logits).sum())
                    pgm_logits = pgm_logits - rand_limit * (pgm_logits.max() - pgm_logits.min()) - pgm_logits.min()
                else:
                    pgm_logits = pgm_logits - self.pgm_limit
                pgm_weights = D.RelaxedBernoulli(0.5, logits=pgm_logits).rsample()
            else:
                pgm_weights = (pgm_logits > self.pgm_limit).float()
            self.update_cache("moniter_dict", pgm_weights_mean=pgm_weights.mean())
        elif self.pgm_logits_format == "categorical":
            if self.training and not self.freeze_pgm:
                # sample from pgm
                pgm_weights = D.RelaxedOneHotCategorical(0.5, logits=pgm_logits).rsample()
            else:
                pgm_weights = F.one_hot(pgm_logits.argmax(-1), pgm_logits.shape[-1]).type_as(pgm_logits)
            # self.update_cache("moniter_dict", pgm_weights_mean=pgm_weights.mean())
        else:
            raise NotImplementedError(f"Unknown pgm_logits_format {self.pgm_logits_format}")

        return pgm_weights

    def get_bottleneck_out_weights_from_pgm(self, pgm=None, **kwargs):
        if pgm is None:
            pgm = self.default_pgm
        input_weights = pgm.sum(0).unsqueeze(-1)
        return input_weights / input_weights.sum()

    def forward_bottleneck_in(self, input, *args, pgm=None, input_mask=None, **kwargs):
        raise NotImplementedError()

    def forward_bottleneck_out(self, bottleneck, *args, pgm=None, input_mask=None, **kwargs):
        raise NotImplementedError()

    def forward(self, input, *args, pgm=None, input_mask=None, **kwargs):
        # raise NotImplementedError()
        output = self.forward_bottleneck_in(input, *args, pgm=pgm, input_mask=input_mask, **kwargs)
        if self.bottleneck_nodes is not None:
            output = self.forward_bottleneck_out(output, *args, pgm=pgm, input_mask=input_mask, **kwargs)
        return output

class GroupConv2dPGMModel(BasePGMLayer):
    def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, 
                 bottleneck_groups=None,
                 bottleneck_channels_per_group=None,
                 mid_channels_per_group=16,
                 num_layers=3,
                 kernel_size=5,
                 use_skip_model=False,
                 use_pre_aggregate=False,
                 use_aggregator=False,
                 aggregator_in_channels_per_group=None,
                 aggregator_num_layers=2,
                 aggregator_act_func="leakyrelu",
                 **kwargs):
        kwargs.update(bottleneck_nodes=bottleneck_groups)
        self.in_channels = in_channels
        self.in_groups = in_groups
        self.in_channels_per_group = in_channels // in_groups
        assert self.in_channels_per_group * self.in_groups == self.in_channels
        self.out_channels = out_channels
        self.out_groups = out_groups # if bottleneck_groups is None else bottleneck_groups
        self.out_channels_per_group = out_channels // self.out_groups
        assert self.out_channels_per_group * self.out_groups == self.out_channels
        self.bottleneck_groups = bottleneck_groups if bottleneck_groups is not None else self.out_groups
        self.bottleneck_channels_per_group = bottleneck_channels_per_group if bottleneck_channels_per_group is not None else self.out_channels_per_group
        self.bottleneck_channels = self.bottleneck_groups * self.bottleneck_channels_per_group
        self.mid_channels_per_group = mid_channels_per_group

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.use_skip_model = use_skip_model
        self.use_pre_aggregate = use_pre_aggregate
        self.use_aggregator = use_aggregator
        self.aggregator_in_channels_per_group = aggregator_in_channels_per_group
        self.aggregator_num_layers = aggregator_num_layers
        self.aggregator_act_func = aggregator_act_func

        if self.use_pre_aggregate:
            self.mid_channels_total = mid_channels_per_group * self.bottleneck_groups
            self.mid_groups = self.bottleneck_groups
            self.pgm_input_channels = self.in_channels_per_group * self.bottleneck_groups
            self.pgm_output_channels = self.bottleneck_channels
        else:
            self.mid_channels_total = mid_channels_per_group * in_groups * self.bottleneck_groups
            self.mid_groups = in_groups * self.bottleneck_groups
            self.pgm_input_channels = self.in_channels * self.bottleneck_groups
            self.pgm_output_channels = self.bottleneck_channels * self.in_groups

        # kwargs.update(
        #     pgm_logits_format = "categorical" if self.use_skip_model else "bernoulli",
        #     pgm_num_categories = 2 if self.use_skip_model else 1,
        # )
        super().__init__(in_groups, out_groups, *args, **kwargs)

        if self.use_aggregator:
            if not self.use_pre_aggregate and self.aggregator_in_channels_per_group is not None:
                self.pgm_output_channels = self.aggregator_in_channels_per_group * self.in_groups
            self.pgm_agg = self.build_pgm_aggregator()

        self.pgm_model = self.build_pgm_model() # recreate model as pgm_output_channels may be changed

        if self.use_skip_model:
            self.pgm_skip_model = self.build_pgm_skip_model()

        if self.bottleneck_nodes is not None:
            self.pgm_out_model = self.build_pgm_out_model()

        if self.freeze_pgm:
            for param in self.pgm_model:
                param.lr_modifier = 0.0

        # register flop hooks
        flop_models = [self.pgm_model, ]
        if self.use_aggregator:
            flop_models.append(self.pgm_agg)
        if self.bottleneck_nodes is not None:
            flop_models.append(self.pgm_out_model)
        for model in flop_models:
            for module in model.modules():
                module.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    module.register_forward_hook(count_convNd)
                elif isinstance(module, (nn.LeakyReLU, )):
                    module.register_forward_hook(count_relu)
                elif isinstance(module, (GroupGDN, )):
                    module.register_forward_hook(count_gdn)
        self.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)

    def get_current_flops(self, input=None):
        return self.total_ops.squeeze().clone()

    def get_nn_complexity(self, input=None, metric=None):
        # TODO: implement other metrics. By default we return total_ops regardless of metrics
        return self.total_ops.squeeze().clone()

    def build_pgm_model(self):
        # PGM model definition. This is a simple multilayer FCN
        # pgm_model = [
        #     nn.Conv2d(self.in_channels_per_group, self.mid_channels_per_group, self.kernel_size, padding=(self.kernel_size // 2)),
        #     nn.ReLU(),
        # ]
        # for _ in range(self.num_layers):
        #     pgm_model.append(nn.Conv2d(self.mid_channels_per_group, self.mid_channels_per_group, self.kernel_size, padding=(self.kernel_size // 2)))
        #     pgm_model.append(nn.ReLU())
        # pgm_model.append(nn.Conv2d(self.mid_channels_per_group, self.bottleneck_channels_per_group, self.kernel_size, padding=(self.kernel_size // 2)))

        # faster impl
        pgm_model = [
            nn.Conv2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.ReLU(),
        ]
        if self.num_layers > 2:
            for _ in range(self.num_layers-1):
                pgm_model.append(nn.Conv2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups))
                pgm_model.append(nn.ReLU())
        pgm_model.append(nn.Conv2d(self.mid_channels_total, self.pgm_output_channels, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups))
        return nn.Sequential(*pgm_model)

    def build_pgm_aggregator(self):
        if self.use_pre_aggregate:
            agg_input_channels = self.in_channels_per_group*self.in_groups*self.bottleneck_groups
            bottleneck_channels = self.pgm_input_channels
        else:
            agg_input_channels = self.pgm_output_channels
            bottleneck_channels = self.bottleneck_channels
        bottleneck_ratio = agg_input_channels // bottleneck_channels
        # act_func = None
        # if aggregator_act_func == "leakyrelu":
        #     act_func = functools.partial(nn.LeakyReLU, inplace=True)
        # elif aggregator_act_func == "gdn":
        #     act_func = functools.partial(GroupGDN, inverse=True)
        # else:
        #     raise NotImplementedError()
        if self.aggregator_num_layers > 1:
            bottleneck_ratio_step = (bottleneck_ratio - 1) / (self.aggregator_num_layers - 1)
            pgm_agg = [
                GroupGDN(agg_input_channels, groups=self.bottleneck_groups) if self.aggregator_act_func == "gdn" else nn.LeakyReLU(inplace=True),
                nn.Conv2d(agg_input_channels, int((bottleneck_ratio - bottleneck_ratio_step) * bottleneck_channels), 1, groups=self.bottleneck_groups),
            ]
            for i in range(1, self.aggregator_num_layers-1):
                pgm_agg.extend([
                    GroupGDN(int((bottleneck_ratio - i * bottleneck_ratio_step) * bottleneck_channels), groups=self.bottleneck_groups) if self.aggregator_act_func == "gdn"
                    else nn.LeakyReLU(inplace=True),
                    nn.Conv2d(
                        int((bottleneck_ratio - i * bottleneck_ratio_step) * bottleneck_channels, groups=self.bottleneck_groups), 
                        int((bottleneck_ratio - (i+1) * bottleneck_ratio_step) * bottleneck_channels, groups=self.bottleneck_groups), 
                        1),
                ])
            pgm_agg.append(
                GroupGDN(bottleneck_channels, groups=self.bottleneck_groups) if self.aggregator_act_func == "gdn"
                else nn.LeakyReLU(inplace=True),
            )
            pgm_agg.append(nn.Conv2d(bottleneck_channels, bottleneck_channels, 1, groups=self.bottleneck_groups))
        else:
            pgm_agg = [
                GroupGDN(agg_input_channels, groups=self.bottleneck_groups) if self.aggregator_act_func == "gdn" else nn.LeakyReLU(inplace=True),
            ]
            if self.aggregator_num_layers==1:
                pgm_agg.append(nn.Conv2d(agg_input_channels, bottleneck_channels, 1, groups=self.bottleneck_groups))
        return nn.Sequential(*pgm_agg)
    
    def build_pgm_out_model(self):
        pgm_model = []
        #     nn.Conv2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
        #     nn.ReLU(),
        # ]
        # if self.num_layers > 2:
        #     for _ in range(self.num_layers-1):
        #         pgm_model.append(nn.Conv2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups))
        #         pgm_model.append(nn.ReLU())
        pgm_model.append(nn.Conv2d(self.bottleneck_channels_per_group, self.out_channels, 1))
        return nn.Sequential(*pgm_model)
    
    def build_pgm_skip_model(self):
        return nn.Conv2d(self.pgm_input_channels, self.pgm_output_channels, 1, groups=self.mid_groups)

    def forward_bottleneck_in(self, input, *args, pgm=None, input_mask=None, **kwargs):
        # clear all ops counter
        for name, param in self.named_buffers():
            if "total_ops" in name:
                param.data.fill_(0)

        # if pgm is None:
        #     pgm = self.default_pgm
        pgm_weights = self.get_pgm_weights(self.default_pgm) if pgm is None else pgm
        if input_mask is None and self.enable_default_input_mask:
            input_mask = torch.sigmoid(self.default_input_mask)
        if input_mask is not None:
            pgm_weights = pgm_weights * input_mask.unsqueeze(-1)

        input_grouped = input.repeat(1, self.bottleneck_groups, 1, 1)
        if self.use_pre_aggregate:
            pgm_weights = pgm_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            batch_size, channels, height, width = input_grouped.shape
            input_weighted = input_grouped.reshape(batch_size, -1, self.in_groups, self.bottleneck_groups, height, width)\
                * pgm_weights
            
            # pre param aggregation
            if self.use_aggregator:
                input_grouped = self.pgm_agg(input_weighted.reshape(batch_size, channels, height, width))
                if self.aggregator_num_layers == 0:
                    input_grouped = input_grouped.reshape(batch_size, -1, self.in_groups, self.bottleneck_groups, height, width).mean(2)\
                        .reshape(batch_size, -1, height, width)
            else:
                # normalize by pgm
                # pgm_weights_norm = 1 / pgm_weights.sum(2)
                # pgm_weights_norm[pgm_weights_norm == 0] = 0
                input_grouped = input_weighted.mean(2).reshape(batch_size, -1, height, width) # * pgm_weights_norm

        output_grouped = self.pgm_model(input_grouped)

        if not self.use_pre_aggregate:
            pgm_weights = pgm_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            batch_size, channels, height, width = output_grouped.shape
            output_weighted = output_grouped.reshape(batch_size, -1, self.in_groups, self.bottleneck_groups, height, width)\
                * pgm_weights
            if self.use_skip_model:
                output_grouped_skip = self.pgm_skip_model(input_grouped)
                output_weighted = output_weighted + output_grouped_skip.reshape(batch_size, -1, self.in_groups, self.bottleneck_groups, height, width)\
                    * (1 - pgm_weights)
            
            # post param aggregation
            if self.use_aggregator:
                output = self.pgm_agg(output_weighted.reshape(batch_size, channels, height, width))
                if self.aggregator_num_layers == 0:
                    output = output.reshape(batch_size, -1, self.in_groups, self.bottleneck_groups, height, width).mean(2)\
                        .reshape(batch_size, self.bottleneck_channels, height, width)
            else:
                # normalize by pgm
                # pgm_weights_norm = 1 / pgm_weights.sum(2)
                # pgm_weights_norm[pgm_weights_norm == 0] = 0
                output = output_weighted.mean(2).reshape(batch_size, self.bottleneck_channels, height, width) # * pgm_weights_norm
        else:
            output = output_grouped

        # calculate flops
        total_ops = 0
        for module in self.pgm_model.modules():
            if "total_ops" in module._buffers:
                total_ops += module.total_ops 
        # TODO: categorical/bernoulli pgm?
        total_ops *= pgm_weights.sum() / pgm_weights.numel()

        self.update_cache("moniter_dict", pgm_model_ops=total_ops)

        # TODO: adjust agg/out flops according to pgm
        if self.use_aggregator:
            agg_ops = 0
            for module in self.pgm_agg.modules():
                if "total_ops" in module._buffers:
                    agg_ops += module.total_ops 

            self.update_cache("moniter_dict", pgm_agg_ops=agg_ops)
            total_ops += agg_ops
        
        if self.bottleneck_nodes is not None:
            out_ops = 0
            for module in self.pgm_out_model.modules():
                if "total_ops" in module._buffers:
                    out_ops += module.total_ops 

            self.update_cache("moniter_dict", pgm_out_ops=out_ops)
            total_ops += out_ops

        self.total_ops = total_ops

        return output
    
    def forward_bottleneck_out(self, bottleneck, *args, pgm=None, input_mask=None, **kwargs):
        output_weights = self.get_bottleneck_out_weights_from_pgm(pgm)
        output_weights = output_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        batch_size, channels, height, width = bottleneck.shape
        output_weighted = (bottleneck.reshape(batch_size, -1, self.bottleneck_groups, self.out_groups, height, width)\
            * output_weights).sum(2)
        output = self.pgm_out_model(output_weighted.reshape(batch_size, -1, height, width))
        return output



class HyperpriorAnalysisGroupConv2dPGMModel(GroupConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            nn.Conv2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups),
            nn.Conv2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups),
            nn.Conv2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups),
            nn.Conv2d(self.mid_channels_total, self.pgm_output_channels, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )

    def build_pgm_skip_model(self):
        return nn.Sequential(
            nn.Conv2d(self.pgm_input_channels, self.pgm_output_channels, self.kernel_size, stride=4, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.AvgPool2d(5, stride=4, padding=2),
        )

class HyperpriorSynthesisGroupConv2dPGMModel(GroupConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.pgm_output_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )

    def build_pgm_skip_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.pgm_output_channels, self.kernel_size, stride=4, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.Upsample(scale_factor=4),
        )

class HyperpriorSynthesisAggregateGDNGroupConv2dPGMModel(GroupConv2dPGMModel):
    def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=True, **kwargs):
        super().__init__(in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=use_aggregator, **kwargs)
    
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )
    
    def build_pgm_aggregator(self):
        if self.use_pre_aggregate:
            # agg_input_channels = self.in_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.pgm_input_channels
        else:
            # agg_input_channels = self.bottleneck_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.bottleneck_channels
        return nn.Sequential(
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, bottleneck_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
        )
    
    def build_pgm_skip_model(self):
        return nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups)

class HyperpriorSynthesisAggregateV2GDNGroupConv2dPGMModel(GroupConv2dPGMModel):
    def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=True, **kwargs):
        super().__init__(in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=use_aggregator, **kwargs)
    
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )
    
    def build_pgm_aggregator(self):
        if self.use_pre_aggregate:
            # agg_input_channels = self.in_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.pgm_input_channels
        else:
            # agg_input_channels = self.bottleneck_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.bottleneck_channels
        return nn.Sequential(
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, bottleneck_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
        )
    
    def build_pgm_skip_model(self):
        return nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=4, output_padding=3, padding=(self.kernel_size // 2), groups=self.mid_groups)

class HyperpriorSynthesisAggregateV2PreGDNGroupConv2dPGMModel(GroupConv2dPGMModel):
    def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=True, **kwargs):
        super().__init__(in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=use_aggregator, **kwargs)
    
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
        )

    def build_pgm_aggregator(self):
        if self.use_pre_aggregate:
            # agg_input_channels = self.in_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.pgm_input_channels
        else:
            # agg_input_channels = self.bottleneck_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.bottleneck_channels
        return nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, bottleneck_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
        )

    def build_pgm_skip_model(self):
        return nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=4, output_padding=3, padding=(self.kernel_size // 2), groups=self.mid_groups)

class HyperpriorSynthesisAggregateOutGDNGroupConv2dPGMModel(GroupConv2dPGMModel):
    def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=True, **kwargs):
        super().__init__(in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=use_aggregator, **kwargs)
    
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )
    
    def build_pgm_aggregator(self):
        if self.use_pre_aggregate:
            # agg_input_channels = self.in_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.pgm_input_channels
        else:
            # agg_input_channels = self.bottleneck_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.bottleneck_channels
        return nn.Sequential(
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, bottleneck_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
        )

    def build_pgm_out_model(self):
        return nn.Sequential(
            GroupGDN(self.bottleneck_channels_per_group, groups=1, inverse=True),
            nn.ConvTranspose2d(self.bottleneck_channels_per_group, self.out_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=1),
        )

    def build_pgm_skip_model(self):
        return nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=4, output_padding=3, padding=(self.kernel_size // 2), groups=self.mid_groups)

class HyperpriorSynthesisAggregateV3GDNGroupConv2dPGMModel(GroupConv2dPGMModel):
    def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=True, **kwargs):
        super().__init__(in_channels, in_groups, out_channels, out_groups, *args, use_aggregator=use_aggregator, **kwargs)
    
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )
    
    def build_pgm_aggregator(self):
        if self.use_pre_aggregate:
            # agg_input_channels = self.in_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.pgm_input_channels
        else:
            # agg_input_channels = self.bottleneck_channels_per_group*self.in_groups*self.out_groups
            bottleneck_channels = self.bottleneck_channels
        return nn.Sequential(
            GroupGDN(self.mid_channels_total, groups=self.bottleneck_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, bottleneck_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.bottleneck_groups),
        )

    def build_pgm_out_model(self):
        return nn.Sequential(
            GroupGDN(self.bottleneck_channels_per_group, groups=1, inverse=True),
            nn.Conv2d(self.bottleneck_channels_per_group, self.out_channels, 1),
        )

    def build_pgm_skip_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.pgm_output_channels, self.kernel_size, stride=4, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.Upsample(scale_factor=2),
        )

class HyperpriorSynthesisNoAggregateOutGDNGroupConv2dPGMModel(GroupConv2dPGMModel):
    # def __init__(self, in_channels, in_groups, out_channels, out_groups, *args, **kwargs):
    #     super().__init__(in_channels, in_groups, out_channels, out_groups, *args, **kwargs)
    
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.mid_channels_total, groups=self.mid_groups, inverse=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.pgm_output_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            GroupGDN(self.pgm_output_channels, groups=self.mid_groups, inverse=True),
        )

    def build_pgm_out_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.bottleneck_channels_per_group, self.bottleneck_channels_per_group, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2)),
            GroupGDN(self.bottleneck_channels_per_group, groups=1, inverse=True),
            nn.ConvTranspose2d(self.bottleneck_channels_per_group, self.out_channels, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2)),
        )


class HyperpriorHyperAnalysisGroupConv2dPGMModel(GroupConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            nn.Conv2d(self.pgm_input_channels, self.mid_channels_total, 3, stride=1, padding=1, groups=self.mid_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels_total, self.pgm_output_channels, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )


class HyperpriorHyperSynthesisGroupConv2dPGMModel(GroupConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels_total, self.pgm_output_channels, 3, stride=1, padding=1, groups=self.mid_groups),
            nn.ReLU(inplace=True),
        )


class MeanScaleHyperpriorHyperAnalysisGroupConv2dPGMModel(GroupConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            nn.Conv2d(self.pgm_input_channels, self.mid_channels_total, 3, stride=1, padding=1, groups=self.mid_groups),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.mid_channels_total, self.pgm_output_channels, self.kernel_size, stride=2, padding=(self.kernel_size // 2), groups=self.mid_groups),
        )


class MeanScaleHyperpriorHyperSynthesisGroupConv2dPGMModel(GroupConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.pgm_input_channels, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.mid_channels_total, self.mid_channels_total, self.kernel_size, stride=2, output_padding=1, padding=(self.kernel_size // 2), groups=self.mid_groups),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.mid_channels_total, self.pgm_output_channels, 3, stride=1, padding=1, groups=self.mid_groups),
        )


class SlimmableConv2dPGMModel(BasePGMLayer):
    def __init__(self, *args, 
                 in_channels=256,
                 in_groups=1,
                 out_channels=256,
                 mid_channels_list=[64, 128, 256], 
                 pgm_weight_one_hot_threshold=0.99,
                 training_self_distillation=False,
                 training_self_distillation_loss_type="L1",
                 use_sandwich_rule=False,
                 **kwargs):
        self.in_channels = in_channels
        self.in_groups = in_groups
        self.out_channels = out_channels
        self.mid_channels_list = mid_channels_list
        self.out_channels_list = self.out_channels if isinstance(self.out_channels, list) else [self.out_channels] * len(self.mid_channels_list)
        self.pgm_weight_one_hot_threshold = pgm_weight_one_hot_threshold
        self.training_self_distillation = training_self_distillation
        self.training_self_distillation_loss_type = training_self_distillation_loss_type
        self.use_sandwich_rule = use_sandwich_rule

        kwargs.update(
            pgm_logits_format = "categorical",
            pgm_num_categories = len(mid_channels_list),
        )
        super().__init__(1, 1, *args, **kwargs)

        # self.pgm_model = self.build_pgm_model()

        # flop counter 
        # for module in self.pgm_model.modules():
        #     module.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)
        #     if isinstance(module, (DynamicConv2d, )):
        #         module.conv.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)
        #         module.conv.register_forward_hook(count_convNd)
        #     # elif isinstance(module, (nn.LeakyReLU, )):
        #     #     module.register_forward_hook(count_relu)
        #     elif isinstance(module, (DynamicGDN, )):
        #         module.register_forward_hook(count_gdn)

        self.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)

    def _set_slimmable_level(self, level):
        for layer in self.pgm_model.modules():
            if isinstance(layer, BaseSlimmableLayer):
                layer.set_dynamic_parameter_value("active_channels_idx", level)

    def get_current_flops(self, input=None):
        # total ops calculated during forward
        return self.total_ops.squeeze().clone()

    def get_nn_complexity(self, input=None, metric=None):
        # TODO: implement other metrics. By default we return total_ops regardless of metrics
        return self.total_ops.squeeze().clone()
    
    def _self_distillation_loss(self, output, output_target):
        batch_size = output.shape[0]
        if self.training_self_distillation_loss_type == "MSE":
            self_distillation_loss = F.mse_loss(output, output_target.detach(), reduction='sum') / batch_size
        elif self.training_self_distillation_loss_type == "L1":
            self_distillation_loss = F.l1_loss(output, output_target.detach(), reduction='sum') / batch_size
        elif self.training_self_distillation_loss_type == "BCE":
            self_distillation_loss = F.binary_cross_entropy_with_logits(output, output_target.detach(), reduction='sum') / batch_size
        else:
            raise NotImplementedError(f"Unknown training_self_distillation_loss_type {self.training_self_distillation_loss_type}")
        return self_distillation_loss

    def _forward_slimmable(self, input, *args, pgm_weights=None, **kwargs):
        if pgm_weights is not None:
            level = pgm_weights.squeeze(0).squeeze(0).argmax().item()
        else:
            # TODO: use max level as default?
            level = len(self.mid_channels_list) - 1

        if self.training and self.training_self_distillation and level != len(self.mid_channels_list) - 1:
            # use max level
            self._set_slimmable_level(len(self.mid_channels_list) - 1)
            output_target = self.pgm_model(input)
            # NOTE: do not backprop input for non-maxlevel inputs 
            input = input.detach()

        # use a one hot threshold to accelerate training?
        if self.training and pgm_weights.max() < self.pgm_weight_one_hot_threshold:
            output_all_level = []
            total_ops_all_level = []
            for level in range(len(self.mid_channels_list)):
                self._set_slimmable_level(level)
                output_current = self.pgm_model(input)
                output_all_level.append(output_current)
                # cache flops
                total_ops = 0
                for module in self.pgm_model.modules():
                    if isinstance(module, DynamicNNTrainableModule):
                        total_ops += module.get_current_flops()
                        # TODO: find a better way to clear total_ops
                        if hasattr(module, "total_ops"):
                            module.total_ops.fill_(0)
                total_ops_all_level.append(total_ops)

            output = (pgm_weights.squeeze(0).squeeze(0) * torch.stack(output_all_level, dim=-1)).sum(-1)
            total_ops = (pgm_weights.squeeze(0).squeeze(0) * torch.stack(total_ops_all_level)).sum()
        else:
            self._set_slimmable_level(level)
            output = self.pgm_model(input)

            # cache flops
            total_ops = 0
            for module in self.pgm_model.modules():
                if isinstance(module, DynamicNNTrainableModule):
                    total_ops += module.get_current_flops()
                    # TODO: find a better way to clear total_ops
                    if hasattr(module, "total_ops"):
                        module.total_ops.fill_(0)
            total_ops = total_ops.squeeze() * pgm_weights.max()

        if self.training and self.training_self_distillation and level != len(self.mid_channels_list) - 1:
            self_distillation_loss = self._self_distillation_loss(output, output_target)
            if self.use_sandwich_rule and level != 0:
                self._set_slimmable_level(0)
                output_min = self.pgm_model(input)
                # self_distillation_loss = (self_distillation_loss + self._self_distillation_loss(output_min, output_target)) / 2
                self_distillation_sandwich_loss = self._self_distillation_loss(output_min, output_target)
                self.update_cache("loss_dict", self_distillation_sandwich_loss=self_distillation_sandwich_loss)
            self.update_cache("loss_dict", self_distillation_loss=self_distillation_loss)
            output = output_target
        # NOTE: we have moved all submodules flops to this module,
        # so clear all ops counter in submodules to avoid multiple op counting (this is slow!!)
        # for name, param in self.named_buffers():
        #     if param != self.total_ops and "total_ops" in name:
        #         param.data.fill_(0)

        return output, total_ops

    def forward_bottleneck_in(self, input, *args, pgm=None, input_mask=None, **kwargs):
        # clear all ops counter
        # for name, param in self.named_buffers():
        #     if "total_ops" in name:
        #         param.data.fill_(0)

        # if pgm is None:
        #     pgm = self.default_pgm
        pgm_weights = self.get_pgm_weights(self.default_pgm) if pgm is None else pgm
        # normalize
        # if (pgm_weights.sum(-1) != 1.0).any():
        #     pgm_weights = torch.softmax(pgm_weights, dim=-1)

        # TODO: do we need input_mask here?
        if input_mask is None and self.enable_default_input_mask:
            input_mask = torch.sigmoid(self.default_input_mask, dim=-1)
        if input_mask is not None:
            assert input_mask.numel() == self.in_groups
            batch_size, channels, height, width = input.shape
            input_group = input.reshape(batch_size, self.in_groups, channels // self.in_groups, height, width)
            if self.training:
                input_group = input_group * input_mask.squeeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                # TODO: slim input during testing
                input_group = input_group[:, input_mask > 0.5]
            input = input_group.reshape(batch_size, -1, height, width)

        # per batch pgm
        if pgm_weights.ndim == 4:
            if pgm_weights.shape[0] == input.shape[0]:
                self.total_ops.fill_(0)
                # TODO: can we apply models with different slimmable levels in batch?
                output = []
                for input_single, pgm_weights_single in zip(input.split(1, dim=0), pgm_weights.split(1, dim=0)):
                    output_single, total_ops_single = self._forward_slimmable(input_single, *args, pgm_weights=pgm_weights_single, **kwargs)
                    self.total_ops += total_ops_single
                    output.append(output_single)
                output = torch.cat(output, dim=0)
            elif pgm_weights.shape[0] == 1:
                output, total_ops = self._forward_slimmable(input, *args, pgm_weights=pgm_weights.squeeze(0), **kwargs)
                self.total_ops = total_ops
            else:
                raise ValueError("")
        else:
            output, total_ops = self._forward_slimmable(input, *args, pgm_weights=pgm_weights, **kwargs)
            self.total_ops = total_ops
        self.update_cache("moniter_dict", pgm_model_ops=self.total_ops.clone())

        return output


class HyperpriorSlimmableConv2dPGMModel(SlimmableConv2dPGMModel):
    def __init__(self, *args, conv_compability=False, **kwargs):
        self.conv_compability = conv_compability
        super().__init__(*args, **kwargs)


class HyperpriorAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicGDN(self.mid_channels_list),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicGDN(self.mid_channels_list),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicGDN(self.mid_channels_list),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
        )


class HyperpriorSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicGDN(self.mid_channels_list, inverse=True),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicGDN(self.mid_channels_list, inverse=True),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicGDN(self.mid_channels_list, inverse=True),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, transposed=True, conv_compability=self.conv_compability),
        )


class HyperpriorHyperAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, stride=1, kernel_size=3, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
        )


class HyperpriorHyperSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, stride=1, kernel_size=3, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
        )


class SFMAHyperpriorSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def __init__(self, *args, sfma_mid_channels_list=[16, 32, 64], sfma_num_modulators=1, dynamic_freeze_backbone=True, freeze_backbone=False, hyperprior_model_compability=False, **kwargs):
        self.sfma_mid_channels_list = sfma_mid_channels_list
        self.sfma_num_modulators = sfma_num_modulators
        self.dynamic_freeze_backbone = dynamic_freeze_backbone
        self.freeze_backbone = freeze_backbone
        self.hyperprior_model_compability = hyperprior_model_compability
        if hyperprior_model_compability:
            kwargs["conv_compability"] = True

        super().__init__(*args, **kwargs)

        if self.freeze_backbone:
            for module in self.pgm_model.children():
                if not isinstance(module, GroupedDynamicSpatialFrequencyModulationAdaptor):
                    module.requires_grad_(False)
                    # for p in module.parameters(): p.lr_modifier = 0.0
    
    def forward(self, input, *args, pgm=None, input_mask=None, idx=0, **kwargs):
        sfma_model = self.sfma_model if self.hyperprior_model_compability else self.pgm_model
        for module in sfma_model.children():
            if isinstance(module, GroupedDynamicSpatialFrequencyModulationAdaptor):
                module.set_dynamic_parameter_value("sfma_idx", idx)
            else:
                if self.dynamic_freeze_backbone and not self.freeze_backbone:
                    module.requires_grad_(idx==0)
                    # for param in module.parameters():
                    #     param.requires_grad = (idx==0)
        return super().forward(input, *args, pgm=pgm, input_mask=input_mask, **kwargs)


class SFMAHyperpriorAnalysisSlimmableConv2dPGMModel(SFMAHyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        if self.hyperprior_model_compability:
            sfma_model = []
            pgm_model = nn.Sequential(
                DynamicConv2d(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list),
                DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
            )
            for module in pgm_model:
                if isinstance(module, DynamicGDN):
                    sfma_module = GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators)
                    sfma_model.append(sfma_module)
                    module.register_forward_hook(lambda module, input, output: sfma_module(output))
            self.sfma_model = nn.ModuleList(sfma_model)
        else:
            pgm_model = nn.Sequential(
                DynamicConv2d(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list),
                GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list),
                GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list),
                GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators),
                DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
            )
        return pgm_model


class SFMAHyperpriorSynthesisSlimmableConv2dPGMModel(SFMAHyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        if self.hyperprior_model_compability:
            sfma_model = []
            pgm_model = nn.Sequential(
                DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list, inverse=True),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list, inverse=True),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
                DynamicGDN(self.mid_channels_list, inverse=True),
                DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, transposed=True, conv_compability=self.conv_compability),
            )
            for module in pgm_model:
                if isinstance(module, DynamicGDN):
                    sfma_module = GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators)
                    sfma_model.append(sfma_module)
                    module.register_forward_pre_hook(lambda module, input: sfma_module(input[0]))
            self.sfma_model = nn.ModuleList(sfma_model)
        else:
            pgm_model = nn.Sequential(
                DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
                GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators),
                DynamicGDN(self.mid_channels_list, inverse=True),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
                GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators),
                DynamicGDN(self.mid_channels_list, inverse=True),
                DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
                GroupedDynamicSpatialFrequencyModulationAdaptor(self.mid_channels_list, self.sfma_mid_channels_list, num_modulators=self.sfma_num_modulators),
                DynamicGDN(self.mid_channels_list, inverse=True),
                DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, transposed=True, conv_compability=self.conv_compability),
            )
        return pgm_model


class MeanScaleHyperpriorHyperAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, stride=1, kernel_size=3, conv_compability=self.conv_compability),
            nn.LeakyReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            nn.LeakyReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
        )


class MeanScaleHyperpriorHyperSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        assert(max(self.out_channels_list) == max(self.mid_channels_list) * 2)
        l2_mid_channels_list = [c * 3 // 2 for c in self.mid_channels_list]
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            nn.LeakyReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), l2_mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            nn.LeakyReLU(inplace=True),
            DynamicConv2d(max(l2_mid_channels_list), self.out_channels_list, stride=1, kernel_size=3, conv_compability=self.conv_compability),
        )


class Cheng2020AnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicResidualBlockWithStride(self.in_channels, self.mid_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list),
            DynamicResidualBlockWithStride(max(self.mid_channels_list), self.mid_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicAttentionBlock(self.mid_channels_list),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list),
            DynamicResidualBlockWithStride(max(self.mid_channels_list), self.mid_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list),
            dynamic_conv3x3(max(self.mid_channels_list), self.out_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicAttentionBlock(self.out_channels_list),
        )


class Cheng2020SynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicAttentionBlock(self.in_channels),
            DynamicResidualBlock(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBlockUpsample(max(self.mid_channels_list), self.mid_channels_list, 2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBlockUpsample(max(self.mid_channels_list), self.mid_channels_list, 2, conv_compability=self.conv_compability),
            DynamicAttentionBlock(self.mid_channels_list),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBlockUpsample(max(self.mid_channels_list), self.mid_channels_list, 2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            dynamic_subpel_conv3x3(max(self.mid_channels_list), self.out_channels_list, 2),
        )


class Cheng2020NoAttnAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicResidualBlockWithStride(self.in_channels, self.mid_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list),
            DynamicResidualBlockWithStride(max(self.mid_channels_list), self.mid_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list),
            DynamicResidualBlockWithStride(max(self.mid_channels_list), self.mid_channels_list, stride=2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list),
            dynamic_conv3x3(max(self.mid_channels_list), self.out_channels_list, stride=2, conv_compability=self.conv_compability),
        )


class Cheng2020NoAttnSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicResidualBlock(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBlockUpsample(max(self.mid_channels_list), self.mid_channels_list, 2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBlockUpsample(max(self.mid_channels_list), self.mid_channels_list, 2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBlockUpsample(max(self.mid_channels_list), self.mid_channels_list, 2, conv_compability=self.conv_compability),
            DynamicResidualBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            dynamic_subpel_conv3x3(max(self.mid_channels_list), self.out_channels_list, 2),
        )


class Cheng2020HyperAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            dynamic_conv3x3(self.in_channels, self.mid_channels_list),
            nn.LeakyReLU(inplace=True),
            dynamic_conv3x3(max(self.mid_channels_list), self.mid_channels_list),
            nn.LeakyReLU(inplace=True),
            dynamic_conv3x3(max(self.mid_channels_list), self.mid_channels_list, 2),
            nn.LeakyReLU(inplace=True),
            dynamic_conv3x3(max(self.mid_channels_list), self.mid_channels_list),
            nn.LeakyReLU(inplace=True),
            dynamic_conv3x3(max(self.mid_channels_list), self.out_channels_list, 2),
        )


class Cheng2020HyperSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        assert(max(self.out_channels_list) == max(self.mid_channels_list) * 2)
        l2_mid_channels_list = [c * 3 // 2 for c in self.mid_channels_list]
        return nn.Sequential(
            dynamic_conv3x3(self.in_channels, self.mid_channels_list),
            nn.LeakyReLU(inplace=True),
            dynamic_subpel_conv3x3(max(self.mid_channels_list), self.mid_channels_list, 2),
            nn.LeakyReLU(inplace=True),
            dynamic_conv3x3(max(self.mid_channels_list), l2_mid_channels_list),
            nn.LeakyReLU(inplace=True),
            dynamic_subpel_conv3x3(max(l2_mid_channels_list), l2_mid_channels_list, 2),
            nn.LeakyReLU(inplace=True),
            dynamic_conv3x3(max(l2_mid_channels_list), self.out_channels_list),
        )


class ELICNoAttnAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
        )


class ELICNoAttnSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, transposed=True, conv_compability=self.conv_compability),
        )


class ELICAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicAttentionBlock(self.mid_channels_list),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
            DynamicAttentionBlock(self.out_channels_list),
        )


class ELICSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicAttentionBlock(self.in_channels),
            DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicAttentionBlock(self.mid_channels_list),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicResidualBottleneckBlock(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, transposed=True, conv_compability=self.conv_compability),
        )


class ELICHyperAnalysisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, stride=1, kernel_size=3, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.mid_channels_list, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), self.out_channels_list, conv_compability=self.conv_compability),
        )


class ELICHyperSynthesisSlimmableConv2dPGMModel(HyperpriorSlimmableConv2dPGMModel):
    def build_pgm_model(self):
        assert(max(self.out_channels_list) == max(self.mid_channels_list) * 2)
        l2_mid_channels_list = [c * 3 // 2 for c in self.mid_channels_list]
        return nn.Sequential(
            DynamicConv2d(self.in_channels, self.mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(self.mid_channels_list), l2_mid_channels_list, transposed=True, conv_compability=self.conv_compability),
            nn.ReLU(inplace=True),
            DynamicConv2d(max(l2_mid_channels_list), self.out_channels_list, stride=1, kernel_size=3, conv_compability=self.conv_compability),
        )

