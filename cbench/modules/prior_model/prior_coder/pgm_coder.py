import math
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import copy
import struct
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision.utils import make_grid

try:
    from survae.distributions import Distribution as SurDistribution
except:
    class SurDistribution:
        pass

# from .base import PriorCoder
from . import NNPriorCoder
from .torch_ans import ContinuousDistributionANSPriorCoder
from .compressai_coder import get_scale_table
from .base import PriorCoder

from cbench.nn.base import NNTrainableModule
from cbench.nn.layers.entroformer_layers import TransDecoder, TransDecoderCheckerboard
from cbench.nn.layers.param_generator import NNParameterGenerator
from cbench.nn.layers.masked_conv import TopoGroupDynamicMaskConv2d, TopoGroupDynamicMaskConv2dContextModel
from cbench.nn.models.unet import GeneratorUNet
from cbench.nn.distributions.mixture import ReparametrizedMixtureSameFamily, StableNormal

from compressai.layers import GDN, MaskedConv2d
from compressai.ops.bound_ops import LowerBound, LowerBoundFunction

class LowerBoundNoBuf(nn.Module):

    bound: torch.Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]), persistent=False)

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)

# class TrainablePGMPrior(NNTrainableModule):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
#         return self.inference(input, prior=prior, **kwargs)

#     def inference(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
#         raise NotImplementedError()

#     def generate(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs):
#         raise NotImplementedError()
    
#     def update_state(self, *args, **kwargs) -> None:
#         return super().update_state(*args, **kwargs)


def get_reinforce_loss(log_weight, log_q,
                       num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)

    # this is term 1 in equation (2) of https://arxiv.org/pdf/1805.10469.pdf
    reinforce_correction = log_evidence.detach() * torch.sum(log_q, dim=1)

    loss = - torch.mean(reinforce_correction)
    return loss


def get_vimco_loss(log_weight, log_q, num_particles=1):
    """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    # shape [batch_size, num_particles]
    # log_weight_[b, k] = 1 / (K - 1) \sum_{\ell \neq k} \log w_{b, \ell}
    log_weight_ = (torch.sum(log_weight, dim=1, keepdim=True) - log_weight) \
        / (num_particles - 1)

    # shape [batch_size, num_particles, num_particles]
    # temp[b, k, k_] =
    #     log_weight_[b, k]     if k == k_
    #     log_weight[b, k]      otherwise
    temp = log_weight.unsqueeze(-1) + torch.diag_embed(
        log_weight_ - log_weight)

    # this is the \Upsilon_{-k} term below equation 3
    # shape [batch_size, num_particles]
    control_variate = torch.logsumexp(temp, dim=1) - np.log(num_particles)

    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    loss = - torch.mean(torch.sum(
        (log_evidence.unsqueeze(-1) - control_variate).detach() * log_q, dim=1
    ))

    return loss

def get_partition(num_partitions, partition_type, log_beta_min=-10,
                  device=None):
    """Create a non-decreasing sequence of values between zero and one.
    See https://en.wikipedia.org/wiki/Partition_of_an_interval.

    Args:
        num_partitions: length of sequence minus one
        partition_type: \'linear\' or \'log\'
        log_beta_min: log (base ten) of beta_min. only used if partition_type
            is log. default -10 (i.e. beta_min = 1e-10).
        device: torch.device object (cpu by default)

    Returns: tensor of shape [num_partitions + 1]
    """
    if device is None:
        device = torch.device('cpu')
    if num_partitions == 1:
        partition = torch.tensor([0, 1], dtype=torch.float, device=device)
    else:
        if partition_type == 'linear':
            partition = torch.linspace(0, 1, steps=num_partitions + 1,
                                       device=device)
        elif partition_type == 'log':
            partition = torch.zeros(num_partitions + 1, device=device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(
                log_beta_min, 0, steps=num_partitions, device=device,
                dtype=torch.float)
    return partition

def get_thermo_loss(log_weight, log_q,
                    partition=None, num_particles=1, integration='left'):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    heated_log_weight = log_weight.unsqueeze(-1) * partition
    log_denominator = torch.logsumexp(heated_log_weight, dim=1, keepdim=True)
    heated_normalized_weight = torch.exp(heated_log_weight - log_denominator)
    log_p = log_weight + log_q
    thermo_logp = partition * log_p.unsqueeze(-1) + \
        (1 - partition) * log_q.unsqueeze(-1)

    wf = heated_normalized_weight * log_weight.unsqueeze(-1)
    w_detached = heated_normalized_weight.detach()
    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    thing_to_add = correction * torch.sum(
        w_detached *
        (log_weight.unsqueeze(-1) -
            torch.sum(wf, dim=1, keepdim=True)).detach() *
        (thermo_logp -
            torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)),
        dim=1)

    multiplier = torch.zeros_like(partition)
    if integration == 'trapz':
        multiplier[0] = 0.5 * (partition[1] - partition[0])
        multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
        multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
    elif integration == 'left':
        multiplier[:-1] = partition[1:] - partition[:-1]
    elif integration == 'right':
        multiplier[1:] = partition[1:] - partition[:-1]

    loss = -torch.mean(torch.sum(
        multiplier * (thing_to_add + torch.sum(
            w_detached * log_weight.unsqueeze(-1), dim=1)),
        dim=1))

    return loss

class NNTrainablePGMPriorCoder(ContinuousDistributionANSPriorCoder):
    def __init__(self, *args, 
                 use_bypass_coding=True, # force bypass coding for stablility
                 fixed_input_shape : Optional[Tuple[int]] = None,
                 force_input_prior_shape_aligned=True,
                 eps=1e-7, 
                 pgm_input_dequantized=False,
                 training_no_quantize_for_likelihood=False,
                 training_output_straight_through=False,
                 training_mc_sampling=False, 
                 training_mc_num_samples=1,
                 training_mc_loss_type="reinforce",
                 mc_loss_weight=1.0,
                 mc_loss_weight_anneal=False,
                 training_mc_for_ga=False,
                 training_mc_for_ga_num_population=5,
                 training_mc_for_ga_fitness_window_size=100,
                 training_mc_for_ga_mutate_entropy_thres=0.001,
                 training_mc_for_ga_force_update_num_steps=-1,
                 training_mc_for_ga_keep_parents=1,
                 training_mc_for_ga_num_new_children=-1,
                 **kwargs):
        super().__init__(*args, use_bypass_coding=use_bypass_coding, **kwargs)
        # self.pgm_prior = pgm_prior
        self.fixed_input_shape = fixed_input_shape
        self.force_input_prior_shape_aligned = force_input_prior_shape_aligned
        self.eps = eps
        self.lower_bound_eps = LowerBoundNoBuf(eps)

        self.pgm_input_dequantized = pgm_input_dequantized
        self.training_no_quantize_for_likelihood = training_no_quantize_for_likelihood
        self.training_output_straight_through = training_output_straight_through
        self.training_mc_sampling = training_mc_sampling
        self.training_mc_num_samples = training_mc_num_samples
        self.training_mc_loss_type = training_mc_loss_type
        if training_mc_loss_type == "thermo":
            self.register_buffer("thermo_partition", get_partition(10, "log", device=self.device))

        self.mc_loss_weight = mc_loss_weight
        self.mc_loss_weight_anneal = mc_loss_weight_anneal
        if mc_loss_weight_anneal:
            self.mc_loss_weight = nn.Parameter(torch.tensor(mc_loss_weight), requires_grad=False)

        self.training_mc_for_ga = training_mc_for_ga
        self.training_mc_for_ga_num_population = training_mc_for_ga_num_population
        self.training_mc_for_ga_fitness_window_size = training_mc_for_ga_fitness_window_size
        self.training_mc_for_ga_mutate_entropy_thres = training_mc_for_ga_mutate_entropy_thres
        self.training_mc_for_ga_force_update_num_steps = training_mc_for_ga_force_update_num_steps
        self.training_mc_for_ga_keep_parents = training_mc_for_ga_keep_parents
        self.training_mc_for_ga_num_new_children = training_mc_for_ga_num_new_children
        if training_mc_for_ga:
            self.register_buffer("ga_fitness_window", torch.zeros(self.training_mc_for_ga_fitness_window_size, self.training_mc_for_ga_num_population))
            self.register_buffer("ga_fitness_idx", torch.zeros(1, dtype=torch.long))
            self.register_buffer("ga_fitness", torch.zeros(self.training_mc_for_ga_num_population))
            self.register_buffer("ga_update_steps", torch.zeros(1, dtype=torch.long))
            self.training_mc_num_samples *= self.training_mc_for_ga_num_population
            # self._ga_update_population()

        # self.lower_bound_scale = LowerBound(0.11)

    # def _to_ans_params(self, params) -> Dict[str, np.ndarray]:
    #     raise NotImplementedError()

    # def _to_dist(self, params) -> D.Distribution:
    #     raise NotImplementedError()

    # def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None) -> torch.Tensor:
    #     raise NotImplementedError()

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, fast_mode : bool = False, **kwargs) -> Any:
        raise NotImplementedError()
    
    def _pgm_sort_topo(self, input : torch.Tensor, pgm : Any, **kwargs) -> Optional[torch.LongTensor]:
        raise NotImplementedError()
    
    def _pgm_inference(self, input : torch.Tensor, pgm : Any, prior : Optional[torch.Tensor] = None, **kwargs):
        raise NotImplementedError()

    def _pgm_generate(self, byte_string : bytes, pgm : Any, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, **kwargs):
        raise NotImplementedError()

    def _dist_param_quant_reparam(self, params : torch.Tensor, quantizer_params: Optional[torch.Tensor] = None, **kwargs):
        raise NotImplementedError()

    def _get_params_with_pgm(self, input : torch.Tensor, prior : Optional[torch.Tensor] = None, pgm : Optional[Any] = None, **kwargs) -> torch.Tensor:
        pgm = self._get_pgm(input, prior=prior, input_shape=input.shape, pgm=pgm)
        return self._pgm_inference(input, pgm, prior=prior, **kwargs)

    def _decode_with_pgm(self, byte_string : bytes, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Optional[Any] = None, **kwargs) -> torch.Tensor:
        pgm = self._get_pgm(byte_string, prior=prior, input_shape=input_shape, pgm=pgm, fast_mode=True)
        return self._pgm_generate(byte_string, pgm, prior=prior, input_shape=input_shape, **kwargs)

    def _encode_with_pgm(self, input : torch.Tensor, *args, prior : torch.Tensor = None, pgm : Optional[Any] = None, quantizer_params: Optional[Any] = None, **kwargs) -> bytes:
        input_quant = self._data_preprocess(input, quantizer_params=quantizer_params)
        if self.pgm_input_dequantized:
            input_dequant = self._data_postprocess(input_quant, quantizer_params=quantizer_params)
            pgm_input = input_dequant.type_as(input)
        else:
            pgm_input = input_quant.type_as(input)
        pgm = self._get_pgm(pgm_input, prior=prior, input_shape=input.shape, pgm=pgm, fast_mode=True)
        params = self._pgm_inference(pgm_input, pgm, prior=prior, quantizer_params=quantizer_params)
        if self.pgm_input_dequantized:
            params = self._dist_param_quant_reparam(params, quantizer_params=quantizer_params)
        indexes, data_offsets = self._build_coding_params(params)
        # NOTE: quantizing data_offsets may be inaccurate for prior!
        # Using self._data_preprocess(input - data_offsets) could be more accurate, but proper decode might be tricky. (See CompressAI JointAR impl)
        data = input_quant - data_offsets #.detach().cpu().numpy().astype(np.int32)
        # data = self._data_preprocess(input - data_offsets)
        # pgm_topo_order = self._pgm_sort_topo(input_quant, pgm)
        # if pgm_topo_order is not None:
        #     pgm_topo_order = pgm_topo_order.detach().cpu().numpy().astype(np.int32)
        #     data = data.reshape(-1)[pgm_topo_order.reshape(-1)]
        #     indexes = indexes.reshape(-1)[pgm_topo_order.reshape(-1)]
        # topological sorting
        data, indexes = self._encode_topo_sort(data, indexes, pgm)

        data = data.contiguous().detach().cpu().numpy().astype(np.int32)
        indexes = indexes.contiguous().detach().cpu().numpy().astype(np.int32)
        byte_string = self.ans_encoder.encode_with_indexes(data, indexes)
        # self.ans_encoder.encode_with_indexes(data, indexes, cache=True)
        # byte_string = self.ans_encoder.flush()

        return byte_string

    # from CompressAI
    # def quantize(
    #     self, inputs: torch.Tensor, mode: str, means: Optional[torch.Tensor] = None
    # ) -> torch.Tensor:
    #     if mode not in ("noise", "dequantize", "symbols"):
    #         raise ValueError(f'Invalid quantization mode: "{mode}"')

    #     if mode == "noise":
    #         half = float(0.5)
    #         noise = torch.empty_like(inputs).uniform_(-half, half)
    #         inputs = inputs + noise
    #         return inputs

    #     outputs = inputs.clone()
    #     if means is not None:
    #         outputs -= means

    #     outputs = torch.round(outputs)

    #     if mode == "dequantize":
    #         if means is not None:
    #             outputs += means
    #         return outputs

    #     assert mode == "symbols", mode
    #     outputs = outputs.int()
    #     return outputs

    def _ga_update_population(self, parents=None):
        # raise NotImplementedError()
        pass

    def _get_likelihood_and_entropy(self, data, params, pgm=None, quantizer_params=None, **kwargs):
        # NOTE: more accurate likelihood estimation (from CompressAI implementation)
        if self.training_no_quantize_for_likelihood:
            dist, data_offset = self._params_to_dist_and_offset(params)
            if self.training and self.training_mc_sampling:
                data = self._data_preprocess(data, differentiable=self.training, quantizer_params=quantizer_params)
                data = data.unsqueeze(1).repeat(1, self.training_mc_num_samples, *([1]*(data.ndim-1)))\
                    .reshape(*data_offset.shape) - data_offset
            else:
                data = self._data_preprocess(data - data_offset, differentiable=self.training)
        else:
            dist = self._params_to_dist(params)
        # TODO: use data step?
        likelihood = dist.cdf(data + 0.5) - dist.cdf(data - 0.5)
        entropy = -torch.log(self.lower_bound_eps(likelihood))
        return likelihood, entropy

    def forward(self, input : torch.Tensor, prior : Optional[torch.Tensor] = None, pgm : Optional[Any] = None, quantizer_params: Optional[Any] = None, **kwargs):
        # input_quant = self.quantize(
        #     input, "noise" if self.training else "dequantize"
        # )
        # input = input[..., :4, :4]
        batch_size = input.shape[0]
        input_quant = self._data_preprocess(input, differentiable=self.training, quantizer_params=quantizer_params)
        input_dequant = self._data_postprocess(input_quant, quantizer_params=quantizer_params)
        if self.pgm_input_dequantized:
            pgm_input = input_dequant.type_as(input)
        else:
            pgm_input = input_quant.type_as(input)

        # make prior the same spatial size as input
        if prior is not None:
            if self.force_input_prior_shape_aligned:
                assert input.shape[2:] == prior.shape[2:], \
                    "Input and prior shape not aligned! Consider setting force_input_prior_shape_aligned = False, which may add a little overhead to the bitstream to save the input shape!"
            else:
                for dim, size in enumerate(input.shape[2:], 2):
                    prior = prior.narrow(dim, 0, size)

        if self.training_no_quantize_for_likelihood:
            input_no_quant = self._data_preprocess(input, quantize=False, quantizer_params=quantizer_params)
        if self.training and self.training_mc_sampling:
            pgm_inference_input = pgm_input.unsqueeze(1).repeat(1, self.training_mc_num_samples, *([1]*(input.ndim-1)))\
                .reshape(batch_size*self.training_mc_num_samples, *input.shape[1:])
            if prior is not None:
                prior = prior.unsqueeze(1).repeat(1, self.training_mc_num_samples, *([1]*(prior.ndim-1)))\
                    .reshape(batch_size*self.training_mc_num_samples, *prior.shape[1:])
            # if self.training_no_quantize_for_likelihood:
            #     input_no_quant = input_no_quant.unsqueeze(1).repeat(1, self.training_mc_num_samples, *([1]*(input.ndim-1)))\
            #         .reshape(batch_size*self.training_mc_num_samples, *input.shape[1:])
        else:
            pgm_inference_input = pgm_input
        pgm = self._get_pgm(pgm_inference_input, prior=prior, input_shape=pgm_inference_input.shape, pgm=pgm)
        params = self._pgm_inference(pgm_inference_input, pgm, prior=prior, quantizer_params=quantizer_params)
        if self.pgm_input_dequantized:
            params = self._dist_param_quant_reparam(params, quantizer_params=quantizer_params)
        if self.training_no_quantize_for_likelihood:
            likelihood, entropy = self._get_likelihood_and_entropy(input_no_quant, params, pgm=pgm, quantizer_params=quantizer_params)
        else:
            likelihood, entropy = self._get_likelihood_and_entropy(input_quant, params, pgm=pgm, quantizer_params=quantizer_params)
        # means, logvars = params.chunk(2, dim=1)
        # scales = torch.exp(logvars)
        # scales = scales.clamp_min(0.11) # self.lower_bound_scale(scales)
        # def _likelihood(inputs, means, scales):
        #     def _standardized_cumulative(inputs):
        #         half = float(0.5)
        #         const = float(-(2**-0.5))
        #         # Using the complementary error function maximizes numerical precision.
        #         return half * torch.erfc(const * inputs)

        #     half = float(0.5)

        #     if means is not None:
        #         values = inputs - means
        #     else:
        #         values = inputs

        #     values = torch.abs(values)
        #     upper = _standardized_cumulative((half - values) / scales)
        #     lower = _standardized_cumulative((-half - values) / scales)
        #     likelihood = upper - lower

        #     return likelihood
        # likelihood = _likelihood(input_quant, means, scales)

        if self.training:
            if self.training_mc_sampling:
                if self.training_mc_for_ga:
                    log_weight = -entropy.reshape(batch_size, self.training_mc_num_samples // self.training_mc_for_ga_num_population, self.training_mc_for_ga_num_population, -1)\
                        .movedim(1,2).sum(-1).reshape(batch_size * self.training_mc_for_ga_num_population, self.training_mc_num_samples // self.training_mc_for_ga_num_population)
                    entropy = -torch.log(self.lower_bound_eps(likelihood.reshape(batch_size, self.training_mc_num_samples // self.training_mc_for_ga_num_population, self.training_mc_for_ga_num_population, -1).mean(1))).sum(-1)
                    num_particles = self.training_mc_num_samples // self.training_mc_for_ga_num_population
                else:
                    log_weight = -entropy.reshape(batch_size, self.training_mc_num_samples, -1).sum(-1)
                    entropy = -torch.log(self.lower_bound_eps(likelihood.reshape(batch_size, self.training_mc_num_samples, -1).mean(1))).sum(-1)
                    num_particles = self.training_mc_num_samples
                    # log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(self.training_mc_num_samples)
                    # entropy = -log_evidence
                pgm_log_prob = self.get_raw_cache().pop("pgm_log_prob", None)
                if pgm_log_prob is not None:
                    pgm_entropy = -pgm_log_prob * pgm_log_prob.exp()
                    self.update_cache("moniter_dict", 
                        pgm_entropy = pgm_entropy.mean(),
                    )
                    if self.training_mc_for_ga:
                        with torch.no_grad():
                            ga_fitness = -entropy.mean(0)
                            # cache fitness (TODO: use momentum on fitness)
                            # self.ga_fitness += ga_fitness.detach()
                            self.ga_fitness_window[self.ga_fitness_idx] = ga_fitness.data
                            self.ga_fitness = self.ga_fitness_window.mean(0).data
                            self.ga_fitness_idx += 1
                            self.ga_update_steps += 1
                            if self.ga_fitness_idx >= self.training_mc_for_ga_fitness_window_size:
                                self.ga_fitness_idx.data.fill_(0)
                            # see if we need mutation
                            pgm_entropy_per_population = pgm_entropy.reshape(batch_size, self.training_mc_num_samples // self.training_mc_for_ga_num_population, self.training_mc_for_ga_num_population, -1).mean((0,1,3))
                            if pgm_entropy_per_population.max() < self.training_mc_for_ga_mutate_entropy_thres or \
                                (self.training_mc_for_ga_force_update_num_steps > 0 and self.ga_update_steps > self.training_mc_for_ga_force_update_num_steps):
                                self._ga_update_population()
                                # clear all fitness data
                                self.ga_fitness_window.data.fill_(0)
                                self.ga_fitness_idx.data.fill_(0)
                                self.ga_update_steps.data.fill_(0)
                        entropy, _ = entropy.min(1) # remove ga population dim, using the best result
                        log_q = pgm_log_prob.reshape(batch_size, self.training_mc_num_samples // self.training_mc_for_ga_num_population, self.training_mc_for_ga_num_population, -1)\
                            .movedim(1,2).sum(-1).reshape(batch_size * self.training_mc_for_ga_num_population, self.training_mc_num_samples // self.training_mc_for_ga_num_population)
                    else:
                        log_q = pgm_log_prob.reshape(batch_size, self.training_mc_num_samples, -1).sum(-1)
                    if self.training_mc_loss_type == "reinforce":
                        # loss_mc = pgm_log_prob.reshape(batch_size, -1).sum(-1) * entropy.detach().reshape(batch_size, -1).sum(-1)
                        loss_mc = get_reinforce_loss(log_weight, log_q, num_particles=num_particles)
                    elif self.training_mc_loss_type == "vimco":
                        loss_mc = get_vimco_loss(log_weight, log_q, num_particles=num_particles)
                    elif self.training_mc_loss_type == "thermo":
                        loss_mc = get_thermo_loss(log_weight, log_q, num_particles=num_particles, partition=self.thermo_partition)
                    else:
                        raise NotImplementedError(f"Unknown training_mc_loss_type {self.training_mc_loss_type}")
                    self.update_cache("loss_dict", 
                        loss_mc = loss_mc * self.mc_loss_weight,
                    )
            # NOTE: we follow most works using bits as rate loss
            loss_rate = entropy.sum() / math.log(2) / batch_size # normalize by batch size
            self.update_cache("loss_dict", 
                loss_rate = loss_rate,
            )
        self.update_cache("metric_dict",
            prior_entropy = entropy.sum() / batch_size, # normalize by batch size
        )

        # moniter ga_fitness
        if self.training_mc_for_ga:
            self.update_cache("hist_dict", ga_fitness=self.ga_fitness)

        # anneal
        if self.mc_loss_weight_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    mc_loss_weight=self.mc_loss_weight
                )

        if self.training_output_straight_through:
            input_dequant_round = self._data_preprocess(input, differentiable=False, dequantize=True, quantizer_params=quantizer_params)
            input_dequant = input_dequant_round + input - input.detach()

        return input_dequant

    def _encode_topo_sort(self, data : torch.IntTensor, indexes : torch.IntTensor, pgm : Any) -> Tuple[torch.IntTensor, torch.IntTensor]:
        return data, indexes
    
    def encode(self, input : torch.Tensor, *args, prior : torch.Tensor = None, pgm : Optional[Any] = None, quantizer_params: Optional[Any] = None, **kwargs) -> bytes:
        # input = input[..., :4, :4]
        # with self.profiler.start_time_profile("time_data_preprocess_encode"):
        #     # input_quant = self.quantize(input, "symbols")
        #     input_quant = self._data_preprocess(input)
        #     # data = input_quant.detach().cpu().numpy().astype(np.int32)

        # with self.profiler.start_time_profile("time_prior_preprocess_encode"):
        #     pgm = self._get_pgm(input_quant.type_as(input), prior=prior, input_shape=input.shape, pgm=pgm, fast_mode=True)
        #     params = self._pgm_inference(input_quant.type_as(input), pgm, prior=prior)
        #     indexes, data_offsets = self._build_coding_params(params)
        #     # data = input_quant - data_offsets
        #     # NOTE: proper decode?
        #     data = self._data_preprocess(input - data_offsets)
        #     # pgm_topo_order = self._pgm_sort_topo(input_quant, pgm)
        #     # if pgm_topo_order is not None:
        #     #     pgm_topo_order = pgm_topo_order.detach().cpu().numpy().astype(np.int32)
        #     #     data = data.reshape(-1)[pgm_topo_order.reshape(-1)]
        #     #     indexes = indexes.reshape(-1)[pgm_topo_order.reshape(-1)]
        #     # topological sorting
        #     data, indexes = self._encode_topo_sort(data, indexes, pgm)

        # with self.profiler.start_time_profile("time_ans_encode"):
        #     data = data.contiguous().detach().cpu().numpy().astype(np.int32)
        #     indexes = indexes.contiguous().detach().cpu().numpy().astype(np.int32)
        #     byte_string = self.ans_encoder.encode_with_indexes(data, indexes)
        
        # make prior the same spatial size as input
        if prior is not None:
            if self.force_input_prior_shape_aligned:
                assert input.shape[2:] == prior.shape[2:], "Input and prior shape not aligned! Consider setting force_input_prior_shape_aligned = False, which may add a little overhead to the bitstream to save the input shape!"
            else:
                for dim, size in enumerate(input.shape[2:], 2):
                    prior = prior.narrow(dim, 0, size)

        byte_string = self._encode_with_pgm(input, prior=prior, pgm=pgm, quantizer_params=quantizer_params, **kwargs)

        # add shape into byte_string if no prior information is available
        byte_strings = []
        batch_size = input.shape[0]
        spatial_shape = input.shape[2:]
        if self.fixed_input_shape is not None:
            assert batch_size == self.fixed_input_shape[0]
            assert spatial_shape == self.fixed_input_shape[1:]
        elif self.force_input_prior_shape_aligned and prior is not None:
            pass
        else:
            byte_head = [struct.pack("B", len(spatial_shape)+1)]
            byte_head.append(struct.pack("<H", batch_size))
            for dim in spatial_shape:
                byte_head.append(struct.pack("<H", dim))
            byte_strings.extend(byte_head)
        byte_strings.append(byte_string)
        return b''.join(byte_strings)

    def decode(self, byte_string : bytes, *args, prior : torch.Tensor = None, pgm : Optional[Any] = None, quantizer_params: Optional[Any] = None, **kwargs) -> torch.Tensor:
        # decode shape from header
        if self.fixed_input_shape is not None:
            byte_ptr = 0
            batch_dim = self.fixed_input_shape[0]
            spatial_shape = self.fixed_input_shape[1:]
        elif self.force_input_prior_shape_aligned and prior is not None:
            byte_ptr = 0
            batch_dim = prior.shape[0]
            spatial_shape = prior.shape[2:]
        else:
            num_shape_dims = struct.unpack("B", byte_string[:1])[0]
            flat_shape = []
            byte_ptr = 1
            for _ in range(num_shape_dims):
                flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
                byte_ptr += 2
            batch_dim = flat_shape[0]
            spatial_shape = flat_shape[1:]
        input_shape = (batch_dim, -1, *spatial_shape)

        # make prior the same spatial size as input
        if prior is not None:
            if self.force_input_prior_shape_aligned:
                assert input_shape[2:] == prior.shape[2:], "Input and prior shape not aligned! Consider setting force_input_prior_shape_aligned = False, which may add a little overhead to the bitstream to save the input shape!"
            else:
                for dim, size in enumerate(input_shape[2:], 2):
                    prior = prior.narrow(dim, 0, size)

        decode_data = self._decode_with_pgm(byte_string[byte_ptr:], prior=prior, input_shape=input_shape, pgm=pgm, quantizer_params=quantizer_params)
        return decode_data # self._data_postprocess(decode_data, quantizer_params=quantizer_params)


class CombinedNNTrainablePGMPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self, coders, *args, 
                 blend_weight_one_hot_threshold=0.9, 
                 fix_weight=False, 
                 training_use_max_capacity=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

        self.coders = nn.ModuleList(coders)
        if fix_weight:
            self.register_buffer("default_blend_weight", torch.zeros(len(coders)), persistent=False)
        else:
            self.default_blend_weight = nn.Parameter(torch.zeros(len(coders)))
        self.blend_weight_one_hot_threshold = blend_weight_one_hot_threshold
        self.training_use_max_capacity = training_use_max_capacity

        # self.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64), persistent=False)

    def forward(self, input : torch.Tensor, prior : Optional[torch.Tensor] = None, blend_weight : Optional[Any] = None, **kwargs):
        if blend_weight is None:
            blend_weight = torch.softmax(self.default_blend_weight, dim=0)

        # TODO: blend likelihood
        if self.training and self.training_use_max_capacity:
            for coder_idx, coder in enumerate(self.coders):
                # NOTE: we suppose the first coder as max capacity!
                if coder_idx == 0:
                    ret = coder(input, prior=prior, **kwargs)
                    loss_rate = coder.get_raw_cache("loss_dict").pop("loss_rate", None)
                    if loss_rate is not None:
                        self.update_cache("loss_dict", loss_rate=loss_rate)
                    prior_entropy = coder.get_raw_cache("metric_dict").pop("prior_entropy", None)
                    if prior_entropy is not None:
                        self.update_cache("metric_dict", prior_entropy=prior_entropy)
                else:
                    # Other codecs should not update input and prior
                    coder(input.detach(), prior=prior.detach() if prior is not None else None, **kwargs)

        elif self.training and blend_weight.max() < self.blend_weight_one_hot_threshold:
            loss_rate_total = 0
            prior_entropy_total = 0
            for coder_idx, weight in enumerate(blend_weight):
                coder = self.coders[coder_idx]
                ret = coder(input, prior=prior, **kwargs)

                loss_rate = coder.get_raw_cache("loss_dict").pop("loss_rate", None)
                if loss_rate is not None:
                    loss_rate_total += loss_rate * weight
                prior_entropy = coder.get_raw_cache("metric_dict").pop("prior_entropy", None)
                if prior_entropy is not None:
                    prior_entropy_total += prior_entropy * weight
            self.update_cache("loss_dict", loss_rate=loss_rate_total)
            self.update_cache("metric_dict", prior_entropy=prior_entropy_total)

        else:
            coder_idx = blend_weight.argmax().item()
            coder = self.coders[coder_idx]
            ret = coder(input, prior=prior, **kwargs)
        
            prior_entropy = coder.get_raw_cache("metric_dict").pop("prior_entropy", None)
            if prior_entropy is not None:
                self.update_cache("metric_dict", prior_entropy=prior_entropy)

        return ret

    def encode(self, input : torch.Tensor, *args, prior : torch.Tensor = None, blend_weight : Optional[Any] = None, **kwargs) -> bytes:
        if blend_weight is None:
            blend_weight = torch.softmax(self.default_blend_weight)

        coder_idx = blend_weight.argmax().item()
        return self.coders[coder_idx].encode(input, prior=prior, **kwargs)

    def decode(self, byte_string : bytes, *args, prior : torch.Tensor = None, blend_weight : Optional[Any] = None, **kwargs) -> torch.Tensor:
        if blend_weight is None:
            blend_weight = torch.softmax(self.default_blend_weight)

        coder_idx = blend_weight.argmax().item()
        return self.coders[coder_idx].decode(byte_string, prior=prior, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        for coder in self.coders:
            coder.update_state(*args, **kwargs)
        return super().update_state(*args, **kwargs)


class GaussianPGMPriorCoderImpl(object):
    # TODO: param dequantize?
    def __init__(self, *args, lower_bound_scale=0.11, use_logvar_scale=False, use_zero_mean=False, mean_scale_split_method="chunk", inverse_mean_scale=False, scale_table=None, **kwargs) -> None:
        self.lower_bound_scale = LowerBound(lower_bound_scale)
        self.use_logvar_scale = use_logvar_scale
        self.use_zero_mean = use_zero_mean
        self.mean_scale_split_method = mean_scale_split_method
        self.inverse_mean_scale = inverse_mean_scale
        # TODO: can we reparent this class to NNTrainableModule so that scale_table gets proper device?
        self.scale_table = get_scale_table() if scale_table is None else torch.as_tensor(scale_table)
        if self.use_logvar_scale:
            self.scale_table = self.scale_table.log()
    
    @property
    def num_dist_params(self):
        return 1 if self.use_zero_mean else 2

    def _params_transform_for_dist(self, params) -> Any:
        if self.use_zero_mean:
            # return params
            means, scales = torch.zeros_like(params), params
        else:
            if self.mean_scale_split_method == "chunk":
                params = params.reshape(params.shape[0], 2, params.shape[1] // 2, *params.shape[2:]).movedim(1, -1)
            else:
                params = params.reshape(params.shape[0], params.shape[1] // 2, 2, *params.shape[2:]).movedim(2, -1)
            if self.inverse_mean_scale:
                params = params[..., [1,0]]
            # return params

            means, scales = params.chunk(2, dim=-1)
            means = means.squeeze(-1)
            scales = scales.squeeze(-1)

        if self.use_logvar_scale:
            scales = torch.exp(scales)

        return means, scales

    def _params_to_dist(self, params) -> D.Distribution:
        # if self.use_zero_mean:
        #     scales = params
        #     means = torch.zeros_like(scales)
        # else:
        #     if self.mean_scale_split_method == "chunk":
        #         means, scales = params.chunk(2, dim=1)
        #     else:
        #         means, scales = params.reshape(params.shape[0], params.shape[1] // 2, 2, *params.shape[2:])\
        #             .chunk(2, dim=2)
        #         means = means.squeeze(2)
        #         scales = scales.squeeze(2)
        if isinstance(params, tuple) and len(params) == 2:
            means, scales = params
        else:
            means, scales = self._params_transform_for_dist(params)#.chunk(2, dim=-1)
        # means = means.squeeze(-1)
        # scales = scales.squeeze(-1)
        # if self.use_logvar_scale:
        #     scales = torch.exp(scales)
        scales = self.lower_bound_scale(scales)
        return D.Normal(means, scales)

    def _params_to_dist_and_offset(self, params : torch.Tensor) -> Tuple[D.Distribution, torch.Tensor]:
        if isinstance(params, tuple) and len(params) == 2:
            means, scales = params
        else:
            means, scales = self._params_transform_for_dist(params)#.chunk(2, dim=-1)
        # means = means.squeeze(-1)
        # scales = scales.squeeze(-1)
        # if self.use_logvar_scale:
        #     scales = torch.exp(scales)
        scales = self.lower_bound_scale(scales)
        return D.Normal(torch.zeros_like(scales), scales), means

    def _init_dist_params(self) -> torch.Tensor:
        if self.use_zero_mean:
            return self.scale_table.to(device=self.device)
        else:
            if self.inverse_mean_scale:
                return torch.stack([self.scale_table, torch.zeros_like(self.scale_table)], dim=-1).to(device=self.device)
            else:
                return torch.stack([torch.zeros_like(self.scale_table), self.scale_table], dim=-1).to(device=self.device)
        # raise NotImplementedError()

    def _select_best_indexes(self, params) -> torch.LongTensor:
        if isinstance(params, tuple) and len(params) == 2:
            means, scales = params
        else:
            means, scales = self._params_transform_for_dist(params)#.chunk(2, dim=-1)
        # means = means.squeeze(-1)
        # scales = scales.squeeze(-1)

        # if self.use_logvar_scale:
        #     scales = torch.exp(scales)
        # scales = self.lower_bound_scale(scales) # TODO: no need for lower bound here, I suppose?

        # CompressAI implementation, Maybe faster?
        # indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        # for s in self.scale_table[:-1]:
        #     indexes -= (scales <= s).int()
        # return indexes

        indexes = (scales.reshape(-1).unsqueeze(-1) - self.scale_table.type_as(scales).unsqueeze(0)).abs().argmin(-1)
        return indexes.reshape_as(scales)


class GaussianChannelGroupPGMPriorCoder(GaussianPGMPriorCoderImpl, NNTrainablePGMPriorCoder):
    def __init__(self, *args, in_channels=256, in_groups=1, **kwargs):
        NNTrainablePGMPriorCoder.__init__(self, *args, **kwargs)
        kwargs_gaussian = dict(**kwargs)
        kwargs_gaussian.update(mean_scale_split_method="split_interleave")
        GaussianPGMPriorCoderImpl.__init__(self, *args, **kwargs_gaussian)
        self.in_channels = in_channels
        self.in_groups = in_groups

        self.input_mask = nn.Parameter(torch.zeros(self.in_groups))

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        return self.input_mask

    def _pgm_inference(self, input: torch.Tensor, pgm: Any, prior=None, **kwargs):
        # if self.training:
        # normal inference, use pgm in _get_likelihood_and_entropy and _pgm_generate instead
        return prior
        # else:
        #     pgm_mask = torch.repeat_interleave(pgm>0.5, self.in_channels // self.in_groups)
        #     # input_masked = input[:, pgm_mask]
        #     if not self.use_zero_mean:
        #         pgm_mask = pgm_mask.repeat(2)
        #     prior_masked = prior[:, pgm_mask]
        #     return prior_masked
    
    def _pgm_generate(self, byte_string : bytes, pgm : Any, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, **kwargs):
        pgm_mask = torch.repeat_interleave(pgm>0.5, self.in_channels // self.in_groups)
        prior_masked = prior[:, pgm_mask]
        data_buffer = torch.zeros(input_shape[0], prior_masked.shape[1], *input_shape[2:]).type_as(prior)
        # TODO: decoding
        data_buffer[:, pgm_mask] = super()._pgm_generate(byte_string, pgm, prior_masked, **kwargs)
        return data_buffer
    
    # def _get_likelihood_and_entropy(self, data, params, pgm=None, **kwargs):
    #     likelihood, entropy = super()._get_likelihood_and_entropy(data, params, pgm=pgm, **kwargs)
    #     pgm_weight = pgm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     # pgm_weight modify entropy
    #     entropy_modified = entropy.reshape(entropy.shape[0], self.in_groups, self.in_channels // self.in_groups, -1) * pgm_weight
    #     return likelihood, entropy_modified.reshape_as(entropy)


class TopoGroupPGMPriorCoder(NNTrainablePGMPriorCoder):
    def __init__(self, *args, in_channels=256, use_autoregressive_encode=True, **kwargs):
        self.in_channels = in_channels
        self.use_autoregressive_encode = use_autoregressive_encode
        super().__init__(*args, **kwargs)

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, fast_mode : bool = False, **kwargs) -> Any:
        if pgm is not None:
            topo_groups = pgm
        else:
            # TODO: get shape from bytes?
            pgm_shape = (input_shape[0], self.in_channels, *input_shape[2:])
            topo_groups = torch.zeros(*input_shape)
        return topo_groups

    # TODO: rewrite encode/decode, discard this!
    # def _pgm_sort_topo(self, input : torch.Tensor, pgm : Any, **kwargs) -> Optional[torch.LongTensor]:
    #     return pgm

    # TODO: implement this in c++?
    def _topo_group_to_masks(self, data : torch.Tensor, pgm : torch.LongTensor) -> torch.BoolTensor:
        assert data.shape[2:] == pgm.shape[2:]
        # NOTE: Currently we use post-repeat as we want dist parameters in the same group of channels
        # TODO: Maybe we should explicitly define if we want pre-repeat or post-repeat on pgm channel dimensions
        pgm_reshape = pgm.unsqueeze(2).repeat(data.shape[0] // pgm.shape[0], 1, data.shape[1] // pgm.shape[1], 1, 1).reshape_as(data)
        return [(pgm_reshape==i) for i in range(pgm.max()+1)]

    # TODO: implement this in c++?
    def _encode_topo_sort(self, data : torch.IntTensor, indexes : torch.IntTensor, pgm : Any) -> Tuple[torch.IntTensor, torch.IntTensor]:
        data_list, indexes_list = [], []
        # pgm_reshape = pgm.expand_as(data)
        for mask in self._topo_group_to_masks(data, pgm):
            data_list.append(data[mask])
            indexes_list.append(indexes[mask])
        return torch.cat(data_list), torch.cat(indexes_list)
    
    def _pgm_inference(self, input : torch.Tensor, pgm : Any, prior : Optional[torch.Tensor] = None, **kwargs):
        return self._pgm_inference_full(input, pgm, prior=prior)

    def _pgm_inference_full(self, input : torch.Tensor, pgm : Any, prior : Optional[torch.Tensor] = None, **kwargs):
        raise NotImplementedError()

    def _pgm_inference_group_mask(self, input : torch.Tensor, topo_group_mask : torch.BoolTensor, pgm : Optional[Any] = None, prior : Optional[torch.Tensor] = None, **kwargs):
        raise NotImplementedError()

    # NOTE: We implement autoregressive encoding for more accurate data offset! (maybe set a parameter?)
    def _encode_with_pgm(self, input : torch.Tensor, *args, prior : torch.Tensor = None, pgm : Optional[Any] = None, quantizer_params: Optional[Any] = None, **kwargs) -> bytes:
        if self.use_autoregressive_encode:
            input_no_quant = self._data_preprocess(input, quantize=False, quantizer_params=quantizer_params)
            data_buffer = torch.zeros(input.shape[0], self.in_channels, *input.shape[2:], device=self.device)
            pgm = self._get_pgm(input, prior=prior, input_shape=input.shape, pgm=pgm, fast_mode=True)
            # TODO: we should split data_buffer and prior in one step for best efficiency
            topo_group_masks = self._topo_group_to_masks(data_buffer, pgm)
            # last_group = None
            # TODO: update grouping method
            data_list, indexes_list = [], []
            for idx, mask in enumerate(topo_group_masks):
                # if last_group is None: last_group = torch.zeros_like(data_buffer[mask])
                params = self._pgm_inference_group_mask(data_buffer, mask, pgm=pgm, prior=prior)
                if self.pgm_input_dequantized:
                    params = self._dist_param_quant_reparam(params, quantizer_params=quantizer_params)
                indexes = self._build_coding_indexes(params)[mask].contiguous()
                data_offsets = self._build_coding_offsets(params)[mask].contiguous()
                data_coding = self._data_preprocess(input_no_quant[mask] - data_offsets, transform=False, quantize=True, quantizer_params=quantizer_params)
                # NOTE: it seems that caching somehow causes decompression to fail! Look into this problem in c++!
                # self.ans_encoder.encode_with_indexes(
                #     data_coding.detach().cpu().numpy().astype(np.int32),
                #     indexes.detach().cpu().numpy().astype(np.int32),
                #     cache=True,
                # )
                data_list.append(data_coding)
                indexes_list.append(indexes)
                if self.pgm_input_dequantized:
                    data_buffer[mask] = self._data_postprocess(data_coding + data_offsets, quantizer_params=quantizer_params)
                else:
                    data_buffer[mask] = data_coding + data_offsets
            # return self.ans_encoder.flush()
            data = torch.cat(data_list).contiguous().detach().cpu().numpy().astype(np.int32)
            indexes = torch.cat(indexes_list).contiguous().detach().cpu().numpy().astype(np.int32)
            return self.ans_encoder.encode_with_indexes(data, indexes)
        else:
            return super()._encode_with_pgm(input, *args, prior=prior, pgm=pgm, **kwargs)

    def _pgm_generate(self, byte_string : bytes, pgm : Any, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, quantizer_params: Optional[Any] = None, **kwargs):
        # TODO: if prior is None?
        self.ans_decoder.set_stream(byte_string)
        data_buffer = torch.zeros(input_shape[0], self.in_channels, *input_shape[2:], device=self.device)
        # TODO: we should split data_buffer and prior in one step for best efficiency
        with self.profiler.start_time_profile("topo_group_to_masks"):
            topo_group_masks = self._topo_group_to_masks(data_buffer, pgm)
        # last_group = None
        # TODO: update grouping method
        for idx, mask in enumerate(topo_group_masks):
            # if last_group is None: last_group = torch.zeros_like(data_buffer[mask])
            with self.profiler.start_time_profile("pgm_generate_pgm_inference_group_mask"):
                params = self._pgm_inference_group_mask(data_buffer, mask, pgm=pgm, prior=prior)
                if self.pgm_input_dequantized:
                    params = self._dist_param_quant_reparam(params, quantizer_params=quantizer_params)
            with self.profiler.start_time_profile("pgm_generate_build_coding_indexes"):
                indexes = self._build_coding_indexes(params)[mask].contiguous()
                # with self.profiler.start_time_profile("pgm_generate_indexes_mask"):
                #     indexes = indexes[mask].contiguous()
            with self.profiler.start_time_profile("pgm_generate_build_coding_offsets"):
                data_offsets = self._build_coding_offsets(params)[mask].contiguous()
            with self.profiler.start_time_profile("pgm_generate_coding"):
                symbols = self.ans_decoder.decode_stream(indexes.detach().cpu().numpy().astype(np.int32))
            with self.profiler.start_time_profile("pgm_generate_coding_post"):
                data_quant = torch.as_tensor(symbols).type_as(data_buffer) 
                # last_group = self._data_postprocess(symbols)
                data_decode = data_quant + (data_offsets if self.use_autoregressive_encode else data_offsets.round())
                if self.pgm_input_dequantized:
                    data_decode = self._data_postprocess(data_decode, quantizer_params=quantizer_params)
                data_buffer[mask] = data_decode
        if not self.pgm_input_dequantized:
            data_buffer = self._data_postprocess(data_buffer, quantizer_params=quantizer_params)
        return data_buffer

class GaussianChannelGroupMaskConv2DTopoGroupPGMPriorCoder(GaussianPGMPriorCoderImpl, TopoGroupPGMPriorCoder):
    def __init__(self, *args, 
                 in_channels=256, 
                 channel_groups=1, # TODO: spatial groups?
                 default_topo_group_method="none",
                 default_num_topo_groups=-1,
                 topo_group_context_model : Optional[TopoGroupDynamicMaskConv2dContextModel] = None,
                 kernel_size=5, 
                 use_param_merger=True, 
                 use_joint_ar_model_impl=False,
                 param_merger_expand_bottleneck=False,
                 freeze_context_model=False,
                 detach_context_model=False,
                 use_dynamic_kernel_predictor=False,
                 dynamic_kernel_predictor_mid_channels=256,
                 pgm_include_dynamic_kernel=False,
                 pgm_include_dynamic_kernel_full=False,
                 pgm_dynamic_kernel_enable_tiling=False,
                 pgm_dynamic_kernel_add_self=False,
                 topo_group_predictor=None,
                 topo_group_predictor_add_default=False,
                 topo_group_predictor_add_default_modifier=1.0,
                 topo_group_predictor_enable_reset=False,
                 topo_group_predictor_reset_entropy_threshold=1e-3,
                 topo_group_predictor_reset_num_iter=0,
                 topo_group_predictor_reset_window_size=20,
                 topo_group_predictor_include_dynamic_kernel=False,
                 topo_group_predictor_is_batched=False,
                 topo_group_predictor_batch_score_window_size=20,
                 topo_group_logits_modifier=1.0,
                 topo_group_allow_continuous=False,
                 topo_group_continuous_smooth_func="st",
                 topo_group_continuous_sampling=False,
                 topo_group_continuous_sampling_method="relaxed",
                 topo_group_continuous_add_noise=False,
                 continuous_topo_groups_training_use_uniform_noise=False,
                 training_pgm_logits_use_random_prob=0.0,
                 training_pgm_logits_use_random_num_iter=0,
                 training_pgm_logits_use_random_entropy_threshold=0.05,
                 training_mc_sampling_share_sample_batch=False,
                 training_mc_sampling_add_default_as_sample=False,
                 training_mc_sampling_add_default_as_sample_prob=1.0,
                 gs_temp=0.5, gs_temp_anneal=False,
                 relax_temp=1.0, relax_temp_anneal=False, relax_temp_threshold=0.0,
                 **kwargs):
        TopoGroupPGMPriorCoder.__init__(self, *args, in_channels=in_channels, **kwargs)
        self.channel_groups = channel_groups
        self.default_topo_group_method = default_topo_group_method
        self.default_num_topo_groups = default_num_topo_groups
        self.kernel_size = kernel_size
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.use_param_merger = use_param_merger
        self.use_joint_ar_model_impl = use_joint_ar_model_impl
        self.param_merger_expand_bottleneck = param_merger_expand_bottleneck
        self.freeze_context_model = freeze_context_model
        self.detach_context_model = detach_context_model
        self.use_dynamic_kernel_predictor = use_dynamic_kernel_predictor
        self.dynamic_kernel_predictor_mid_channels = dynamic_kernel_predictor_mid_channels
        self.pgm_include_dynamic_kernel = pgm_include_dynamic_kernel
        self.pgm_include_dynamic_kernel_full = pgm_include_dynamic_kernel_full
        self.pgm_dynamic_kernel_enable_tiling = pgm_dynamic_kernel_enable_tiling
        self.pgm_dynamic_kernel_add_self = pgm_dynamic_kernel_add_self

        kwargs_gaussian = dict(**kwargs)
        if self.use_joint_ar_model_impl:
            kwargs_gaussian.update(mean_scale_split_method="chunk", inverse_mean_scale=True)
        else:
            kwargs_gaussian.update(mean_scale_split_method="split_interleave")
        GaussianPGMPriorCoderImpl.__init__(self, *args, **kwargs_gaussian)

        # NOTE: tmp solution for device of scale_table
        # scale_table = self.scale_table
        # delattr(self, "scale_table")
        # self.register_buffer("scale_table", scale_table)

        self.topo_group_predictor = topo_group_predictor
        self.topo_group_predictor_add_default = topo_group_predictor_add_default
        self.topo_group_predictor_add_default_modifier = topo_group_predictor_add_default_modifier
        self.topo_group_predictor_enable_reset = topo_group_predictor_enable_reset
        self.topo_group_predictor_reset_entropy_threshold = topo_group_predictor_reset_entropy_threshold
        self.topo_group_predictor_reset_num_iter = topo_group_predictor_reset_num_iter
        self.topo_group_predictor_reset_window_size = topo_group_predictor_reset_window_size
        self.topo_group_predictor_include_dynamic_kernel = topo_group_predictor_include_dynamic_kernel
        self.topo_group_predictor_is_batched = topo_group_predictor_is_batched
        self.topo_group_predictor_batch_score_window_size = topo_group_predictor_batch_score_window_size
        self.topo_group_logits_modifier = topo_group_logits_modifier
        self.topo_group_allow_continuous = topo_group_allow_continuous
        self.topo_group_continuous_smooth_func = topo_group_continuous_smooth_func
        self.topo_group_continuous_sampling = topo_group_continuous_sampling
        self.topo_group_continuous_sampling_method = topo_group_continuous_sampling_method
        self.topo_group_continuous_add_noise = topo_group_continuous_add_noise
        self.continuous_topo_groups_training_use_uniform_noise = continuous_topo_groups_training_use_uniform_noise

        # compability
        if self.topo_group_predictor_include_dynamic_kernel:
            self.pgm_include_dynamic_kernel = True
            self.pgm_dynamic_kernel_enable_tiling = True

        self.training_pgm_logits_use_random_prob = training_pgm_logits_use_random_prob
        self.training_pgm_logits_use_random_num_iter = training_pgm_logits_use_random_num_iter
        self.training_pgm_logits_use_random_entropy_threshold = training_pgm_logits_use_random_entropy_threshold
        self.training_mc_sampling_share_sample_batch = training_mc_sampling_share_sample_batch
        self.training_mc_sampling_add_default_as_sample = training_mc_sampling_add_default_as_sample
        self.training_mc_sampling_add_default_as_sample_prob = training_mc_sampling_add_default_as_sample_prob

        if self.training_pgm_logits_use_random_num_iter or self.topo_group_predictor_reset_num_iter:
            self.register_buffer("training_iter_cnt", torch.zeros(1, dtype=torch.int))

        if self.topo_group_predictor_is_batched:
            self.register_buffer("topo_group_predictor_batch_idx", torch.zeros(1, dtype=torch.long))
            self.topo_group_predictor_batch_scores = []

        if self.topo_group_predictor is not None:
            # assert self.training_mc_sampling, "Currently topo_group_predictor should only be optimized through Monte-Carlo loss!"
            topo_group_predictor_result = self.topo_group_predictor()
            if isinstance(topo_group_predictor_result, torch.Tensor):
                # For retrieval from state dict
                self.register_buffer("topo_group_predictor_cache", topo_group_predictor_result)
            else:
                # TODO: cannot retrieve from state dict! Need to rerun topo_group_predictor after self.topo_group_predictor is changed!
                self.topo_group_predictor_cache = topo_group_predictor_result
            if self.topo_group_predictor_enable_reset:
                self.topo_group_predictor_base = [copy.deepcopy(topo_group_predictor)]
                self.topo_group_predictor_best = [copy.deepcopy(topo_group_predictor)]
                self.topo_group_predictor_reset_scores = []
                self.topo_group_predictor_reset_activate = False

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.relax_temp = relax_temp
        self.relax_temp_threshold = relax_temp_threshold

        if self.default_topo_group_method == "none":
            pass
        elif self.default_topo_group_method == "random":
            pass
        elif self.default_topo_group_method == "channelwise-random":
            pass
        elif self.default_topo_group_method == "scanline":
            pass
            # self.channel_groups = 1
        elif self.default_topo_group_method == "zigzag":
            pass
            # self.channel_groups = 1
        elif self.default_topo_group_method == "checkerboard":
            # self.channel_groups = 1
            self.default_num_topo_groups = 2
        elif self.default_topo_group_method == "half-checkerboard":
            # self.channel_groups = 1
            self.default_num_topo_groups = 2
        elif self.default_topo_group_method == "halfinv-checkerboard":
            # self.channel_groups = 1
            self.default_num_topo_groups = 2
        elif self.default_topo_group_method == "quarter-checkerboard":
            # self.channel_groups = 1
            self.default_num_topo_groups = 2
        elif self.default_topo_group_method == "interlace-checkerboard":
            # self.channel_groups = 1
            self.default_num_topo_groups = 2
        elif self.default_topo_group_method == "raster2x2":
            # self.channel_groups = 1
            self.default_num_topo_groups = 4
        elif self.default_topo_group_method == "channelwise":
            self.default_num_topo_groups = self.channel_groups
        elif self.default_topo_group_method == "channelwise-checkerboard":
            self.default_num_topo_groups = self.channel_groups * 2
        elif self.default_topo_group_method == "channelwise-scanline":
            pass
        elif self.default_topo_group_method == "channelwise-g10":
            self.channel_groups = self.in_channels // 16
            assert self.channel_groups >= 9
            self.default_num_topo_groups = 10
        elif self.default_topo_group_method == "elic":
            self.channel_groups = self.in_channels // 16
            assert self.channel_groups >= 8
            self.default_num_topo_groups = 10
        else:
            raise NotImplementedError(f"Unknown default_topo_group_method {self.default_topo_group_method}")

        self.out_channels = self.in_channels * self.num_dist_params

        self.topo_group_context_model = topo_group_context_model
        if topo_group_context_model is not None:
            if self.freeze_context_model:
                for param in self.topo_group_context_model.parameters():
                    param.requires_grad = False
        else:
            # TODO: only for compability
            self.conv_kernel_weight = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, kernel_size, kernel_size))
            self.conv_kernel_bias = nn.Parameter(torch.zeros(self.out_channels))
            self.context_prediction = TopoGroupDynamicMaskConv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, 
                                                                dynamic_channel_groups=self.channel_groups, 
                                                                allow_continuous_topo_groups=self.topo_group_allow_continuous,
                                                                continuous_topo_groups_training_use_uniform_noise=self.continuous_topo_groups_training_use_uniform_noise,
                                                                continuous_topo_groups_smooth_func=self.topo_group_continuous_smooth_func,
                                                                detach_context_model=self.detach_context_model)
            # self.context_prediction.weight.data = self.conv_kernel_weight.data
            # self.context_prediction.bias.data = self.conv_kernel_bias.data
            if self.freeze_context_model:
                for param in self.context_prediction.parameters():
                    param.requires_grad = False
            
            if self.use_dynamic_kernel_predictor:
                self.dynamic_kernel_predictor = nn.Sequential(
                    nn.Conv2d(self.channel_groups, self.dynamic_kernel_predictor_mid_channels, self.kernel_size, padding=self.padding),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(self.dynamic_kernel_predictor_mid_channels, self.dynamic_kernel_predictor_mid_channels, self.kernel_size, padding=self.padding),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(self.dynamic_kernel_predictor_mid_channels, self.out_channels * self.in_channels * self.kernel_size * self.kernel_size + self.out_channels, 1),
                )

            if self.use_param_merger:
                self.param_merger_out_channels = self.out_channels # // self.channel_groups
                if self.use_joint_ar_model_impl: #self.channel_groups == 1:
                    assert self.channel_groups == 1
                    self.entropy_parameters = nn.Sequential(
                        nn.Conv2d(self.param_merger_out_channels * 2, self.param_merger_out_channels * 5 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self.param_merger_out_channels * 5 // 3, self.param_merger_out_channels * 4 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self.param_merger_out_channels * 4 // 3, self.param_merger_out_channels, 1),
                    )
                else:
                    bottleneck_channels = self.param_merger_out_channels * 4 if param_merger_expand_bottleneck else self.param_merger_out_channels * 2
                    self.param_merger = nn.Sequential(
                        TopoGroupDynamicMaskConv2d(self.param_merger_out_channels * 2, bottleneck_channels, 1, 
                                                dynamic_channel_groups=self.channel_groups * 2, 
                                                allow_same_topogroup_conv=True,
                                                allow_continuous_topo_groups=self.topo_group_allow_continuous,
                                                continuous_topo_groups_training_use_uniform_noise=self.continuous_topo_groups_training_use_uniform_noise,
                                                continuous_topo_groups_smooth_func=self.topo_group_continuous_smooth_func,
                                                detach_context_model=self.detach_context_model),
                        nn.LeakyReLU(inplace=True),
                        TopoGroupDynamicMaskConv2d(bottleneck_channels, bottleneck_channels, 1,
                                                dynamic_channel_groups=self.channel_groups * 2, 
                                                allow_same_topogroup_conv=True,
                                                allow_continuous_topo_groups=self.topo_group_allow_continuous,
                                                continuous_topo_groups_training_use_uniform_noise=self.continuous_topo_groups_training_use_uniform_noise,
                                                continuous_topo_groups_smooth_func=self.topo_group_continuous_smooth_func,
                                                detach_context_model=self.detach_context_model),
                        nn.LeakyReLU(inplace=True),
                        TopoGroupDynamicMaskConv2d(bottleneck_channels, self.param_merger_out_channels * 2, 1,
                                                dynamic_channel_groups=self.channel_groups * 2, 
                                                allow_same_topogroup_conv=True,
                                                allow_continuous_topo_groups=self.topo_group_allow_continuous,
                                                continuous_topo_groups_training_use_uniform_noise=self.continuous_topo_groups_training_use_uniform_noise,
                                                continuous_topo_groups_smooth_func=self.topo_group_continuous_smooth_func,
                                                detach_context_model=self.detach_context_model),
                    )

                    if self.freeze_context_model:
                        for param in self.param_merger.parameters():
                            param.requires_grad = False

    def _apply(self, fn):
        super()._apply(fn)
        # currently we refill topo_group_predictor_cache for every _apply function call
        if self.topo_group_predictor is not None:
            self.topo_group_predictor_cache = self.topo_group_predictor()

    # for profiling
    # def _select_best_indexes(self, params) -> torch.LongTensor:
    #     with self.profiler.start_time_profile("select_best_indexes_params_preprocess"):
    #         means, scales = self._params_transform_for_dist(params).chunk(2, dim=-1)
    #         means = means.squeeze(-1)
    #         scales = scales.squeeze(-1)

    #         if self.use_logvar_scale:
    #             scales = torch.exp(scales)
    #         # scales = self.lower_bound_scale(scales) # TODO: no need for lower bound here, I suppose?

    #     with self.profiler.start_time_profile("select_best_indexes_scales_reshape"):
    #         scales = scales.reshape(-1)
    #     with self.profiler.start_time_profile("select_best_indexes_scale_table_reshape"):
    #         scale_table = self.scale_table.type_as(scales)

    #     with self.profiler.start_time_profile("select_best_indexes_scales_delta"):
    #         scales_delta = scales.unsqueeze(-1) - scale_table.unsqueeze(0)

    #     with self.profiler.start_time_profile("select_best_indexes_build_indexes"):
    #         # CompressAI implementation. Maybe faster?
    #         # indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
    #         # for s in self.scale_table[:-1]:
    #         #     indexes -= (scales <= s).int()
    #         # return indexes

    #         indexes = scales_delta.abs().argmin(-1)
    #         return indexes.reshape_as(means)

    def _dist_param_quant_reparam(self, params : torch.Tensor, quantizer_params: Optional[torch.Tensor] = None, **kwargs):
        quantizer_params = self.quantizer_params if quantizer_params is None else quantizer_params
        means, scales = self._params_transform_for_dist(params)

        if self.quantizer_type == "uniform":
            means = (means - quantizer_params[0]) / quantizer_params[2]
            scales = scales / quantizer_params[2]
        elif self.quantizer_type == "uniform_scale":
            means = means / quantizer_params
            scales = scales / quantizer_params
        elif self.quantizer_type == "nonuniform":
            raise NotImplementedError()
        elif self.quantizer_type == "vector":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Unknown quantizer_type {self.quantizer_type}")

        return (means, scales)

    def _preprocess_pgm(self, pgm, pgm_shape, fast_mode=False):
        # use default pgm
        if pgm is None:
            pgm = self._get_default_pgm(pgm_shape)
            if self.pgm_include_dynamic_kernel:
                if self.pgm_include_dynamic_kernel_full:
                    weight_params, bias_params = [self.context_prediction.weight.unsqueeze(0)], [self.context_prediction.bias.unsqueeze(0)]
                    for name, param in self.param_merger.named_parameters():
                        if ".weight" in name: weight_params.append(param.unsqueeze(0))
                        if ".bias" in name: bias_params.append(param.unsqueeze(0))
                    pgm = pgm, weight_params, bias_params
                else:
                    pgm = pgm, self.context_prediction.weight.unsqueeze(0), self.context_prediction.bias.unsqueeze(0)
            return pgm

        if self.pgm_include_dynamic_kernel:
            pgm, dynamic_kernel_weight, dynamic_kernel_bias = pgm
            if self.pgm_dynamic_kernel_add_self:
                if self.pgm_include_dynamic_kernel_full:
                    for idx, (weight, bias) in enumerate(zip(dynamic_kernel_weight, dynamic_kernel_bias)):
                        # weight = weight.reshape(weight.shape[0], self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
                        # bias = bias.reshape(bias.shape[0], self.channel_groups, self.out_channels // self.channel_groups, -1)
                        if idx == 0:
                            kernel_weight = self.context_prediction.weight.unsqueeze(0).detach()
                            kernel_bias = self.context_prediction.bias.unsqueeze(0).detach()
                        else:
                            kernel_weight = self.param_merger[2*(idx-1)].weight.unsqueeze(0).detach()
                            kernel_bias = self.param_merger[2*(idx-1)].bias.unsqueeze(0).detach()
                        if self.pgm_dynamic_kernel_enable_tiling:
                            kernel_weight = kernel_weight.reshape(1, -1, 1, 1)
                            kernel_bias = kernel_bias.reshape(1, -1, 1, 1)
                        dynamic_kernel_weight[idx] = dynamic_kernel_weight[idx] + kernel_weight
                        dynamic_kernel_bias[idx] = dynamic_kernel_bias[idx] + kernel_bias
                else:
                    kernel_weight = self.context_prediction.weight.unsqueeze(0).detach()
                    kernel_bias = self.context_prediction.bias.unsqueeze(0).detach()
                    if self.pgm_dynamic_kernel_enable_tiling:
                        kernel_weight = kernel_weight.reshape(1, -1, 1, 1)
                        kernel_bias = kernel_bias.reshape(1, -1, 1, 1)
                    dynamic_kernel_weight = dynamic_kernel_weight + kernel_weight
                    dynamic_kernel_bias = dynamic_kernel_bias + kernel_bias

        is_topo_groups_logits = torch.is_floating_point(pgm)
        # expand input topogroups to required size
        if is_topo_groups_logits:
            # move logit dimension to last
            if pgm.ndim == len(pgm_shape):
                pgm = pgm.reshape(pgm.shape[0], self.channel_groups, pgm.shape[1] // self.channel_groups, *pgm.shape[2:])\
                    .movedim(2, -1)
            logits_size = pgm.shape[-1]
            if fast_mode:
                is_topo_groups_logits = False
                pgm = pgm.argmax(-1)
            else:
                pgm_shape += (logits_size, )
            # assert topo_groups.ndim == len(pgm_shape)
        else:
            assert pgm.ndim == len(pgm_shape)
        assert pgm.shape[1] == pgm_shape[1] # check channel groups equal

        topo_groups = pgm
        topo_groups_batch_size = topo_groups.shape[0]

        # trim spatial size
        if topo_groups.shape[2] >= pgm_shape[2]:
            topo_groups = topo_groups[:, :, :pgm_shape[2]]
        if topo_groups.shape[3] >= pgm_shape[3]:
            topo_groups = topo_groups[:, :, :, :pgm_shape[3]]
        patch_height, patch_width = topo_groups.shape[2:4]
        # tile spatial size
        if topo_groups.shape[2] < pgm_shape[2] or topo_groups.shape[3] < pgm_shape[3]:
            if is_topo_groups_logits:
                topo_groups = topo_groups.movedim(-1, 1).reshape(topo_groups_batch_size * logits_size, -1)
            topo_groups = topo_groups.reshape(topo_groups.shape[0], -1, 1).repeat(1, 1, math.floor(pgm_shape[2] / patch_height) * math.floor(pgm_shape[3] / patch_width))
            # NOTE: F.fold only supports float?
            topo_groups_fold = F.fold(topo_groups.float(), pgm_shape[2:4], (patch_height, patch_width), stride=(patch_height, patch_width))
            topo_groups = topo_groups_fold.type_as(pgm)
            if is_topo_groups_logits:
                topo_groups = topo_groups.reshape(topo_groups_batch_size, logits_size, *topo_groups.shape[1:]).movedim(1, -1)
            if self.pgm_include_dynamic_kernel and self.pgm_dynamic_kernel_enable_tiling:
                dynamic_kernel_weight = dynamic_kernel_weight.reshape(topo_groups.shape[0], -1, 1).repeat(1, 1, math.floor(pgm_shape[2] / patch_height) * math.floor(pgm_shape[3] / patch_width))
                dynamic_kernel_weight = F.fold(dynamic_kernel_weight, pgm_shape[2:4], (patch_height, patch_width), stride=(patch_height, patch_width))
                dynamic_kernel_bias = dynamic_kernel_bias.reshape(topo_groups.shape[0], -1, 1).repeat(1, 1, math.floor(pgm_shape[2] / patch_height) * math.floor(pgm_shape[3] / patch_width))
                dynamic_kernel_bias = F.fold(dynamic_kernel_bias, pgm_shape[2:4], (patch_height, patch_width), stride=(patch_height, patch_width))
        # expand batch size if needed
        if topo_groups.shape[0] == 1:
            topo_groups = topo_groups.expand(*pgm_shape)
        if self.training_mc_sampling and topo_groups.shape[0] == pgm_shape[0] // self.training_mc_num_samples:
            topo_groups = topo_groups.repeat(self.training_mc_num_samples, *([1]*(topo_groups.ndim-1)))

        if is_topo_groups_logits and self.topo_group_predictor is not None and self.topo_group_predictor_add_default:
            topo_groups = topo_groups + F.one_hot(self._get_default_pgm(pgm_shape[:-1]), topo_groups.shape[-1]).type_as(topo_groups) * self.topo_group_predictor_add_default_modifier

        if self.training and is_topo_groups_logits:
            if self.training_pgm_logits_use_random_prob > 0 or self.training_pgm_logits_use_random_num_iter > 0:
                topo_groups_logprob = torch.log_softmax(topo_groups, dim=-1)
                topo_groups_entropy = (-topo_groups_logprob * topo_groups_logprob.exp()).mean()
                if topo_groups_entropy > self.training_pgm_logits_use_random_entropy_threshold:
                    if (self.training_pgm_logits_use_random_prob > 0 and random.random() < self.training_pgm_logits_use_random_prob)\
                        or self.training_iter_cnt < self.training_pgm_logits_use_random_num_iter:
                        default_num_topo_groups = self.default_num_topo_groups if self.default_num_topo_groups > 1 else self.channel_groups * np.prod(pgm_shape[2:4])
                        topo_groups = torch.randint(0, default_num_topo_groups, pgm_shape[:-1])
                        # NOTE: if dynamic kernel adds self, optimize self kernels when using random topogroup
                        if self.pgm_include_dynamic_kernel and self.pgm_dynamic_kernel_add_self:
                            if self.pgm_include_dynamic_kernel_full:
                                dynamic_kernel_weight, dynamic_kernel_bias = [self.context_prediction.weight.unsqueeze(0)], [self.context_prediction.bias.unsqueeze(0)]
                                for name, param in self.param_merger.named_parameters():
                                    if ".weight" in name: dynamic_kernel_weight.append(param.unsqueeze(0))
                                    if ".bias" in name: dynamic_kernel_bias.append(param.unsqueeze(0))
                            else:
                                dynamic_kernel_weight, dynamic_kernel_bias = self.context_prediction.weight.unsqueeze(0), self.context_prediction.bias.unsqueeze(0)

        if self.pgm_include_dynamic_kernel:
            return topo_groups, dynamic_kernel_weight, dynamic_kernel_bias
        else:
            return topo_groups

    def _get_default_pgm(self, pgm_shape):
        # TODO: get shape from bytes?
        topo_groups = torch.zeros(pgm_shape[0], self.channel_groups, *pgm_shape[-2:], dtype=torch.long, device=self.device)
        if self.default_topo_group_method == "none":
            pass
        elif self.default_topo_group_method == "random":
            # set default_num_topo_groups as number of topogroups if not specified
            default_num_topo_groups = self.default_num_topo_groups if self.default_num_topo_groups > 1 else self.channel_groups * np.prod(pgm_shape[-2:])
            topo_groups += torch.randint_like(topo_groups, default_num_topo_groups)
        elif self.default_topo_group_method == "channelwise-random":
            # set default_num_topo_groups as number of topogroups if not specified
            spatial_num_topo_groups = self.default_num_topo_groups // self.channel_groups if self.default_num_topo_groups > 1 else np.prod(pgm_shape[-2:])
            for i in range(self.channel_groups):
                topo_groups[:, i] = torch.randint_like(topo_groups[:, i], spatial_num_topo_groups) + i * spatial_num_topo_groups
        elif self.default_topo_group_method == "scanline":
            spatial_dim = np.prod(pgm_shape[-2:])
            topo_groups = torch.arange(0, spatial_dim).reshape(1, 1, *pgm_shape[-2:])\
                .repeat(pgm_shape[0], self.channel_groups, 1, 1).type_as(topo_groups)
        elif self.default_topo_group_method == "zigzag":
            zigzag_idx = torch.arange(0, pgm_shape[-2]).reshape(pgm_shape[-2], 1) + \
                torch.arange(0, pgm_shape[-1]).reshape(1, pgm_shape[-1])
            topo_groups = zigzag_idx.reshape(1, 1, *pgm_shape[-2:])\
                .repeat(pgm_shape[0], self.channel_groups, 1, 1).type_as(topo_groups)
        elif self.default_topo_group_method == "checkerboard":
            topo_groups[..., 0::2, 1::2] = 1
            topo_groups[..., 1::2, 0::2] = 1
        elif self.default_topo_group_method == "half-checkerboard":
            topo_groups.fill_(1)
            topo_groups[..., 1::2, 1::2] = 0
        elif self.default_topo_group_method == "halfinv-checkerboard":
            topo_groups[..., 1::2, 1::2] = 1
        elif self.default_topo_group_method == "quarter-checkerboard":
            topo_groups.fill_(1)
            topo_groups[..., 1::4, 3::4] = 0
            topo_groups[..., 3::4, 1::4] = 0
        elif self.default_topo_group_method == "interlace-checkerboard":
            for i in range(self.channel_groups):
                if i % 2 == 0:
                    topo_groups[..., i, 0::2, 0::2] = 1
                    topo_groups[..., i, 1::2, 1::2] = 1
                else:
                    topo_groups[..., i, 0::2, 1::2] = 1
                    topo_groups[..., i, 1::2, 0::2] = 1
        elif self.default_topo_group_method == "raster2x2":
            topo_groups[..., 0::2, 1::2] = 1
            topo_groups[..., 1::2, 0::2] = 2
            topo_groups[..., 1::2, 1::2] = 3
        elif self.default_topo_group_method == "channelwise":
            for i in range(self.channel_groups):
                topo_groups[:, i] = i
        elif self.default_topo_group_method == "channelwise-checkerboard":
            for i in range(self.channel_groups):
                topo_groups[:, i] = i * 2
                topo_groups[:, i, 1::2, 0::2] = i * 2 + 1
                topo_groups[:, i, 0::2, 1::2] = i * 2 + 1
        elif self.default_topo_group_method == "channelwise-scanline":
            spatial_dim = np.prod(pgm_shape[-2:])
            for i in range(self.channel_groups):
                topo_groups[:, i] = torch.arange(0, spatial_dim).reshape(1, 1, *pgm_shape[-2:])\
                    .repeat(pgm_shape[0], 1, 1, 1).type_as(topo_groups)\
                    + i * spatial_dim
        elif self.default_topo_group_method == "channelwise-g10":
            channel_group_splits = [1] * 9 + [self.channel_groups - 9]
            split_idx = 0
            for i, split in enumerate(channel_group_splits):
                topo_groups[:, split_idx:(split_idx+split)] = i
                split_idx += split
        elif self.default_topo_group_method == "elic":
            channel_group_splits = [1, 1, 2, 4, self.channel_groups - 8]
            split_idx = 0
            for i, split in enumerate(channel_group_splits):
                topo_groups[:, split_idx:(split_idx+split)] = i * 2
                topo_groups[:, split_idx:(split_idx+split), 1::2, 0::2] = i * 2 + 1
                topo_groups[:, split_idx:(split_idx+split), 0::2, 1::2] = i * 2 + 1
                split_idx += split
        return topo_groups

    def _get_pgm_from_predictor(self, predictor, pgm_shape):
        pgm = predictor()
        return pgm

    # TODO: cache pgm for faster computation according to fast_mode
    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, fast_mode : bool = False, **kwargs) -> Any:
        # pgm_shape = (input_shape[0], self.channel_groups, *input_shape[2:])
        # use single batch pgm for faster computation
        pgm_shape = (1, self.channel_groups, *input_shape[2:])
        if pgm is not None:
            # avoid passing samples during coding
            # if fast_mode and torch.is_floating_point(pgm):
            #     # TODO: check if pgm is logits
            #     pgm = pgm.argmax(-1)
            pgm_shape = (pgm_shape[0], self.channel_groups, *pgm_shape[2:])
            topo_groups = self._preprocess_pgm(pgm, pgm_shape, fast_mode=fast_mode)
        elif self.topo_group_predictor is not None:
            if self.training:
                if isinstance(self.topo_group_predictor, SurDistribution) and self.training_mc_sampling:
                    add_default = (self.training_mc_sampling_add_default_as_sample and random.random() < self.training_mc_sampling_add_default_as_sample_prob)
                    if self.training_mc_sampling_share_sample_batch:
                        num_samples = self.training_mc_num_samples
                        if add_default:
                            num_samples -= 1
                    else:
                        num_samples = input_shape[0]
                    pgm = self.topo_group_predictor.sample(num_samples)
                    if add_default:
                        pgm = torch.cat([
                            pgm,
                            self._get_default_pgm(pgm_shape)
                        ], dim=0)
                    log_prob = self.topo_group_predictor.log_prob(pgm)
                    if self.training_mc_sampling_share_sample_batch:
                        pgm = pgm.repeat(input_shape[0] // pgm.shape[0], 1, 1, 1)
                        log_prob = log_prob.repeat(input_shape[0] // self.training_mc_num_samples, 1, 1, 1)
                    self.update_cache(pgm_log_prob=log_prob)
                    self.topo_group_predictor_cache = pgm[log_prob.argmax()].unsqueeze(0) # take the max sample as cache
                elif self.topo_group_predictor_enable_reset:
                    # we only update topo_group_predictor_cache when a new best model is found!
                    if len(self.topo_group_predictor_reset_scores) > self.topo_group_predictor_reset_window_size:
                        # better than best
                        best_idx = np.argmin(np.array(self.topo_group_predictor_reset_scores).sum(0))
                        self.update_cache("moniter_dict", topo_group_predictor_reset_best_idx=best_idx)
                        if best_idx == 0:
                            self.topo_group_predictor_best[0].load_state_dict(self.topo_group_predictor.state_dict())
                            self.topo_group_predictor_cache = self._get_pgm_from_predictor(self.topo_group_predictor_best[0], pgm_shape)
                        # reset predictor
                        # for module in self.topo_group_predictor_base[0].modules():
                        #     if isinstance(module, nn.Conv2d):
                        #         nn.init.xavier_uniform_(module.weight.data, 1.)
                        self.topo_group_predictor.load_state_dict(self.topo_group_predictor_base[0].state_dict())
                        # TODO: maybe reset parameters or change some hyperparams?
                        self.topo_group_predictor_reset_scores = []
                        if self.topo_group_predictor_reset_num_iter > 0:
                            self.training_iter_cnt.fill_(0)
                    pgm = self._get_pgm_from_predictor(self.topo_group_predictor, pgm_shape)
                else:
                    pgm = self._get_pgm_from_predictor(self.topo_group_predictor, pgm_shape)
                    self.topo_group_predictor_cache = pgm
            else:
                pgm = self.topo_group_predictor_cache
            if self.topo_group_predictor_is_batched:
                assert len(pgm) != 1
                if self.training:
                    if len(self.topo_group_predictor_batch_scores) > self.topo_group_predictor_batch_score_window_size:
                        self.topo_group_predictor_batch_idx.data[:] = int(np.argmin(np.array(self.topo_group_predictor_batch_scores).sum(0)))
                        self.topo_group_predictor_batch_scores = []
                    # select random pgm to optimize
                    rand_idx = random.randint(0, len(pgm)-1)
                    pgm = pgm[rand_idx]#.unsqueeze(0)
                else:
                    # TODO: select the best pgm?
                    pgm = pgm[self.topo_group_predictor_batch_idx.item()]
                if self.topo_group_predictor_include_dynamic_kernel:
                    # TODO: proper unsqueeze
                    pgm = pgm.unsqueeze(0)
            if self.topo_group_predictor_include_dynamic_kernel:
                pgm = pgm.split((pgm.shape[1] - self.out_channels * self.in_channels * self.kernel_size * self.kernel_size - self.out_channels, self.out_channels * self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), dim=1)
            # avoid passing samples during coding
            topo_groups = self._preprocess_pgm(pgm, pgm_shape, fast_mode=fast_mode)
            if self.pgm_include_dynamic_kernel:
                topo_groups, dynamic_kernel_weight, dynamic_kernel_bias = topo_groups

            # if fast_mode:
            #     topo_groups = topo_groups.argmax(-1)
            if self.training and self.topo_group_predictor_enable_reset:
                if self.topo_group_predictor_reset_num_iter > 0:
                    self.topo_group_predictor_reset_activate = self.training_iter_cnt > self.topo_group_predictor_reset_num_iter
                else:
                    topo_groups_log_prob = torch.log_softmax(topo_groups, dim=-1)
                    topo_groups_entropy = (-topo_groups_log_prob * topo_groups_log_prob.exp()).mean()
                    self.update_cache("moniter_dict", 
                        pgm_entropy = topo_groups_entropy,
                    )
                    self.topo_group_predictor_reset_activate = topo_groups_entropy.item() < self.topo_group_predictor_reset_entropy_threshold
            
            if self.pgm_include_dynamic_kernel:
                topo_groups = topo_groups, dynamic_kernel_weight, dynamic_kernel_bias
        else:
            topo_groups = self._get_default_pgm(pgm_shape)
            if self.pgm_include_dynamic_kernel:
                if self.pgm_include_dynamic_kernel_full:
                    weight_params, bias_params = [self.context_prediction.weight.unsqueeze(0)], [self.context_prediction.bias.unsqueeze(0)]
                    for name, param in self.param_merger.named_parameters():
                        if ".weight" in name: weight_params.append(param.unsqueeze(0))
                        if ".bias" in name: bias_params.append(param.unsqueeze(0))
                    topo_groups = topo_groups, weight_params, bias_params
                else:
                    topo_groups = topo_groups, self.context_prediction.weight.unsqueeze(0), self.context_prediction.bias.unsqueeze(0)

        return topo_groups

    def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None, pgm : Any = None) -> torch.Tensor:
        batch_size = pgm_params.shape[0]
        if prior_params is None:
            prior_params = torch.zeros_like(pgm_params)
        # else:
        #     prior_params = prior_params.reshape(batch_size, self.out_channels, *pgm_params.shape[2:])
        # TODO: tmp solution for single channel prior_params! Should remove this!
        # if prior_params.shape[1] == pgm_params.shape[1] // 2:
        #     prior_params = prior_params.repeat(1, 2, 1, 1)
        if self.use_param_merger:
            # concat_params = torch.cat([pgm_params, prior_params], dim=1)
            if self.use_joint_ar_model_impl: #self.channel_groups == 1:
                concat_params = torch.cat([prior_params, pgm_params], dim=1)
                merged_params = self.entropy_parameters(concat_params)
            else:
                concat_params = torch.cat([pgm_params, prior_params], dim=1)
                if pgm is None:
                    pgm = torch.zeros(1, self.channel_groups, *pgm_params.shape[2:], dtype=torch.long)
                concat_pgms = torch.cat([pgm, torch.zeros_like(pgm) - 1], dim=1)# assign -1 topo group for prior
                for layer in self.param_merger:
                    if isinstance(layer, TopoGroupDynamicMaskConv2d):
                        layer.set_topo_groups(concat_pgms)
                merged_params = self.param_merger(concat_params)
                merged_params = merged_params.reshape(batch_size, self.channel_groups * 2, self.out_channels // self.channel_groups, *pgm_params.shape[2:])\
                    [:, :self.channel_groups].reshape(batch_size, self.out_channels, *pgm_params.shape[2:])
        else:
            # TODO: dist specific add?
            merged_params = pgm_params + prior_params if prior_params is not None else pgm_params
        # NOTE: gaussian impl chunks mean/scale by channel
        # so we swap dims to [batch_size, self.out_channels // self.channel_groups, self.channel_groups, spatial_size]
        # return merged_params.reshape(batch_size, self.channel_groups, self.out_channels // self.channel_groups, *pgm_params.shape[3:]).movedim(1,2)\
        #     .reshape(batch_size, self.out_channels, *pgm_params.shape[3:]).contiguous()
        return merged_params

    def _pgm_sample_from_topo_group_logits(self, topo_group_logits : torch.Tensor, input_shape : Optional[torch.Size] = None):
        # nan check
        if torch.isnan(topo_group_logits).any():
            return torch.zeros_like(topo_group_logits[...,0])
        
        topo_group_logits = torch.log_softmax(topo_group_logits, dim=-1) / self.relax_temp
        if self.training:
                
            if self.relax_temp_anneal:
                self.update_cache("moniter_dict", 
                    relax_temp=self.relax_temp
                )

        if self.training and not self.training_mc_sampling: # training_mc_sampling has its own logger
            topo_groups_log_prob = torch.log_softmax(topo_group_logits, dim=-1)
            topo_groups_entropy = (-topo_groups_log_prob * topo_groups_log_prob.exp()).mean()
            self.update_cache("moniter_dict", 
                pgm_entropy = topo_groups_entropy,
            )

        if self.topo_group_allow_continuous:
            cont_samples = torch.arange(0, topo_group_logits.shape[-1], dtype=topo_group_logits.dtype, device=topo_group_logits.device).unsqueeze(-1)
            if self.topo_group_continuous_sampling:
                if not self.training or self.relax_temp < self.relax_temp_threshold:
                    cont_samples = topo_group_logits.argmax(-1).type_as(cont_samples)
                else:
                    # annealing
                    if self.gs_temp_anneal:
                        self.update_cache("moniter_dict", 
                            gs_temp=self.gs_temp
                        )
                    if self.topo_group_continuous_sampling_method == "relaxed":
                        topo_group_dist = D.RelaxedOneHotCategorical(self.gs_temp, logits=topo_group_logits)
                        topo_group_samples = topo_group_dist.rsample().reshape(-1, topo_group_logits.shape[-1])
                        cont_samples = torch.matmul(topo_group_samples, cont_samples).reshape(*topo_group_logits.shape[:-1])
                    elif self.topo_group_continuous_sampling_method == "gmm":
                        mix = D.Categorical(logits=topo_group_logits)
                        means = torch.zeros_like(topo_group_logits)
                        means[..., :] = cont_samples.squeeze(-1)
                        comp = StableNormal(means, torch.zeros_like(means) + self.gs_temp)
                        topo_group_dist = ReparametrizedMixtureSameFamily(mix, comp)
                        cont_samples = topo_group_dist.rsample()
                    else:
                        raise NotImplementedError()
                return cont_samples
            else:
                topo_group_probs = torch.softmax(topo_group_logits, dim=-1).reshape(-1, topo_group_logits.shape[-1])
                cont_samples = torch.matmul(topo_group_probs, cont_samples).reshape(*topo_group_logits.shape[:-1])
                if not self.training:
                    return cont_samples.round()
                else:
                    if self.topo_group_continuous_add_noise:
                        return cont_samples + torch.zeros_like(cont_samples).uniform_(-0.5, 0.5)
                    else:
                        return cont_samples

        if self.training and self.training_mc_sampling:
            if self.training_mc_sampling_share_sample_batch:
                # remove batch and replace with multisamples
                num_samples = self.training_mc_num_samples # // topo_group_logits.shape[0]
                topo_group_dist = D.Categorical(logits=topo_group_logits)
                if self.training_mc_sampling_add_default_as_sample and random.random() < self.training_mc_sampling_add_default_as_sample_prob:
                    topo_group_indices = topo_group_dist.sample([num_samples-1]).long()
                    topo_group_indices = torch.cat([
                        topo_group_indices,
                        self._get_default_pgm(topo_group_logits.shape[:-1]).unsqueeze(0)
                    ], dim=0)
                else:
                    topo_group_indices = topo_group_dist.sample([num_samples]).long()
                topo_group_indices = topo_group_indices.reshape(self.training_mc_num_samples*topo_group_indices.shape[1], *topo_group_indices.shape[2:])
            else:
                if topo_group_logits.shape[0] == 1:
                    topo_group_logits = topo_group_logits.repeat(input_shape[0], *([1]*(topo_group_logits.ndim-1)))
                assert topo_group_logits.shape[0] == input_shape[0]
                topo_group_dist = D.Categorical(logits=topo_group_logits)
                topo_group_indices = topo_group_dist.sample().long()
        else:
            topo_group_indices = topo_group_logits.argmax(-1)

        return topo_group_indices
    
    def _pgm_logits_to_indices(self, pgm : torch.Tensor, input_shape : Optional[torch.Size] = None, force_argmax=False):
        if torch.is_floating_point(pgm):
            if force_argmax or input_shape is None:
                return pgm.argmax(-1)

            # TODO: check if pgm is logits
            topo_group_logits = pgm * self.topo_group_logits_modifier
            # sample pgm
            topo_group_indices = self._pgm_sample_from_topo_group_logits(topo_group_logits, input_shape=input_shape)

            # logging
            if self.training:
                if self.training_mc_sampling:
                    # NOTE: for Flow-based predictor, we can get pgm_log_prob during _get_pgm
                    if self.get_raw_cache().get("pgm_log_prob") is None:
                        topo_group_indices_logprob = torch.log_softmax(topo_group_logits, dim=-1)
                        # NOTE: one_hot seems faster for lower logits but may take more memory
                        topo_group_indices_logprob = (topo_group_indices_logprob * F.one_hot(topo_group_indices, topo_group_logits.shape[-1])).sum(-1)
                        # topo_group_indices_logprob = topo_group_indices_logprob.reshape(-1, topo_group_logits.shape[-1])\
                        #     [torch.arange(0, topo_group_indices.numel()).type_as(topo_group_indices), topo_group_indices.reshape(-1)].reshape_as(topo_group_indices)
                        # align batch size
                        if self.training_mc_sampling_share_sample_batch:
                            assert topo_group_indices.shape[0] == self.training_mc_num_samples 
                            topo_group_indices = topo_group_indices.repeat(input_shape[0] // topo_group_indices.shape[0], 1, 1, 1)
                            topo_group_indices_logprob = topo_group_indices_logprob.repeat(input_shape[0] // self.training_mc_num_samples, 1, 1, 1)
                        self.update_cache(pgm_log_prob=topo_group_indices_logprob)
            else:
                # merge channels to width dimension (TODO: maybe add a padding)
                topo_group_indices_thumbnail = topo_group_indices.movedim(1, -2)\
                    .reshape(topo_group_indices.shape[0], 1, topo_group_indices.shape[-2], -1)\
                    .float() / (topo_group_logits.shape[-1]-1)
                # convert to gnuplot colors
                # topo_group_indices_thumbnail = torch.cat(
                #     [topo_group_indices_thumbnail.sqrt(), topo_group_indices_thumbnail ** 3, topo_group_indices_thumbnail.sin() * 2 * np.pi ],
                #     dim=1
                # )
                self.update_cache("image_dict", topo_group_indices=topo_group_indices_thumbnail)

            return topo_group_indices
        else:
            return pgm

    def _topo_group_to_masks(self, data : torch.Tensor, pgm : torch.LongTensor) -> torch.BoolTensor:
        if self.pgm_include_dynamic_kernel:
            pgm, dynamic_kernel_weight, dynamic_kernel_bias = pgm

        assert data.shape[2:] == pgm.shape[2:]
        # NOTE: Currently we use post-repeat as we want dist parameters in the same group of channels
        # TODO: Maybe we should explicitly define if we want pre-repeat or post-repeat on pgm channel dimensions
        pgm_reshape = pgm.unsqueeze(2).repeat(data.shape[0] // pgm.shape[0], 1, data.shape[1] // pgm.shape[1], 1, 1).reshape_as(data)
        return [(pgm_reshape==i) for i in range(pgm.max()+1)]

    def _pgm_inference(self, input : torch.Tensor, pgm : Any, prior : Optional[torch.Tensor] = None, **kwargs):
        if self.training and (self.training_pgm_logits_use_random_num_iter or self.topo_group_predictor_reset_num_iter):
            self.training_iter_cnt += input.shape[0]
        # NOTE: perform reset during validation
        if not self.training and self.topo_group_predictor_enable_reset and self.topo_group_predictor_reset_activate:
            pgm_scores = []
            # pgm_shape = (input_shape[0], self.channel_groups, *input.shape[2:])
            # use single batch pgm for faster computation
            pgm_shape = (1, self.channel_groups, *input.shape[2:])
            pgm_current = self._get_pgm_from_predictor(self.topo_group_predictor, pgm_shape)
            if self.topo_group_predictor_include_dynamic_kernel:
                pgm_current = pgm_current.split((pgm_current.shape[1] - self.out_channels * self.in_channels * self.kernel_size * self.kernel_size - self.out_channels, self.out_channels * self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), dim=1)
            pgm_current = self._preprocess_pgm(pgm_current, pgm_shape) 
            params_current = super()._pgm_inference(input, pgm_current, prior, **kwargs)
            _, entropy_current = self._get_likelihood_and_entropy(input, params_current, pgm=pgm_current)
            pgm_scores.append(entropy_current.mean().item())
            pgm_best = self._get_pgm_from_predictor(self.topo_group_predictor_best[0].to(device=input.device), pgm_shape)
            if self.topo_group_predictor_include_dynamic_kernel:
                pgm_best = pgm_best.split((pgm_best.shape[1] - self.out_channels * self.in_channels * self.kernel_size * self.kernel_size - self.out_channels, self.out_channels * self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), dim=1)
            pgm_best = self._preprocess_pgm(pgm_best, pgm_shape) 
            params_best = super()._pgm_inference(input, pgm_best, prior, **kwargs)
            _, entropy_best = self._get_likelihood_and_entropy(input, params_best, pgm=pgm_best)
            pgm_scores.append(entropy_best.mean().item())
            if self.default_topo_group_method != "none" and not self.topo_group_predictor_include_dynamic_kernel:
                pgm_default = self._get_default_pgm(pgm_shape)
                params_default = super()._pgm_inference(input, pgm_default, prior, **kwargs)
                _, entropy_default = self._get_likelihood_and_entropy(input, params_default, pgm=pgm_default)
                pgm_scores.append(entropy_default.mean().item())
            self.topo_group_predictor_reset_scores.append(pgm_scores)
            # use best model as output
            params = params_best
        elif not self.training and self.topo_group_predictor_is_batched:
            pgm_scores = []
            # pgm_shape = (input_shape[0], self.channel_groups, *input.shape[2:])
            # use single batch pgm for faster computation
            pgm_shape = (1, self.channel_groups, *input.shape[2:])
            pgm_all = self._get_pgm_from_predictor(self.topo_group_predictor, pgm_shape)
            for idx in range(len(pgm_all)):
                pgm_current = pgm_all[idx]#.unsqueeze(0)
                if self.topo_group_predictor_include_dynamic_kernel:
                    # TODO: proper unsqueeze
                    pgm_current = pgm_current.unsqueeze(0)
                    pgm_current = pgm_current.split((pgm_current.shape[1] - self.out_channels * self.in_channels * self.kernel_size * self.kernel_size - self.out_channels, self.out_channels * self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), dim=1)
                pgm_current = self._preprocess_pgm(pgm_current, pgm_shape) 
                params_current = super()._pgm_inference(input, pgm_current, prior, **kwargs)
                _, entropy_current = self._get_likelihood_and_entropy(input, params_current, pgm=pgm_current)
                pgm_scores.append(entropy_current.mean().item())
                # logging
                self.update_cache("metric_dict", **{f"prior_entropy_mean_pgm_{idx}" : entropy_current.mean()})
                # use topo_group_predictor_batch_idx model as output
                if idx == self.topo_group_predictor_batch_idx:
                    params = params_current
            self.topo_group_predictor_batch_scores.append(pgm_scores)
        else:
            params = super()._pgm_inference(input, pgm, prior, **kwargs)
        return params

    def _pgm_inference_full(self, input : torch.Tensor, pgm : Any, prior : Optional[torch.Tensor] = None, **kwargs):
        
        if self.pgm_include_dynamic_kernel:
            pgm, dynamic_kernel_weight, dynamic_kernel_bias = pgm

        pgm = self._pgm_logits_to_indices(pgm, input_shape=input.shape)

        if not self.training:
            # make grid
            pgm_thumbnail = pgm.float() / pgm.max()
            pgm_thumbnail = make_grid(pgm_thumbnail.reshape(pgm.shape[0]*pgm.shape[1], 1, *pgm.shape[-2:]), nrow=2, padding=2)
            self.update_cache("image_dict", pgm_thumbnail=pgm_thumbnail)


        # NOTE: Moved to TopoGroupDynamicMaskConv2d
        # batch_size = input.shape[0]
        # input_unfold_group = F.unfold(input, (self.kernel_size, self.kernel_size), padding=self.padding).unsqueeze(1)
        # # Note: as padding is zero, we make padded values the largest topogroup so that they are excluded from maskconv
        # pgm_offset = pgm.type_as(input) - pgm.max() - 1
        # pgm_2d_unfold_group = F.unfold(pgm_offset, (self.kernel_size, self.kernel_size), padding=self.padding).unsqueeze(1)
        # pgm_center_group = pgm_offset.reshape(pgm.shape[0], pgm.shape[1], 1, -1)
        #     # .repeat(1, 1, self.channel_groups * self.kernel_size * self.kernel_size, 1).contiguous()
        # # [batch_size, self.channel_groups, self.in_channels // self.channel_groups * self.kernel_size * self.kernel_size, spatial_size]
        # pgm_2d_unfold_mask_group = (pgm_2d_unfold_group < pgm_center_group)\
        #     .unsqueeze(2).repeat(1, 1, self.in_channels // self.channel_groups, 1, 1).reshape(batch_size, self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
        # # pgm_2d_unfold_mask_group = pgm_2d_unfold_mask_group.repeat(1, 1, input.shape[1] // self.channel_groups, 1).reshape_as(input_unfold_group).contiguous()
        # # input_unfold_group_masked = (input_unfold_group * pgm_2d_unfold_mask_group)\
        # #     .reshape(batch_size * self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
        # input_conv_masked = input_unfold_group * pgm_2d_unfold_mask_group
        # conv_kernel_weight = self.conv_kernel_weight
        # conv_kernel_bias = self.conv_kernel_bias
        # # [batch_size, self.channel_groups, self.out_channels // self.channel_groups, spatial_size]
        # pgm_params_unfold_group = conv_kernel_weight.reshape(1, self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size)\
        #     .matmul(input_conv_masked) + conv_kernel_bias.reshape(1, self.channel_groups, self.out_channels // self.channel_groups, 1)
        # pgm_params = pgm_params_unfold_group.reshape(batch_size, self.channel_groups, self.out_channels // self.channel_groups, *input.shape[2:])
        
        if self.topo_group_context_model is not None:
            merged_params = self.topo_group_context_model(input, pgm, prior=prior)
        else:
            if self.use_dynamic_kernel_predictor:
                dynamic_kernel = self.dynamic_kernel_predictor(pgm.type_as(input))
                dynamic_kernel_weight, dynamic_kernel_bias = dynamic_kernel.split((self.out_channels * self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), dim=1)
                dynamic_kernel_weight = dynamic_kernel_weight.reshape(pgm.shape[0], self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
                dynamic_kernel_bias = dynamic_kernel_bias.reshape(pgm.shape[0], self.channel_groups, self.out_channels // self.channel_groups, -1)
                pgm_params = self.context_prediction(input, pgm, dynamic_kernel_weight=dynamic_kernel_weight, dynamic_kernel_bias=dynamic_kernel_bias)
            elif self.pgm_include_dynamic_kernel:
                if self.pgm_include_dynamic_kernel_full:
                    for idx, (weight, bias) in enumerate(zip(dynamic_kernel_weight, dynamic_kernel_bias)):
                        # weight = weight.reshape(weight.shape[0], self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
                        # bias = bias.reshape(bias.shape[0], self.channel_groups, self.out_channels // self.channel_groups, -1)
                        if idx == 0:
                            self.context_prediction.set_dynamic_kernel(weight, bias)
                        else:
                            # TODO: proper set for param_merger
                            self.param_merger[2*(idx-1)].set_dynamic_kernel(weight, bias)
                    pgm_params = self.context_prediction(input, pgm)
                else:
                    dynamic_kernel_weight = dynamic_kernel_weight.reshape(dynamic_kernel_weight.shape[0], self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
                    dynamic_kernel_bias = dynamic_kernel_bias.reshape(dynamic_kernel_bias.shape[0], self.channel_groups, self.out_channels // self.channel_groups, -1)
                    pgm_params = self.context_prediction(input, pgm, dynamic_kernel_weight=dynamic_kernel_weight, dynamic_kernel_bias=dynamic_kernel_bias)
            else:
                pgm_params = self.context_prediction(input, pgm)

            merged_params = self._merge_prior_params(pgm_params, prior_params=prior, pgm=pgm)
        # if prior is None:
        #     prior_params = torch.zeros_like(pgm_params)
        # else:
        #     prior_params = prior.reshape(batch_size, self.out_channels, *input.shape[2:])
        # if self.use_param_merger:
        #     concat_params = torch.cat([pgm_params, prior_params], dim=1)
        #     concat_pgms = torch.cat([pgm, torch.zeros_like(pgm) - 1], dim=1) # assign -1 topo group for prior
        #     for layer in self.param_merger:
        #         if isinstance(layer, TopoGroupDynamicMaskConv2d):
        #             layer.set_topo_groups(concat_pgms)
        #     merged_params = self.param_merger(concat_params)\
        #         .reshape(batch_size, self.channel_groups * 2, self.out_channels // self.channel_groups, *input.shape[2:])\
        #         [:, :pgm.shape[1]].reshape(batch_size, self.out_channels, *input.shape[2:])
        # else:
        #     # TODO: dist specific add?
        #     merged_params = pgm_params + prior if prior is not None else pgm_params
        # NOTE: gaussian impl chunks mean/scale by channel
        # so we swap dims to [batch_size, self.out_channels // self.channel_groups, self.channel_groups, spatial_size]
        # return merged_params.movedim(1,2).reshape(batch_size, self.out_channels, *input.shape[2:]).contiguous()        
        return merged_params

    def _pgm_inference_group_mask(self, input : torch.Tensor, topo_group_mask : torch.BoolTensor, pgm : Optional[Any] = None, prior : Optional[torch.Tensor] = None, **kwargs):
        if self.pgm_include_dynamic_kernel:
            pgm, dynamic_kernel_weight, dynamic_kernel_bias = pgm

        # NOTE: Use GaussianChannelAutoregressiveMaskConv2DTopoGroupPGMPriorCoder for faster practical coding!
        # Because of channel-specific kernels, masked input is not allowed and we use full convolution on input 
        # batch_size = input.shape[0]
        # input_unfold_group = F.unfold(input, (self.kernel_size, self.kernel_size), padding=self.padding).unsqueeze(1)
        # input_conv_masked = input_unfold_group
        # conv_kernel_weight = self.context_prediction.weight # self.conv_kernel_weight
        # conv_kernel_bias = self.context_prediction.bias # self.conv_kernel_bias
        # # [batch_size, self.channel_groups, self.out_channels // self.channel_groups, spatial_size]
        # pgm_params_unfold_group = conv_kernel_weight.reshape(1, self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size)\
        #     .matmul(input_conv_masked)
        # if conv_kernel_bias is not None:
        #     pgm_params_unfold_group = pgm_params_unfold_group + conv_kernel_bias.reshape(1, self.channel_groups, self.out_channels // self.channel_groups, 1)
        # # pgm_params = pgm_params_unfold_group.reshape(batch_size, self.channel_groups, self.out_channels // self.channel_groups, *input.shape[2:])
        # pgm_params = pgm_params_unfold_group.reshape(batch_size, self.out_channels, *input.shape[2:])
        
        if self.topo_group_context_model is not None:
            merged_params = self.topo_group_context_model(input, pgm, prior=prior)
        else:
            with self.profiler.start_time_profile("context_prediction"):
                # TODO: dynamic_kernel_predictor?
                if self.use_dynamic_kernel_predictor: 
                    raise NotImplementedError("use_dynamic_kernel_predictor not implemented for coding")
                elif self.pgm_include_dynamic_kernel:
                    if self.pgm_include_dynamic_kernel_full:
                        for idx, (weight, bias) in enumerate(zip(dynamic_kernel_weight, dynamic_kernel_bias)):
                            # weight = weight.reshape(weight.shape[0], self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
                            # bias = bias.reshape(bias.shape[0], self.channel_groups, self.out_channels // self.channel_groups, -1)
                            if idx == 0:
                                self.context_prediction.set_dynamic_kernel(weight, bias)
                            else:
                                # TODO: proper set for param_merger
                                self.param_merger[2*(idx-1)].set_dynamic_kernel(weight, bias)
                        pgm_params = self.context_prediction(input, pgm)
                    else:
                        dynamic_kernel_weight = dynamic_kernel_weight.reshape(dynamic_kernel_weight.shape[0], self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
                        dynamic_kernel_bias = dynamic_kernel_bias.reshape(dynamic_kernel_bias.shape[0], self.channel_groups, self.out_channels // self.channel_groups, -1)
                        pgm_params = self.context_prediction(input, pgm, dynamic_kernel_weight=dynamic_kernel_weight, dynamic_kernel_bias=dynamic_kernel_bias)
                else:
                    pgm_params = self.context_prediction(input, pgm)
            with self.profiler.start_time_profile("merge_prior_params"):
                merged_params = self._merge_prior_params(pgm_params, prior_params=prior, pgm=pgm)
        return merged_params
        # if prior is None:
        #     prior_params = torch.zeros_like(pgm_params)
        # else:
        #     prior_params = prior.reshape(batch_size, self.channel_groups, self.out_channels // self.channel_groups, *input.shape[2:])
        # if self.use_param_merger:
        #     concat_params = torch.cat([pgm_params, prior_params], dim=2).reshape(batch_size, self.out_channels * 2, *input.shape[2:])
        #     merged_params = self.param_merger(concat_params).reshape(batch_size, self.channel_groups, self.out_channels // self.channel_groups, *input.shape[2:])
        # else:
        #     # TODO: dist specific add?
        #     merged_params = pgm_params + prior if prior is not None else pgm_params
        # # NOTE: gaussian impl chunks mean/scale by channel
        # # so we swap dims to [batch_size, self.out_channels // self.channel_groups, self.channel_groups, spatial_size]
        # return merged_params.movedim(1,2).reshape(batch_size, self.out_channels, *input.shape[2:]).contiguous()

    def _encode_with_pgm(self, input : torch.Tensor, *args, prior : torch.Tensor = None, pgm : Optional[Any] = None, quantizer_params: Optional[Any] = None, **kwargs) -> bytes:
        # TODO: implement kernel-wise non-parallel coding in super class
        # TODO: implement pgm_input_dequantized case
        if self.use_joint_ar_model_impl:
            assert self.channel_groups == 1
            input_no_quant = self._data_preprocess(input, quantize=False, quantizer_params=quantizer_params)
            # data_buffer = torch.zeros(input.shape[0], self.in_channels, *input.shape[2:]).to(device=self.device)
            # pgm = self._get_pgm(input, prior=prior, input_shape=input.shape, pgm=pgm, fast_mode=True)
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            height, width = input.shape[2:]
            kernel_size = self.kernel_size
            padding = self.padding
            y_hat = F.pad(input_no_quant, (padding[0], padding[0], padding[1], padding[1]))
            
            mask = torch.ones_like(self.context_prediction.weight.data)
            mask[:, :, kernel_size // 2, kernel_size // 2 :] = 0
            mask[:, :, kernel_size // 2 + 1 :] = 0
            masked_weight = self.context_prediction.weight * mask
            data_list, indexes_list = [], []
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
                    p = prior[:, :, h : h + 1, w : w + 1]
                    params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                    params = params.squeeze(-1).squeeze(-1)
                    scales_hat, means_hat = params.chunk(2, 1)

                    y_crop = y_crop[:, :, padding[0], padding[1]]
                    indexes = self._build_coding_indexes(params)
                    data_offsets = self._build_coding_offsets(params)
                    data_coding = self._data_preprocess(y_crop - data_offsets, transform=False, quantize=True, quantizer_params=quantizer_params)
                    # self.ans_encoder.encode_with_indexes(
                    #     data_coding.detach().cpu().numpy().astype(np.int32),
                    #     indexes.detach().cpu().numpy().astype(np.int32),
                    #     cache=True,
                    # )
                    y_hat[:, :, h + padding[0], w + padding[1]] = data_coding + data_offsets
                    data_list.append(data_coding)
                    indexes_list.append(indexes)
            # return self.ans_encoder.flush()
            data = torch.cat(data_list).contiguous().detach().cpu().numpy().astype(np.int32)
            indexes = torch.cat(indexes_list).contiguous().detach().cpu().numpy().astype(np.int32)
            return self.ans_encoder.encode_with_indexes(data, indexes)

        else:
            return super()._encode_with_pgm(input, *args, prior=prior, pgm=pgm, quantizer_params=quantizer_params, **kwargs)

    def _pgm_generate(self, byte_string : bytes, pgm : Any, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, **kwargs):
        # TODO: implement kernel-wise non-parallel coding in super class
        if self.use_joint_ar_model_impl:
            assert self.channel_groups == 1
            self.ans_decoder.set_stream(byte_string)
            # data_buffer = torch.zeros(input_shape[0], self.in_channels, *input_shape[2:]m device=self.device)

            height, width = input_shape[2:]
            kernel_size = self.kernel_size
            padding = self.padding
            # Warning: this is slow due to the auto-regressive nature of the
            # decoding... See more recent publication where they use an
            # auto-regressive module on chunks of channels for faster decoding...
            y_hat = torch.zeros(input_shape[0], self.in_channels, height + 2 * padding[0], width + 2 * padding[1], device=self.device)
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
                    p = prior[:, :, h : h + 1, w : w + 1]
                    params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))

                    indexes, data_offsets = self._build_coding_params(params)
                    symbols = self.ans_decoder.decode_stream(indexes.detach().cpu().numpy().astype(np.int32))

                    rv = torch.Tensor(symbols).type_as(y_hat) + data_offsets
                    hp = h + padding[0]
                    wp = w + padding[1]
                    y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

            return F.pad(y_hat, (-padding[0], -padding[0], -padding[1], -padding[1]))
        else:
            return super()._pgm_generate(byte_string, pgm, prior, input_shape, **kwargs)

    # def _standardized_cumulative(self, inputs):
    #     half = float(0.5)
    #     const = float(-(2**-0.5))
    #     # Using the complementary error function maximizes numerical precision.
    #     return half * torch.erfc(const * inputs)

    # @staticmethod
    # def _standardized_quantile(quantile):
    #     import scipy.stats
    #     return scipy.stats.norm.ppf(quantile)

    # def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
    #     from compressai._CXX import pmf_to_quantized_cdf
    #     cdf = torch.zeros(
    #         (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
    #     )
    #     for i, p in enumerate(pmf):
    #         prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
    #         _cdf = pmf_to_quantized_cdf(prob.tolist(), 16)
    #         _cdf = torch.IntTensor(_cdf)
    #         cdf[i, : _cdf.size(0)] = _cdf
    #     return cdf

    # def update_state(self, *args, **kwargs) -> None:
    #     from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder
    #     encoder = Rans64Encoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding, bypass_precision=self.bypass_precision)
    #     decoder = Rans64Decoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding, bypass_precision=self.bypass_precision)

    #     multiplier = -self._standardized_quantile(1e-9 / 2)
    #     pmf_center = torch.ceil(self.scale_table * multiplier).int()
    #     pmf_length = 2 * pmf_center + 1
    #     max_length = torch.max(pmf_length).item()

    #     device = pmf_center.device
    #     samples = torch.abs(
    #         torch.arange(max_length, device=device).int() - pmf_center[:, None]
    #     )
    #     samples_scale = self.scale_table.unsqueeze(1)
    #     samples = samples.float()
    #     samples_scale = samples_scale.float()
    #     upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
    #     lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
    #     pmf = upper - lower

    #     tail_mass = 2 * lower[:, :1]

    #     quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
    #     quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
    #     self._quantized_cdf = quantized_cdf
    #     self._offset = -pmf_center
    #     self._cdf_length = pmf_length + 2

    #     cdf_sizes = pmf_length.cpu().numpy() + 2
    #     offsets_np = -pmf_center.cpu().numpy()
    #     cdfs_np = quantized_cdf.cpu().numpy()
    #     encoder.init_cdf_params(cdfs_np, cdf_sizes, offsets_np)
    #     decoder.init_cdf_params(cdfs_np, cdf_sizes, offsets_np)

    #     # freqs, nfreqs, offsets = self._get_ans_params()
    #     # encoder.init_params(freqs, nfreqs, offsets)
    #     # decoder.init_params(freqs, nfreqs, offsets)

    #     self.ans_encoder = encoder
    #     self.ans_decoder = decoder


# TODO:
# class GaussianChannelAutoregressiveMaskConv2DTopoGroupPGMPriorCoder(GaussianPGMPriorCoderImpl, TopoGroupPGMPriorCoder):
#     def __init__(self, *args, 
#                  in_channels=256, 
#                  channel_groups : int = 1, # TODO: spatial groups?
#                  in_channel_splits : Optional[List[int]] = None,
#                  default_topo_group_method="none",
#                  kernel_size=5, 
#                  use_param_merger=True, 
#                  **kwargs):
#         TopoGroupPGMPriorCoder.__init__(self, *args, in_channels=in_channels, **kwargs)
#         GaussianPGMPriorCoderImpl.__init__(self, *args, **kwargs)
#         if in_channel_splits is not None:
#             self.in_channel_splits = in_channel_splits
#             self.channel_groups = len(in_channel_splits)
#         else:
#             self.in_channel_splits = [in_channels // channel_groups] * channel_groups
#             self.channel_groups = channel_groups
#         self.out_channel_splits = [c * self.num_dist_params for c in in_channel_splits]
#         self.default_topo_group_method = default_topo_group_method
#         self.kernel_size = kernel_size
#         self.padding = (kernel_size // 2, kernel_size // 2)
#         self.use_param_merger = use_param_merger

#         if self.default_topo_group_method == "none":
#             pass
#         elif self.default_topo_group_method == "checkerboard":
#             self.channel_groups = 1
#             self.default_num_topo_groups = 2
#         elif self.default_topo_group_method == "half-checkerboard":
#             self.channel_groups = 1
#             self.default_num_topo_groups = 2
#         elif self.default_topo_group_method == "raster2x2":
#             self.channel_groups = 1
#             self.default_num_topo_groups = 4
#         else:
#             raise NotImplementedError(f"Unknown default_topo_group_method {self.default_topo_group_method}")

#         self.out_channels = self.in_channels * self.num_dist_params
#         self.conv_kernel_weight = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, kernel_size, kernel_size))
#         self.conv_kernel_bias = nn.Parameter(torch.zeros(self.out_channels))

#         if self.use_param_merger:
#             param_merger_list = []
#             for out_channel_split in self.out_channel_splits:
#                 param_merger_out_channels = out_channel_split
#                 param_merger = nn.Sequential(
#                     nn.Linear(param_merger_out_channels * 2, param_merger_out_channels * 5 // 3),
#                     nn.LeakyReLU(inplace=True),
#                     nn.Linear(param_merger_out_channels * 5 // 3, param_merger_out_channels * 4 // 3),
#                     nn.LeakyReLU(inplace=True),
#                     nn.Linear(param_merger_out_channels * 4 // 3, param_merger_out_channels),
#                 )
#                 param_merger_list.append(param_merger)
#             self.param_merger_list = nn.ModuleList(param_merger_list)

#     # TODO: cache pgm for faster computation
#     def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, **kwargs) -> torch.LongTensor:
#         # TODO: get shape from bytes?
#         pgm_shape = (input_shape[0], *input_shape[-2:])
#         topo_groups_spatial = torch.zeros(pgm_shape[0], *pgm_shape[-2:], dtype=torch.long, device=self.device)
#         if self.default_topo_group_method == "none":
#             pass
#         elif self.default_topo_group_method == "checkerboard":
#             topo_groups_spatial[..., 0::2, 1::2] = 1
#             topo_groups_spatial[..., 1::2, 0::2] = 1
#         elif self.default_topo_group_method == "half-checkerboard":
#             topo_groups_spatial.fill_(1)
#             topo_groups_spatial[..., 1::2, 1::2] = 0
#         elif self.default_topo_group_method == "raster2x2":
#             topo_groups_spatial[..., 0::2, 1::2] = 1
#             topo_groups_spatial[..., 1::2, 0::2] = 2
#             topo_groups_spatial[..., 1::2, 1::2] = 3
#         topo_groups_channel = torch.zeros(pgm_shape[0], self.channel_groups, dtype=torch.long, device=self.device)
#         return topo_groups_spatial, topo_groups_channel

#     def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None) -> torch.Tensor:
#         if prior_params is None:
#             prior_params = torch.zeros_like(pgm_params)
#         assert pgm_params.shape[1] == prior_params.shape[1]
#         concat_params = torch.cat([pgm_params, prior_params], dim=1).movedim(1, -1).contiguous()
#         # .movedim(-1, 2).reshape(pgm_params.shape[0], -1, *pgm_params.shape[2:-1])
#         return self.param_merger(concat_params).movedim(-1, 1).contiguous()
    
#     def _pgm_inference_full(self, input : torch.Tensor, pgm : Any, prior : Optional[torch.Tensor] = None, **kwargs):
#         batch_size = input.shape[0]
#         input_unfold_group = F.unfold(input, (self.kernel_size, self.kernel_size), padding=self.padding)
#         # Note: as padding is zero, we make padded values the largest topogroup so that they are excluded from maskconv
#         pgm_offset = pgm.type_as(input) - pgm.max() - 1
#         pgm_2d_unfold_group = F.unfold(pgm_offset, (self.kernel_size, self.kernel_size), padding=self.padding)
#         pgm_center_group = pgm_offset.reshape(pgm.shape[0], pgm.shape[1], 1, -1)
#         # [batch_size, self.in_channels * self.kernel_size * self.kernel_size, spatial_size]
#         pgm_2d_unfold_mask_group = (pgm_2d_unfold_group < pgm_center_group)
#         # pgm_2d_unfold_mask_group = pgm_2d_unfold_mask_group.repeat(1, 1, input.shape[1] // self.channel_groups, 1).reshape_as(input_unfold_group).contiguous()
#         # input_unfold_group_masked = (input_unfold_group * pgm_2d_unfold_mask_group)\
#         #     .reshape(batch_size * self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size, -1)
#         input_conv_masked = input_unfold_group * pgm_2d_unfold_mask_group
#         conv_kernel_weight = self.conv_kernel_weight
#         conv_kernel_bias = self.conv_kernel_bias
#         # [batch_size, self.channel_groups, self.out_channels // self.channel_groups, spatial_size]
#         pgm_params_unfold_group = conv_kernel_weight.reshape(1, self.channel_groups, self.out_channels // self.channel_groups, self.in_channels * self.kernel_size * self.kernel_size)\
#             .matmul(input_conv_masked) + conv_kernel_bias.unsqueeze(-1).unsqueeze(0)
#         # NOTE: gaussian impl chunks mean/scale by channel
#         # so we swap dims to [batch_size, self.out_channels // self.channel_groups, self.channel_groups, spatial_size]
#         pgm_params = pgm_params_unfold_group.movedim(1,2).contiguous()
#         # pgm_params = self.context_prediction(input)
#         if prior is None:
#             prior_params = torch.zeros_like(pgm_params)
#         else:
#             prior_params = prior.reshape(batch_size, self.channel_groups, self.out_channels // self.channel_groups, *input.shape[2:]).movedim(1,2).contiguous()
#         if self.use_param_merger:
#             merged_params = self._merge_prior_params(pgm_params, prior_params=prior_params)
#         else:
#             # TODO: dist specific add?
#             merged_params = pgm_params + prior if prior is not None else pgm_params
#         return merged_params.view(batch_size, self.out_channels, *input.shape[2:])

#     def _pgm_inference_group_mask(self, input : torch.Tensor, topo_group_mask : torch.BoolTensor, prior : Optional[torch.Tensor] = None, **kwargs):
#         # Use ChannelAutoregressive for practical coding!
#         raise NotImplementedError("")

# NOTE: deprecated
class GaussianMaskConv2DPriorCoder(GaussianPGMPriorCoderImpl, NNTrainablePGMPriorCoder):
    def __init__(self, *args, 
                 in_channels=256, 
                 kernel_size=5, 
                 use_param_merger=True, 
                 use_em_loss=False, 
                 em_loss_beta=1.0, 
                 em_loss_beta_variable=False, 
                 **kwargs):
        NNTrainablePGMPriorCoder.__init__(self, *args, **kwargs)
        GaussianPGMPriorCoderImpl.__init__(self, *args, **kwargs)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.use_param_merger = use_param_merger
        self.use_em_loss = use_em_loss
        self.em_loss_beta = em_loss_beta
        self.em_loss_beta_variable = em_loss_beta_variable
        if self.em_loss_beta_variable:
            self.em_loss_beta = nn.Parameter(torch.tensor(em_loss_beta), requires_grad=False)

        self.maskconv_kernel = nn.Parameter(torch.zeros(self.in_channels * self.num_dist_params, self.in_channels, kernel_size, kernel_size))
        
        # self.context_prediction = MaskedConv2d(
        #     self.in_channels, self.in_channels * self.num_dist_params, kernel_size=5, padding=2, stride=1
        # )

        self.param_merger = nn.Sequential(
            nn.Conv2d(self.in_channels * self.num_dist_params * 2, self.in_channels * self.num_dist_params * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.num_dist_params * 5 // 3, self.in_channels * self.num_dist_params * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.num_dist_params * 4 // 3, self.in_channels * self.num_dist_params, 1),
        )

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        # TODO: get shape from bytes?
        pgm_shape = (input_shape[0], *input_shape[-2:])
        mask = torch.ones(*pgm_shape, self.kernel_size, self.kernel_size).type_as(self.maskconv_kernel)
        mask[..., self.kernel_size // 2, self.kernel_size // 2 :] = 0
        mask[..., self.kernel_size // 2 + 1 :, :] = 0
        return mask
    
    def _pgm_sort_topo(self, input : torch.Tensor, pgm: Any, **kwargs) -> Optional[torch.LongTensor]:
        pgm_topo_groups = self._pgm_topo_groups(input, pgm)
        input_mask = torch.zeros_like(input)
        start_idx = 0
        for group in pgm_topo_groups:
            group_length = len(input_mask[group])
            group_idxs = torch.arange(start_idx, group_length)
            input_mask[group] = group_idxs
            start_idx += group_length
        return input_mask

    # pytorch implementation of topolocial sorting
    def _pgm_topo_groups(self, input : torch.Tensor, pgm: Any, **kwargs) -> List[torch.LongTensor]:
        topo_group = []
        input_mask = torch.ones_like(input)
        pgm = pgm.reshape(*input.shape, -1)
        # indegrees = F.fold(pgm, *input.shape[2:], (self.kernel_size, self.kernel_size))
        # zero_indegree_mask = (indegrees == 0)
        # topo_group.append(input[zero_indegree_mask])
        # input_mask[zero_indegree_mask] -= 1
        while (input_mask != 0).any():
            input_mask_unfold = F.unfold(input_mask, (self.kernel_size, self.kernel_size), padding=self.padding)
            input_mask_unfold = input_mask_unfold.reshape(input.shape[0], input.shape[1], -1, *input.shape[2:]).movedim(2, -1)
            indegrees = (pgm * input_mask_unfold).sum(-1)
            zero_indegree_mask = (indegrees == 0)
            assert zero_indegree_mask.any(), "PGM invalid! May contain loops!"
            topo_group.append(zero_indegree_mask.nonzero(as_tuple=False))
            input_mask[zero_indegree_mask] -= 1

        return topo_group

    def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None) -> torch.Tensor:
        if prior_params is None:
            prior_params = torch.zeros_like(pgm_params)
        assert pgm_params.shape[1] == prior_params.shape[1]
        concat_params = torch.cat([pgm_params, prior_params], dim=1)
        # .movedim(-1, 2).reshape(pgm_params.shape[0], -1, *pgm_params.shape[2:-1])
        return self.param_merger(concat_params)
    
    def _pgm_inference(self, input : torch.Tensor, pgm : Any, prior : torch.Tensor = None, detach_kernel : bool = False):
        input_unfold = F.unfold(input, (self.kernel_size, self.kernel_size), padding=self.padding)
        pgm_2d_unfold = pgm.reshape(input.shape[0], *input.shape[2:], -1).movedim(-1, 1)
        pgm_2d_unfold = pgm_2d_unfold.reshape(pgm_2d_unfold.shape[0], pgm_2d_unfold.shape[1], -1)\
            .repeat(1, self.in_channels, 1).contiguous()
        maskconv_kernel = self.maskconv_kernel.detach() if detach_kernel else self.maskconv_kernel
        pgm_params_unfold = (input_unfold * pgm_2d_unfold).movedim(1, 2).matmul(maskconv_kernel.reshape(maskconv_kernel.shape[0], -1).t())
        pgm_params = pgm_params_unfold.movedim(1, 2).contiguous().view(pgm_params_unfold.shape[0], pgm_params_unfold.shape[2], *input.shape[2:])
        # pgm_params = self.context_prediction(input)
        merged_params = self._merge_prior_params(pgm_params, prior_params=prior)
        return merged_params

    def _pgm_generate(self, byte_string : bytes, pgm : Any, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, **kwargs):
        # TODO: if prior is None?
        self.ans_decoder.set_stream(byte_string)
        data_buffer = torch.zeros(input_shape[0], self.in_channels, *input_shape[2:]).type_as(prior)
        pgm_topo_groups = self._pgm_topo_groups(pgm)
        last_group = None
        # TODO: update grouping method
        for group in pgm_topo_groups:
            if last_group is None: last_group = torch.zeros_like(data_buffer[group])
            params = self._get_params_with_pgm(last_group, pgm[group], prior[group], pgm)
            indexes, data_offsets = self._build_coding_params(params)
            symbols = self.ans_decoder.decode_stream(indexes.detach().cpu().numpy().astype(np.int32))
            symbols = symbols + data_offsets.detach().cpu().numpy().astype(np.int32)
            # last_group = self._data_postprocess(symbols)
            # data_buffer[group] = last_group
            data_buffer[group] = torch.as_tensor(symbols).type_as(data_buffer)
        return data_buffer

    # def _get_params_with_pgm(self, input: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
    #     # pgm_2d = self._get_pgm_2d(input.movedim(1, -1))
    #     # pgm_2d_unfold = pgm_2d.reshape(input.shape[0], *input.shape[2:], -1).movedim(-1, 1)
    #     # pgm_2d_unfold = pgm_2d_unfold.reshape(pgm_2d_unfold.shape[0], pgm_2d_unfold.shape[1], -1).contiguous()
    #     input_unfold = F.unfold(input, (self.kernel_size, self.kernel_size), padding=self.padding)
    #     pgm_2d = self._get_pgm(prior)
    #     pgm_2d_unfold = pgm_2d.reshape(1, *input.shape[2:], -1).movedim(-1, 1)
    #     pgm_2d_unfold = pgm_2d_unfold.reshape(pgm_2d_unfold.shape[0], pgm_2d_unfold.shape[1], -1)\
    #         .repeat(1, self.in_channels, 1).contiguous()
    #     pgm_params_unfold = (input_unfold * pgm_2d_unfold).movedim(1, 2).matmul(self.maskconv_kernel.reshape(self.maskconv_kernel.shape[0], -1).t())
    #     pgm_params = pgm_params_unfold.movedim(1, 2).contiguous().view(pgm_params_unfold.shape[0], pgm_params_unfold.shape[2], *input.shape[2:])
    #     # mask = torch.ones_like(self.maskconv_kernel)
    #     # mask[..., self.kernel_size // 2, self.kernel_size // 2 :] = 0
    #     # mask[..., self.kernel_size // 2 + 1 :, :] = 0
    #     # pgm_params = F.conv2d(input, self.maskconv_kernel * mask, padding=self.padding)
    #     merged_params = self._merge_prior_params(pgm_params, prior_params=prior)
    #     return merged_params

    # def _decode_with_pgm(self, byte_string: bytes, prior: torch.Tensor = None) -> torch.Tensor:
    #     # TODO: if prior is None?
    #     data_buffer = torch.zeros(prior.shape[0], self.in_channels, *prior.shape[2:]).type_as(prior)
    #     pgm_2d = self._get_pgm(prior)
    #     pgm_topo_groups = self._pgm_topo_groups(pgm_2d)
    #     last_group = None
    #     for group in pgm_topo_groups:
    #         if last_group is None: last_group = torch.zeros_like(data_buffer[group])
    #         params = self._get_params_with_pgm(last_group, pgm_2d[group], prior[group])
    #         indexes, data_offsets = self._build_coding_params(params)
    #         symbols = self.ans_decoder.decode_with_indexes(byte_string, indexes)
    #         symbols = symbols + data_offsets
    #         last_group = self._data_postprocess(symbols)
    #         data_buffer[group] = last_group
    #     return data_buffer

    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        if self.training and self.use_em_loss:
            input_quant = self._data_preprocess(input, differentiable=self.training)
            pgm = self._get_pgm(input_quant, prior=prior, input_shape=input.shape)
            params = self._pgm_inference(input_quant, pgm.detach(), prior=prior)
            dist = self._params_to_dist(params)
            likelihood = dist.cdf(input_quant + 0.5) - dist.cdf(input_quant - 0.5)
            entropy = -torch.log(likelihood.clamp_min(self.eps)).sum() 
            self.update_cache("loss_dict", 
                loss_rate = entropy / math.log(2) / input.shape[0] # normalize by batch size
            )
            self.update_cache("metric_dict",
                prior_entropy = entropy / input.shape[0], # normalize by batch size
            )
            params = self._pgm_inference(input_quant, pgm, prior=prior, detach_kernel=True)
            dist = self._params_to_dist(params)
            likelihood = dist.cdf(input_quant + 0.5) - dist.cdf(input_quant - 0.5)
            entropy = -torch.log(likelihood.clamp_min(self.eps)).sum()
            self.update_cache("loss_dict", 
                loss_rate_em_beta = entropy * self.em_loss_beta / math.log(2) / input.shape[0] # normalize by batch size
            )
            if self.em_loss_beta_variable:
                if self.training:
                    self.update_cache("moniter_dict", 
                        em_loss_beta=self.em_loss_beta
                    )
            return input_quant
        else:
            return super().forward(input, prior, **kwargs)


class NoContextMaskConv2DGaussianPriorCoder(GaussianMaskConv2DPriorCoder):
    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        # TODO: get shape from bytes?
        pgm_shape = (input_shape[0], *input_shape[-2:])
        mask = torch.zeros(*pgm_shape, self.kernel_size, self.kernel_size).type_as(self.maskconv_kernel)
        return mask
    
    # TODO: optimize _pgm_topo_groups and _pgm_sort_topo


class CheckerboardMaskConv2DGaussianPriorCoder(GaussianMaskConv2DPriorCoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        checkerboard_mask = torch.zeros(self.kernel_size, self.kernel_size)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if (i+j) % 2 == 1:
                    checkerboard_mask[i, j] = 1
        self.register_buffer("checkerboard_mask", checkerboard_mask, persistent=False)

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        # TODO: get shape from bytes?
        pgm_shape = (input_shape[0], *input_shape[-2:])
        mask = torch.zeros(*pgm_shape, self.kernel_size, self.kernel_size).type_as(self.maskconv_kernel)
        checkerboard_mask = self.checkerboard_mask
        mask[..., :, :] = checkerboard_mask.unsqueeze(0).unsqueeze(0)
        checkerboard_mask_h_0 = torch.arange(0, input.shape[-2], 2, dtype=torch.long, device=input.device)
        checkerboard_mask_h_1 = torch.arange(1, input.shape[-2], 2, dtype=torch.long, device=input.device)
        checkerboard_mask_w_0 = torch.arange(0, input.shape[-1], 2, dtype=torch.long, device=input.device)
        checkerboard_mask_w_1 = torch.arange(1, input.shape[-1], 2, dtype=torch.long, device=input.device)
        checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
        checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
        mask[..., checkerboard_index_h_01, checkerboard_index_w_01, :, :] = 0
        mask[..., checkerboard_index_h_10, checkerboard_index_w_10, :, :] = 0
        mask[..., checkerboard_index_h_01, checkerboard_index_w_01, :, :] = 0
        mask[..., checkerboard_index_h_10, checkerboard_index_w_10, :, :] = 0
        # from torchvision.utils import save_image
        # save_image(mask[0].reshape(-1, 1, 5, 5), "checkerboard_mask.png", nrow=mask.shape[1])
        return mask

    # TODO: optimize _pgm_topo_groups and _pgm_sort_topo


# TODO: not working for now! require 3d unfold implementation!
class GaussianMaskConv3DPriorCoder(NNTrainablePGMPriorCoder, GaussianPGMPriorCoderImpl):
    def __init__(self, in_channels=256, kernel_size=5, channel_kernel_size=None, channel_group_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        # As there-s no 3d unfold implementation, we actually use 2d maskconv so channel_kernel_size is unavailable
        self.channel_kernel_size = kernel_size if channel_kernel_size is None else channel_kernel_size
        self.channel_group_size = channel_group_size
        # self.padding = (self.channel_kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2)
        self.padding = (self.kernel_size // 2, self.kernel_size // 2)
        self.maskconv_kernel = nn.Parameter(torch.zeros(self.in_channels * self.num_dist_params, self.in_channels, self.kernel_size, self.kernel_size))

        self.param_merger = nn.Sequential(
            nn.Conv2d(self.channel_group_size * self.num_dist_params * 2, self.channel_group_size * self.num_dist_params * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.channel_group_size * self.num_dist_params * 5 // 3, self.channel_group_size * self.num_dist_params * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.channel_group_size * self.num_dist_params * 4 // 3, self.channel_group_size * self.num_dist_params, 1),
        )

    def _get_pgm_3d(self, input):
        mask = torch.ones(*input.shape[-2:], self.kernel_size, self.kernel_size).type_as(self.maskconv_kernel)
        mask[..., self.channel_kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2 :] = 0
        mask[..., self.channel_kernel_size // 2, self.kernel_size // 2 + 1 :, :] = 0
        mask[..., self.channel_kernel_size // 2 + 1 :, :, :] = 0
        return mask
    
    def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None) -> torch.Tensor:
        if prior_params is None:
            prior_params = torch.zeros_like(pgm_params)
        assert pgm_params.shape[1] == prior_params.shape[1]
        concat_params = torch.cat([pgm_params, prior_params], dim=1)
        # .movedim(-1, 2).reshape(pgm_params.shape[0], -1, *pgm_params.shape[2:-1])
        return self.param_merger(concat_params)

    def _get_params_with_pgm(self, input: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
        num_channel_groups = input.shape[1] // self.channel_group_size
        assert(self.channel_group_size * num_channel_groups == input.shape[1])
        input_channel_group = input.reshape(input.shape[0], self.channel_group_size, num_channel_groups, *input.shape[2:])
        input_unfold = F.unfold(input_channel_group, (self.channel_kernel_size, self.kernel_size, self.kernel_size), padding=self.padding)
        pgm_3d = self._get_pgm_3d(input_channel_group)
        pgm_3d_unfold = pgm_3d.reshape(1, *input_channel_group.shape[-3:], -1).movedim(-1, 1)
        pgm_3d_unfold = pgm_3d_unfold.reshape(pgm_3d_unfold.shape[0], pgm_3d_unfold.shape[1], -1)\
            .repeat(1, self.channel_group_size, 1).contiguous()
        pgm_params_unfold = (input_unfold * pgm_3d_unfold).movedim(1, 2).matmul(self.maskconv_kernel.reshape(self.maskconv_kernel.shape[0], -1).t())
        pgm_params = pgm_params_unfold.movedim(1, 2).contiguous().view(pgm_params_unfold.shape[0], -1, *input.shape[2:])
        # mask = torch.ones_like(self.maskconv_kernel)
        # mask[..., self.kernel_size // 2, self.kernel_size // 2 :] = 0
        # mask[..., self.kernel_size // 2 + 1 :, :] = 0
        # pgm_params = F.conv2d(input, self.maskconv_kernel * mask, padding=self.padding)
        merged_params = self._merge_prior_params(pgm_params, prior_params=prior)
        return merged_params

    # def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None) -> torch.Tensor:
    #     if prior_params is None:
    #         prior_params = torch.zeros_like(pgm_params)
    #     assert pgm_params.shape[1] == prior_params.shape[1]
    #     concat_params = torch.cat([pgm_params, prior_params], dim=1)
    #     # .movedim(-1, 2).reshape(pgm_params.shape[0], -1, *pgm_params.shape[2:-1])
    #     return self.param_merger(concat_params)

    # def _get_params_with_pgm(self, input: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
    #     pgm_params = F.conv3d(input.unsqueeze(1), self.maskconv_kernel, padding=self.padding).movedim(1, -1)
    #     merged_params = self._merge_prior_params(pgm_params, prior_params=prior)
    #     return merged_params


class ConditionalTopoGroupGaussianMaskConv2DPriorCoder(GaussianMaskConv2DPriorCoder):
    def __init__(self, *args, 
                 num_topo_groups=2, 
                 max_len=5000,
                 topo_group_mask_prob_thres=None,
                 topo_group_mask_regularizer_weight=0.0,
                 topo_group_mask_normalize=False,
                 detach_pgm_mask=False,
                 predictor_model_type="simple",
                 predictor_soft_sampling=True,
                 predictor_mc_sampling=False, # deprecated
                 predictor_training_skip_sampling=False,
                 predictor_autoregressive_sampling=False,
                 predictor_autoregressive_sampling_num_iter=1,
                 predictor_add_pe=False,
                 predictor_pe_preprocess=False,
                 predictor_use_2d_pe=False,
                 predictor_pe_div=None,
                 predictor_pe_div_trainable=False,
                 predictor_use_pe_only=False,
                 predictor_use_attention=False,
                 predictor_use_global_logits=False,
                 predictor_use_noise_input=False,
                 predictor_global_logits_lr_modifier=1.0, # deprecated
                 predictor_global_logits_init_method="zeros",
                 predictor_logits_as_dp=False,
                 predictor_tile_logits_patch_size=None,
                 predictor_tile_samples_not_logits=False,
                 predictor_model_kernel_size=1,
                 predictor_model_num_layers=2,
                 predictor_model_no_bias=False,
                 predictor_model_lr_modifier=1.0,
                 predictor_output_weights=False,
                 predictor_regularizer_weight=0.0,
                 predictor_kl_prior_weight=0.0,
                 random_mask_weight=0.0, random_mask_weight_variable=False,
                 gs_temp=0.5, gs_temp_anneal=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_topo_groups = num_topo_groups
        self.topo_group_mask_prob_thres = topo_group_mask_prob_thres
        self.topo_group_mask_regularizer_weight = topo_group_mask_regularizer_weight
        self.topo_group_mask_normalize = topo_group_mask_normalize
        self.max_len = max_len
        self.detach_pgm_mask = detach_pgm_mask
        self.predictor_model_type = predictor_model_type
        self.predictor_soft_sampling = predictor_soft_sampling
        # self.predictor_mc_sampling = predictor_mc_sampling
        self.predictor_training_skip_sampling = predictor_training_skip_sampling
        self.predictor_autoregressive_sampling = predictor_autoregressive_sampling
        self.predictor_autoregressive_sampling_num_iter = predictor_autoregressive_sampling_num_iter
        self.predictor_add_pe = predictor_add_pe
        self.predictor_pe_preprocess = predictor_pe_preprocess
        self.predictor_use_2d_pe = predictor_use_2d_pe
        self.predictor_pe_div = predictor_pe_div
        self.predictor_pe_div_trainable = predictor_pe_div_trainable
        self.predictor_use_pe_only = predictor_use_pe_only
        self.predictor_use_attention = predictor_use_attention
        self.predictor_use_global_logits = predictor_use_global_logits
        self.predictor_use_noise_input = predictor_use_noise_input
        # self.predictor_global_logits_lr_modifier = predictor_global_logits_lr_modifier
        self.predictor_global_logits_init_method = predictor_global_logits_init_method
        self.predictor_logits_as_dp = predictor_logits_as_dp
        self.predictor_tile_logits_patch_size = predictor_tile_logits_patch_size
        self.predictor_tile_samples_not_logits = predictor_tile_samples_not_logits
        self.predictor_model_kernel_size = predictor_model_kernel_size
        self.predictor_model_num_layers = predictor_model_num_layers
        self.predictor_model_no_bias = predictor_model_no_bias
        self.predictor_model_lr_modifier = predictor_model_lr_modifier
        self.predictor_output_weights = predictor_output_weights
        self.predictor_regularizer_weight = predictor_regularizer_weight
        self.predictor_kl_prior_weight = predictor_kl_prior_weight

        self.random_mask_weight = random_mask_weight
        self.random_mask_weight_variable = random_mask_weight_variable
        if self.random_mask_weight_variable:
            self.random_mask_weight = nn.Parameter(torch.tensor(random_mask_weight), requires_grad=False)

        self.predictor_output_channels = self.num_topo_groups
        if self.predictor_output_weights:
            self.predictor_output_channels += 1


        if self.training_mc_for_ga:
            self.ga_genes = nn.ModuleList([self._init_topo_group_predictor() for _ in range(self.training_mc_for_ga_num_population)])
            # self._ga_update_population()
        else:
            self.topo_group_predictor = self._init_topo_group_predictor()


        if self.predictor_autoregressive_sampling:
            self.topo_group_autoregressive_predictor = nn.Sequential(
                MaskedConv2d(
                    self.predictor_output_channels, self.predictor_output_channels, kernel_size=self.kernel_size, padding=(self.kernel_size // 2), stride=1
                ),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.predictor_output_channels, self.predictor_output_channels, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.predictor_output_channels, self.predictor_output_channels, 1),
            )

        # self.topo_group_predictor = nn.Conv2d(self.in_channels * self.num_dist_params, self.predictor_output_channels, 1)
        # self.topo_group_predictor.weight.data[0] = 0.1
        # self.topo_group_predictor.weight.data[1] = 0.05
        # nn.init.constant_(self.topo_group_predictor.bias, 0.0)
        # self.topo_group_predictor.bias.data = torch.tensor([0.0 * self.in_channels, 0.05 * self.in_channels])

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

    def _init_topo_group_predictor(self, topo_group_predictor : Optional[nn.Module] = None):
        # topo_group_predictor = None
        if self.predictor_use_global_logits:
            # reinit from current value
            if topo_group_predictor is not None:
                # TODO: is this proper?
                topo_group_predictor.params.data *= 0.01
            else:
                global_logits = torch.zeros(self.predictor_output_channels, self.max_len, self.max_len)
                if self.predictor_global_logits_init_method == "zeros":
                    pass
                elif self.predictor_global_logits_init_method == "ps":
                    assert self.num_topo_groups > 1
                    scaling_factor = math.ceil(math.sqrt(self.num_topo_groups))
                    ps_length = scaling_factor*scaling_factor
                    base_indices = []
                    while len(base_indices) < ps_length:
                        base_indices += list(range(self.num_topo_groups))
                    base_indices = torch.as_tensor(base_indices[:ps_length]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    topo_group_indices = F.pixel_shuffle(
                        base_indices.repeat(1, 1, self.max_len // scaling_factor, self.max_len // scaling_factor),
                        scaling_factor
                    )[0, 0]
                    global_logits[:self.num_topo_groups] = F.one_hot(topo_group_indices, self.num_topo_groups).movedim(-1, 0)
                elif self.predictor_global_logits_init_method == "ps-checkerboard":
                    assert self.num_topo_groups > 1
                    scaling_factor = math.ceil(math.sqrt(self.num_topo_groups))
                    ps_length = scaling_factor*scaling_factor
                    base_indices = []
                    rev_flag = False
                    while len(base_indices) < ps_length:
                        base_indices += list(reversed(range(self.num_topo_groups)) if rev_flag else range(self.num_topo_groups))
                        rev_flag = not rev_flag
                    base_indices = torch.as_tensor(base_indices[:ps_length]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    topo_group_indices = F.pixel_shuffle(
                        base_indices.repeat(1, 1, self.max_len // scaling_factor, self.max_len // scaling_factor),
                        scaling_factor
                    )[0, 0]
                    global_logits[:self.num_topo_groups] = F.one_hot(topo_group_indices, self.num_topo_groups).movedim(-1, 0)
                else:
                    raise NotImplementedError(f"predictor_global_logits_init_method {self.predictor_global_logits_init_method} unknown!")

                # DP has zero grad for zero logits, use ones as base instead
                # if self.predictor_logits_as_dp:
                #     global_logits += torch.ones_like(global_logits)

                # TODO: only for backward compability! should remove this!
                self.global_logits = nn.Parameter(global_logits)
                # self.global_logits.lr_modifier = self.predictor_global_logits_lr_modifier
                # topo_group_predictor = lambda x : self.global_logits.unsqueeze(0)
                global_logits = global_logits.unsqueeze(0)
                topo_group_predictor = NNParameterGenerator(global_logits.shape, init_method="value", init_value=global_logits)
        else:
            if topo_group_predictor is not None:
                # TODO: better reinit?
                for param in topo_group_predictor.parameters():
                    param.data = param.data + torch.normal(torch.zeros_like(param), std=0.1)
            else:
                if self.predictor_use_pe_only or self.predictor_add_pe:
                    d_model = self.in_channels * self.num_dist_params
                    pe = torch.zeros(self.max_len, d_model)
                    position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                    if self.predictor_pe_div is not None:
                        div_term = self.predictor_pe_div
                    if self.predictor_pe_div_trainable:
                        self.register_buffer('position', position, persistent=False)
                        self.pe_div_term = nn.Parameter(div_term)
                    else:
                        # pe[:, 0::2] = torch.sin(position * div_term)
                        # pe[:, 1::2] = torch.cos(position * div_term)
                        pe[:, 0::2] = torch.cos(position * div_term)
                        pe[:, 1::2] = torch.cos(position * div_term)
                        pe = pe.unsqueeze(0).movedim(1, 2)
                        self.register_buffer('pe', pe, persistent=False)

                    if self.predictor_pe_preprocess:
                        self.pe_preprosser = nn.Sequential(
                            nn.Conv2d(d_model, d_model, 5, padding=2),
                        ) 

                if self.predictor_model_type == "simple":
                
                    # TODO: using attention modules might be better
                    if self.predictor_use_attention:
                        self.attention_module = nn.MultiheadAttention(self.in_channels * self.num_dist_params, self.num_dist_params)

                    topo_group_predictor_layers = [
                        nn.Conv2d(self.in_channels * self.num_dist_params, self.in_channels, self.predictor_model_kernel_size, padding=(self.predictor_model_kernel_size // 2), bias=(not self.predictor_model_no_bias)),
                        nn.LeakyReLU(inplace=True),
                    ]
                    for _ in range(self.predictor_model_num_layers-2):
                        topo_group_predictor_layers.extend([
                            nn.Conv2d(self.in_channels, self.in_channels, self.predictor_model_kernel_size, padding=(self.predictor_model_kernel_size // 2), bias=(not self.predictor_model_no_bias)),
                            nn.LeakyReLU(inplace=True),
                        ])
                    topo_group_predictor_layers.append(nn.Conv2d(self.in_channels, self.predictor_output_channels, 1, bias=(not self.predictor_model_no_bias)))
                    topo_group_predictor = nn.Sequential(*topo_group_predictor_layers)

                    for name, param in topo_group_predictor.named_parameters():
                        nn.init.constant_(param, 0.01)

                elif self.predictor_model_type == "unet":
                    topo_group_predictor = GeneratorUNet(
                        in_channels=self.in_channels * self.num_dist_params, 
                        out_channels=self.predictor_output_channels,
                        mid_channels=[64, 128, 256, 256],
                        dropout_probs=[0.0, 0.0, 0.5, 0.5]
                    )
                elif self.predictor_model_type == "gan_g":
                    ngf = self.in_channels
                    topo_group_predictor = nn.Sequential(
                        # input is Z, going into a convolution
                        nn.ConvTranspose2d( self.in_channels * self.num_dist_params, ngf * 8, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(ngf * 8),
                        nn.ReLU(True),
                        # state size. ``(ngf*8) x 4 x 4``
                        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 4),
                        nn.ReLU(True),
                        # state size. ``(ngf*4) x 8 x 8``
                        nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 2),
                        nn.ReLU(True),
                        # state size. ``(ngf*2) x 16 x 16``
                        nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf),
                        nn.ReLU(True),
                        # state size. ``(ngf) x 32 x 32``
                        nn.ConvTranspose2d( ngf, self.predictor_output_channels, 4, 2, 1, bias=False),
                        # nn.Tanh()
                        # state size. ``(nc) x 64 x 64``
                    )
                elif self.predictor_model_type == "transgan_g":
                    from cbench.nn.models.transgan_generator import Generator
                    topo_group_predictor = Generator(
                        latent_dim=self.in_channels * self.num_dist_params, 
                        out_dim=self.predictor_output_channels,
                        # bottom_width=8,
                        # embed_dim=192,
                        # gf_dim=512,
                        # g_depth=[2,2,2],
                    )

                elif self.predictor_model_type == "diffusion":
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()
            
        for param in topo_group_predictor.parameters():
            param.lr_modifier = self.predictor_model_lr_modifier

        return topo_group_predictor

    def _get_pe(self, prior):
        if self.predictor_pe_div_trainable:
            pe = torch.zeros(self.max_len, self.in_channels * self.num_dist_params).type_as(self.pe_div_term)
            self.update_cache("moniter_dict", pe_div_term_mean=self.pe_div_term.mean())
            pe[:, 0::2] = torch.sin(self.position * self.pe_div_term)
            pe[:, 1::2] = torch.cos(self.position * self.pe_div_term)
            pe = pe.unsqueeze(0).movedim(1, 2)
        else:
            pe = self.pe

        if self.predictor_use_2d_pe:
            pe = 2 * pe[:, :, :prior.shape[-2], None] + pe[:, :, None, :prior.shape[-1]]
        else:
            spatial_size = np.prod(prior.shape[2:])
            assert spatial_size <= pe.shape[2]
            pe = pe[:, :, :spatial_size].reshape(1, *prior.shape[1:])
        if self.predictor_pe_preprocess:
            pe = self.pe_preprosser(pe)
        return pe
    
    def _pgm_topo_group_logits(self, topo_group_predictor : Optional[nn.Module] = None, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, **kwargs):
        batch_size, channels, height, width = input_shape
        # NOTE: tmp fix for no prior
        if prior is None:
            prior = torch.zeros_like(input).repeat(1, self.num_dist_params, 1, 1)

        if topo_group_predictor is None:
            if self.training_mc_for_ga:
                topo_group_predictor = self.ga_genes[self.ga_fitness.argmax()]
            else:
                topo_group_predictor = self.topo_group_predictor

        if self.predictor_use_noise_input:
            if self.training and self.training_mc_sampling:
                training_mc_num_samples = self.training_mc_num_samples if not self.training_mc_for_ga else self.training_mc_num_samples // self.training_mc_for_ga_num_population
                prior = prior.reshape(batch_size // training_mc_num_samples, training_mc_num_samples, *prior.shape[1:])[:, 0]
                prior = torch.normal(prior).unsqueeze(1).repeat(1, training_mc_num_samples, *([1]*(prior.ndim-1)))\
                    .reshape(batch_size, *prior.shape[1:])
            else:
                prior = torch.normal(prior)

        if self.predictor_tile_logits_patch_size is not None:
            height, width = self.predictor_tile_logits_patch_size

        # if self.training_mc_for_ga:
        #     if self.training:
        #         topo_group_logits = self.ga_genes[..., :height, :width].unsqueeze(0).repeat(batch_size // self.training_mc_for_ga_num_population, 1, 1, 1, 1)
        #         topo_group_logits = topo_group_logits.reshape(batch_size, *topo_group_logits.shape[2:])
        #     else:
        #         self.update_cache("hist_dict", ga_fitness=self.ga_fitness)
        #         # select best gene
        #         topo_group_logits = self.ga_genes[self.ga_fitness.argmax(), ..., :height, :width].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # else:

        # if self.predictor_use_global_logits:
        #     topo_group_logits = self.global_logits[..., :height, :width].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # else:
        if self.predictor_use_pe_only:
            prior = torch.zeros_like(prior) + self._get_pe(prior)
        else:
            if self.predictor_add_pe:
                prior = prior + self._get_pe(prior)
            if self.predictor_use_attention:
                prior_attention = prior.reshape(prior.shape[0], prior.shape[1], -1).movedim(-1, 0)
                prior_attention, _ = self.attention_module(prior_attention, prior_attention, prior_attention, need_weights=False)
                prior = prior_attention.movedim(0, -1).reshape_as(prior)
        
        prior_pred = prior
        if self.predictor_model_type == "gan_g" or self.predictor_model_type == "transgan_g":
            prior_pred = prior_pred[:1, ..., :1, :1]
        topo_group_logits = topo_group_predictor(prior_pred)
        # if self.predictor_model_type == "gan_g" or self.predictor_model_type == "transgan_g":
        topo_group_logits = topo_group_logits[..., :height, :width].repeat(batch_size, 1,1,1)

        if self.predictor_tile_logits_patch_size is not None and not self.predictor_tile_samples_not_logits:
            topo_group_logits = topo_group_logits.reshape(batch_size, -1, 1).repeat(1, 1, math.floor(input_shape[-2] / height) * math.floor(input_shape[-1] / width))
            topo_group_logits_fold = F.fold(topo_group_logits, input_shape[-2:], (height, width), stride=(height, width))
            topo_group_logits = topo_group_logits_fold # topo_group_logits_fold[:, :input_shape[-2], :input_shape[-1]].reshape(batch_size, -1, *input_shape[-2:])

        return topo_group_logits

    def _pgm_sample_from_topo_group_logits(self, topo_group_logits : torch.Tensor):
        # nan check
        if torch.isnan(topo_group_logits).any():
            return torch.zeros_like(topo_group_logits).movedim(1, -1)

        # soft sampling
        force_hard_sampling = self.training_mc_sampling # and not self.training_mc_for_ga
        if self.training and self.predictor_soft_sampling and not force_hard_sampling:
            if self.predictor_training_skip_sampling:
                topo_group_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
            else:
                topo_group_dist = D.RelaxedOneHotCategorical(self.gs_temp, logits=topo_group_logits.movedim(1, -1))
                topo_group_probs = topo_group_dist.rsample()
        # hard sampling
        else:
            # TODO: faster hard-sampling?
            if force_hard_sampling:
                topo_group_dist = D.Categorical(logits=topo_group_logits.movedim(1, -1))
                topo_group_indices = topo_group_dist.sample().long()
                # topo_group_indices_neg_logprob = topo_group_dist.log_prob(topo_group_indices)
            else:
                topo_group_indices = topo_group_logits.argmax(1)
            
            # logging
            # topo_group_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
            # self.update_cache("image_dict", topo_group_probs_mean0=(topo_group_probs[..., 0].unsqueeze(1).float()))
            # self.update_cache("image_dict", input_images=input[:, [2,1,0]]) # BGR to RGB
            topo_group_probs = F.one_hot(topo_group_indices, self.num_topo_groups).type_as(topo_group_logits)
            # self.update_cache("hist_dict", topo_group_counts=topo_group_probs.view(-1, self.num_topo_groups).sum(0))
            # straight through
            if self.training:
                if not force_hard_sampling:
                #     topo_group_indices_logprob = topo_group_dist.logits * topo_group_probs
                #     self.update_cache(pgm_log_prob=topo_group_indices_logprob)
                # else:
                    topo_group_soft_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
                    topo_group_probs = (topo_group_probs + topo_group_soft_probs) - topo_group_soft_probs.detach()

        return topo_group_probs

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        mask = super()._get_pgm(input, *args, prior=prior, input_shape=input_shape, pgm=pgm, **kwargs)
        batch_size, channels, height, width = input_shape
        # NOTE: tmp fix for no prior
        if prior is None:
            prior = torch.zeros_like(input).repeat(1, self.num_dist_params, 1, 1)

        # if self.predictor_use_global_logits:
        #     topo_group_logits = self.global_logits[..., :height, :width].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # else:
        #     if self.predictor_use_pe_only:
        #         prior = torch.zeros_like(prior) + self._get_pe(prior)
        #     else:
        #         if self.predictor_add_pe:
        #             prior = prior + self._get_pe(prior)
        #         if self.predictor_use_attention:
        #             prior_attention = prior.reshape(prior.shape[0], prior.shape[1], -1).movedim(-1, 0)
        #             prior_attention, _ = self.attention_module(prior_attention, prior_attention, prior_attention, need_weights=False)
        #             prior = prior_attention.movedim(0, -1).reshape_as(prior)
        #     topo_group_logits = self.topo_group_predictor(prior)

        if topo_group_logits is None:
            if self.training and self.training_mc_for_ga:
                # NOTE: tmp solution for updating genes from last population update
                if hasattr(self, "new_ga_genes"):
                    self.ga_genes.load_state_dict(self.new_ga_genes[0].state_dict())
                    delattr(self, "new_ga_genes")
                prior_ga_reshape = prior.reshape(batch_size // self.training_mc_for_ga_num_population, self.training_mc_for_ga_num_population, channels * self.num_dist_params, height, width)
                topo_group_logits = torch.stack([
                    self._pgm_topo_group_logits(topo_group_predictor=self.ga_genes[idx], prior=prior_ga.squeeze(1), input_shape=(batch_size // self.training_mc_for_ga_num_population, channels, height, width)) \
                        for idx, prior_ga in enumerate(prior_ga_reshape.split(1, dim=1))
                ], dim=1).reshape(batch_size, -1, height, width)
            else:
                topo_group_logits = self._pgm_topo_group_logits(prior=prior, input_shape=input_shape)
        
        if self.predictor_output_weights:
            topo_group_logits, topo_group_weights = torch.split(topo_group_logits, (self.num_topo_groups, 1), dim=1)

        if self.predictor_logits_as_dp:
            topo_group_probs = D.Dirichlet(topo_group_logits.exp().movedim(1, -1)).rsample()
            topo_group_logits = topo_group_probs.clamp_min(self.eps).log().movedim(-1, 1)

        if self.training and self.predictor_regularizer_weight > 0:
            topo_group_probs_mean = torch.softmax(topo_group_logits.movedim(1, -1).reshape(-1, self.num_topo_groups), dim=-1).mean(dim=0)
            topo_group_predictor_regularizer = F.mse_loss(topo_group_probs_mean, torch.zeros_like(topo_group_probs_mean) + 1 / self.num_topo_groups)
            self.update_cache("loss_dict", topo_group_predictor_regularizer=topo_group_predictor_regularizer * self.predictor_regularizer_weight)

        if self.training and self.predictor_kl_prior_weight > 0:
            posterior_logits = torch.log_softmax(topo_group_logits.movedim(1, -1).reshape(-1, self.num_topo_groups), dim=-1)
            posterior_probs = posterior_logits.exp()
            kld = (posterior_probs * (posterior_logits + np.log(self.num_topo_groups)))
            kld[posterior_probs==0] = 0 # prevent nan
            topo_group_predictor_kl_prior = kld.sum() # / batch_size
            self.update_cache("loss_dict", topo_group_predictor_kl_prior=topo_group_predictor_kl_prior * self.predictor_kl_prior_weight)

        # if self.training and self.predictor_soft_sampling and not self.training_mc_sampling:
        #     if self.predictor_training_skip_sampling:
        #         topo_group_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
        #     else:
        #         topo_group_dist = D.RelaxedOneHotCategorical(self.gs_temp, logits=topo_group_logits.movedim(1, -1))
        #         topo_group_probs = topo_group_dist.rsample()
        # else:
        #     # TODO: faster hard-sampling?
        #     if self.training_mc_sampling:
        #         topo_group_dist = D.Categorical(logits=topo_group_logits.movedim(1, -1))
        #         topo_group_indices = topo_group_dist.sample().long()
        #         # topo_group_indices_neg_logprob = topo_group_dist.log_prob(topo_group_indices)
        #     else:
        #         topo_group_indices = topo_group_logits.argmax(1)
        #     # logging
        #     # topo_group_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
        #     # self.update_cache("image_dict", topo_group_probs_mean0=(topo_group_probs[..., 0].unsqueeze(1).float()))
        #     # self.update_cache("image_dict", input_images=input[:, [2,1,0]]) # BGR to RGB
        #     topo_group_probs = F.one_hot(topo_group_indices, self.num_topo_groups).type_as(topo_group_logits)
        #     # self.update_cache("hist_dict", topo_group_counts=topo_group_probs.view(-1, self.num_topo_groups).sum(0))
        #     # straight through
        #     if self.training:
        #         if not self.training_mc_sampling:
        #         #     topo_group_indices_logprob = topo_group_dist.logits * topo_group_probs
        #         #     self.update_cache(pgm_log_prob=topo_group_indices_logprob)
        #         # else:
        #             topo_group_soft_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
        #             topo_group_probs = (topo_group_probs + topo_group_soft_probs) - topo_group_soft_probs.detach()
        topo_group_sample_weights = self._pgm_sample_from_topo_group_logits(topo_group_logits)
        # autoregressive sample
        if self.predictor_autoregressive_sampling:
            for _ in range(self.predictor_autoregressive_sampling_num_iter):
                topo_group_logits = self.topo_group_autoregressive_predictor(topo_group_sample_weights.movedim(-1, 1))
                topo_group_sample_weights = self._pgm_sample_from_topo_group_logits(topo_group_logits)

        if self.predictor_tile_logits_patch_size is not None and self.predictor_tile_samples_not_logits:
            patch_height, patch_width = self.predictor_tile_logits_patch_size
            topo_group_logits = topo_group_logits.reshape(batch_size, -1, 1).repeat(1, 1, math.floor(input_shape[-2] / patch_height) * math.floor(input_shape[-1] / patch_width))
            topo_group_logits = F.fold(topo_group_logits, input_shape[-2:], (patch_height, patch_width), stride=(patch_height, patch_width))
            topo_group_sample_weights = topo_group_sample_weights.movedim(-1, 1).reshape(batch_size, -1, 1).repeat(1, 1, math.floor(input_shape[-2] / patch_height) * math.floor(input_shape[-1] / patch_width))
            topo_group_sample_weights = F.fold(topo_group_sample_weights, input_shape[-2:], (patch_height, patch_width), stride=(patch_height, patch_width)).movedim(1, -1)

        # logging
        if self.training:
            if self.training_mc_sampling:
                topo_group_indices_logprob = torch.log_softmax(topo_group_logits.movedim(1, -1), dim=-1) * topo_group_sample_weights
                self.update_cache(pgm_log_prob=topo_group_indices_logprob)
        else:
            self.update_cache("image_dict", topo_group_indices=(topo_group_sample_weights.movedim(-1, 1).argmax(1, keepdim=True).float() / (self.num_topo_groups-1)))

        # from torchvision.utils import save_image
        # save_image((topo_group_indices.unsqueeze(1).float() / (self.num_topo_groups-1))[0], "topo_group_indices.png")
        # save_image(torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)[0, ..., 0], "topo_group_sample_weights0.png")

        self.update_cache("moniter_dict", topo_group_probs_mean0=topo_group_sample_weights[..., 0].mean())

        batch_size = topo_group_logits.shape[0]
        # topo_group_logits_unfold = F.unfold(topo_group_logits, (self.kernel_size, self.kernel_size), padding=self.padding)
        topo_group_sample_weights_unfold = F.unfold(topo_group_sample_weights.movedim(-1, 1), (self.kernel_size, self.kernel_size), padding=self.padding)\
            .reshape(batch_size, 1, self.num_topo_groups, self.kernel_size*self.kernel_size, -1).permute(0, 4, 3, 2, 1)
        # NOTE : ********************************************************************
        # calculate p(kernel_topo_group_idx < topo_group_idx) 
        # = sum_{kernel_topo_group_idx, topo_group_idx}(triu(diagonal=1) * p[kernel_topo_group_idx, topo_group_idx])
        # E.g. num_topo_groups=2, p(kernel_topo_group_idx < topo_group_idx) = p(0,1)
        # [[p(0,0), p(0,1)],    *   [[0,    1]  =   [[0,    p(0,1) ],
        # [ p(1,0), p(1,1)]]        [ 0,    0]]     [ 0,    0      ]]
        # NOTE : ********************************************************************
        topo_group_sample_weights_matrices = topo_group_sample_weights.reshape(batch_size, -1, 1, 1, self.num_topo_groups) * topo_group_sample_weights_unfold
        topo_group_mask_prob = torch.triu(topo_group_sample_weights_matrices, diagonal=1).sum((-2, -1)).reshape(batch_size, *prior.shape[-2:], self.kernel_size, self.kernel_size)
        if self.topo_group_mask_prob_thres is not None:
            topo_group_mask = (topo_group_mask_prob > self.topo_group_mask_prob_thres).type_as(topo_group_mask_prob)
            # straight through
            if self.training:
                topo_group_mask = (topo_group_mask + topo_group_mask_prob) - topo_group_mask_prob.detach()
        else:
            topo_group_mask = topo_group_mask_prob
        self.update_cache("moniter_dict", topo_group_mask_mean=topo_group_mask.mean())

        if self.training and self.topo_group_mask_regularizer_weight > 0:
            topo_group_mask_regularizer = 1 - topo_group_mask_prob.mean()
            self.update_cache("loss_dict", topo_group_mask_regularizer=topo_group_mask_regularizer * self.topo_group_mask_regularizer_weight)

        if self.topo_group_mask_normalize:
            topo_group_mask = topo_group_mask / topo_group_mask.sum(dim=(-2, -1), keepdim=True)

        if self.predictor_output_weights:
            topo_group_weights_unfold = F.unfold(topo_group_weights, (self.kernel_size, self.kernel_size), padding=self.padding).permute(0, 2, 1)\
                .reshape(batch_size, *prior.shape[-2:], self.kernel_size, self.kernel_size)
            topo_group_mask = topo_group_mask * topo_group_weights_unfold

        # if not self.training:
        #     valid_mask_idx = topo_group_sample_weights.view(topo_group_sample_weights.shape[0], -1, self.num_topo_groups).sum(1)[:, 0].argmax()
        #     self.update_cache("image_dict", **{f"topo_group_mask_{valid_mask_idx}" : make_grid(topo_group_mask[valid_mask_idx].reshape(-1, 1, *topo_group_mask_prob.shape[-2:]), nrow=topo_group_mask_prob.shape[1], padding=1)})
        
        # from torchvision.utils import save_image
        # save_image(topo_group_mask[0].reshape(-1, 1, 5, 5), "topo_group_mask.png", nrow=topo_group_mask.shape[1])

        if self.detach_pgm_mask:
            topo_group_mask = topo_group_mask.detach()

        # TODO: should be placed before mask_regularizer
        if self.random_mask_weight > 0:
            random_mask = (torch.rand_like(topo_group_mask) > 0.5).type_as(topo_group_mask)
            random_mask[..., (self.kernel_size) // 2, (self.kernel_size) // 2] = 0
            topo_group_mask = topo_group_mask * (1 - self.random_mask_weight) + random_mask * self.random_mask_weight

        # annealing
        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    gs_temp=self.gs_temp
                )
        
        if self.random_mask_weight_variable:
            if self.training:
                self.update_cache("moniter_dict", 
                    random_mask_weight=self.random_mask_weight
                )

        return torch.ones_like(mask) * topo_group_mask

    def _pgm_topo_groups(self, input : torch.Tensor, topo_group_logits: Any, **kwargs) -> List[torch.LongTensor]:
        if self.predictor_output_weights:
            topo_group_logits, topo_group_weights = torch.split(topo_group_logits, (self.num_topo_groups, 1), dim=1)
        topo_group_indices = topo_group_logits.argmax(1, keepdim=True)
        topo_groups = [(topo_group_indices==i) for i in self.num_topo_groups]
        return topo_groups
    
    def _ga_update_population(self, parents=None):
        # if not hasattr(self, "ga_genes"):
        #     # initialization
        #     self.ga_genes = nn.Parameter(torch.zeros(self.training_mc_for_ga_num_population, self.predictor_output_channels, self.max_len, self.max_len))
        #     # nn.init.uniform_(self.ga_genes, -1, 1)
        #     # self.ga_genes.lr_modifier=100.0
        # else:
        #     if parents is None:
        #         _, fitness_topk_indices = self.ga_fitness.topk(self.training_mc_for_ga_keep_parents)
        #         parent_genes = self.ga_genes[fitness_topk_indices]
        #     else:
        #         parent_genes = parents
        #     num_new_children = self.training_mc_for_ga_num_new_children if self.training_mc_for_ga_num_new_children >= 0 else self.training_mc_for_ga_num_population - parent_genes.shape[0]
        #     children_genes = torch.zeros_like(self.ga_genes)[:num_new_children]
        #     children_num_crossover = self.training_mc_for_ga_num_population - num_new_children - parent_genes.shape[0]
        #     if children_num_crossover > 0:
        #         # TODO: crossover method?
        #         children_crossover_genes = parent_genes.repeat(math.ceil(children_num_crossover / parent_genes.shape[0]), 1, 1, 1)[:children_num_crossover]
        #         # TODO: noise magnitude
        #         children_crossover_genes = children_crossover_genes * torch.zeros_like(children_crossover_genes).uniform_(-1, 1)
        #         children_genes = torch.cat([children_genes, children_crossover_genes], dim=0)
        #     self.ga_genes = torch.cat([parent_genes, children_genes], dim=0)
        new_ga_genes = []
        if parents is None:
            _, fitness_topk_indices = self.ga_fitness.topk(self.training_mc_for_ga_keep_parents)
            parent_genes = [self.ga_genes[i] for i in fitness_topk_indices]
        else:
            parent_genes = parents
        new_ga_genes.extend(parent_genes)
        num_new_children = self.training_mc_for_ga_num_new_children if self.training_mc_for_ga_num_new_children >= 0 else self.training_mc_for_ga_num_population - len(parent_genes)
        for _ in range(num_new_children):
            new_ga_genes.append(self._init_topo_group_predictor())
        children_num_crossover = self.training_mc_for_ga_num_population - num_new_children - len(parent_genes)
        for i in range(children_num_crossover):
            # TODO: crossover method?
            crossover_gene = copy.deepcopy(parent_genes[i % len(parent_genes)])
            new_ga_genes.append(self._init_topo_group_predictor(crossover_gene))
        new_ga_genes = nn.ModuleList(new_ga_genes)
        # update gene states 
        # NOTE: this activates inplace modification, which throws error in autograd
        # self.ga_genes.load_state_dict(new_ga_genes.state_dict())
        # so we just cache new params and update it in the next forward pass
        self.new_ga_genes = [new_ga_genes]
        # self.ga_fitness_window.data.fill_(0)
        # tmp disable gradient as we have modified the values
        # for param in self.ga_genes.parameters():
        #     param.requires_grad = False

    def encode(self, input : torch.Tensor, *args, prior : torch.Tensor = None, **kwargs) -> bytes:
        with self.profiler.start_time_profile("time_data_preprocess_encode"):
            # input_quant = self.quantize(input, "symbols")
            input_quant = self._data_preprocess(input)
            data = input_quant.detach().cpu().numpy().astype(np.int32)

        with self.profiler.start_time_profile("time_prior_preprocess_encode"):
            # get topo logits first
            topo_group_logits = self._pgm_topo_group_logits(prior=prior, input_shape=input.shape)
            # generate order and pgm masks
            pgm_topo_order = self._pgm_sort_topo(input_quant, topo_group_logits)
            pgm = self._get_pgm(input_quant.type_as(input), prior=prior, input_shape=input.shape, topo_group_logits=topo_group_logits)
            params = self._pgm_inference(input_quant.type_as(input), pgm, prior=prior)
            indexes, data_offsets = self._build_coding_params(params)
            data = data - data_offsets
            if pgm_topo_order is not None:
                pgm_topo_order = pgm_topo_order.detach().cpu().numpy().astype(np.int32)
                data = data.reshape(-1)[pgm_topo_order]
                indexes = indexes.reshape(-1)[pgm_topo_order]

        with self.profiler.start_time_profile("time_ans_encode"):
            byte_string = self.ans_encoder.encode_with_indexes(data, indexes)

        return byte_string

    def decode(self, byte_string : bytes, *args, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        self.ans_decoder.set_stream(byte_string)
        data_buffer = torch.zeros(prior.shape[0], self.in_channels, *prior.shape[2:]).type_as(prior)

        with self.profiler.start_time_profile("time_prior_preprocess_decode"):
            # get topo logits first
            topo_group_logits = self._pgm_topo_group_logits(prior=prior, input_shape=data_buffer.shape)
            # generate order and pgm masks
            pgm_topo_groups = self._pgm_topo_groups(input, pgm)
            # pgm_topo_order = self._pgm_sort_topo(data_buffer, topo_group_logits)
            pgm = self._get_pgm(data_buffer, prior=prior, input_shape=data_buffer.shape, topo_group_logits=topo_group_logits)
        
        last_group = None
        # TODO: update grouping method
        for group in pgm_topo_groups:
            if last_group is None: last_group = torch.zeros_like(data_buffer[group])
            params = self._get_params_with_pgm(last_group, pgm[group], prior[group])
            indexes, data_offsets = self._build_coding_params(params)
            symbols = self.ans_decoder.decode_stream(indexes.detach().cpu().numpy().astype(np.int32))
            symbols = symbols + data_offsets.detach().cpu().numpy().astype(np.int32)
            last_group = self._data_postprocess(symbols)
            data_buffer[group] = last_group
        return data_buffer


class GaussianEntroFormerPriorCoder(GaussianPGMPriorCoderImpl, NNTrainablePGMPriorCoder):
    def __init__(self, *args, in_channels=256, 
                 rpe_shared=True,
                 mask_ratio=0.,
                 dim_embed=384,
                 depth=6,
                 heads=6,
                 dim_head=64,
                 mlp_ratio=4,
                 dropout=0.,
                 position_num=7,
                 attn_topk=-1,
                 att_scale=True,
                 **kwargs):
        NNTrainablePGMPriorCoder.__init__(self, *args, **kwargs)
        GaussianPGMPriorCoderImpl.__init__(self, *args, in_channels=in_channels, **kwargs)
        self.in_channels = in_channels

        self.entroformer_context = TransDecoder(self.in_channels, self.in_channels * self.num_dist_params, 
                                                rpe_shared=rpe_shared,
                                                mask_ratio=mask_ratio,
                                                dim_embed=dim_embed,
                                                depth=depth,
                                                heads=heads,
                                                dim_head=dim_head,
                                                mlp_ratio=mlp_ratio,
                                                dropout=dropout,
                                                position_num=position_num,
                                                attn_topk=attn_topk,
                                                att_scale=att_scale,
                                                **kwargs)

        self.param_merger = nn.Sequential(
            nn.Conv2d(self.in_channels * self.num_dist_params * 2, self.in_channels * self.num_dist_params * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.num_dist_params * 5 // 3, self.in_channels * self.num_dist_params * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.num_dist_params * 4 // 3, self.in_channels * self.num_dist_params, 1),
        )

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        batch_size, channels, height, width = input_shape
        mask, token_mask, input_mask, output_mask = self.entroformer_context.get_mask(batch_size, height, width)
        return mask, token_mask, input_mask, output_mask

    def _merge_prior_params(self, pgm_params : torch.Tensor, prior_params : torch.Tensor = None) -> torch.Tensor:
        if prior_params is None:
            prior_params = torch.zeros_like(pgm_params)
        assert pgm_params.shape[1] == prior_params.shape[1]
        concat_params = torch.cat([pgm_params, prior_params], dim=1)
        # .movedim(-1, 2).reshape(pgm_params.shape[0], -1, *pgm_params.shape[2:-1])
        return self.param_merger(concat_params)
    
    def _pgm_inference(self, input : torch.Tensor, pgm : Any, prior : torch.Tensor = None):
        pgm_params = self.entroformer_context(input, pgm)
        merged_params = self._merge_prior_params(pgm_params, prior_params=prior)
        return merged_params


class GaussianEntroFormerCheckerboardPriorCoder(GaussianEntroFormerPriorCoder):
    def __init__(self, *args, in_channels=256, **kwargs):
        super().__init__(*args, in_channels=in_channels, **kwargs)
        self.entroformer_context = TransDecoderCheckerboard(self.in_channels, self.in_channels * self.num_dist_params, **kwargs)


class ConditionalTopoGroupGaussianEntroFormerPriorCoder(GaussianEntroFormerPriorCoder):
    def __init__(self, *args, 
                 num_topo_groups=2, 
                 max_len=5000,
                 topo_group_mask_prob_thres=None,
                 topo_group_mask_regularizer_weight=0.0,
                 topo_group_mask_normalize=False,
                 detach_pgm_mask=False,
                 predictor_soft_sampling=True,
                 predictor_mc_sampling=False,
                 predictor_training_skip_sampling=False,
                 predictor_add_pe=False,
                 predictor_pe_preprocess=False,
                 predictor_use_2d_pe=False,
                 predictor_pe_div=None,
                 predictor_pe_div_trainable=False,
                 predictor_use_pe_only=False,
                 predictor_use_attention=False,
                 predictor_output_weights=False,
                 predictor_regularizer_weight=0.0,
                 gs_temp=0.5, gs_temp_anneal=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_topo_groups = num_topo_groups
        self.topo_group_mask_prob_thres = topo_group_mask_prob_thres
        self.topo_group_mask_regularizer_weight = topo_group_mask_regularizer_weight
        self.topo_group_mask_normalize = topo_group_mask_normalize
        self.max_len = max_len
        self.detach_pgm_mask = detach_pgm_mask
        self.predictor_soft_sampling = predictor_soft_sampling
        self.predictor_mc_sampling = predictor_mc_sampling
        self.predictor_training_skip_sampling = predictor_training_skip_sampling
        self.predictor_add_pe = predictor_add_pe
        self.predictor_pe_preprocess = predictor_pe_preprocess
        self.predictor_use_2d_pe = predictor_use_2d_pe
        self.predictor_pe_div_trainable = predictor_pe_div_trainable
        self.predictor_use_pe_only = predictor_use_pe_only
        self.predictor_use_attention = predictor_use_attention
        self.predictor_output_weights = predictor_output_weights
        self.predictor_regularizer_weight = predictor_regularizer_weight

        if self.predictor_use_pe_only or self.predictor_add_pe:
            d_model = self.in_channels * self.num_dist_params
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            if predictor_pe_div is not None:
                div_term = predictor_pe_div
            if self.predictor_pe_div_trainable:
                self.register_buffer('position', position, persistent=False)
                self.pe_div_term = nn.Parameter(div_term)
            else:
                # pe[:, 0::2] = torch.sin(position * div_term)
                # pe[:, 1::2] = torch.cos(position * div_term)
                pe[:, 0::2] = torch.cos(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).movedim(1, 2)
                self.register_buffer('pe', pe, persistent=False)

            if self.predictor_pe_preprocess:
                self.pe_preprosser = nn.Sequential(
                    nn.Conv2d(d_model, d_model, 5, padding=2),
                ) 

        # TODO: using attention modules might be better
        output_channels = self.num_topo_groups
        if self.predictor_output_weights:
            output_channels += 1

        if self.predictor_use_attention:
            self.attention_module = nn.MultiheadAttention(self.in_channels * self.num_dist_params, self.num_dist_params)

        self.topo_group_predictor = nn.Sequential(
            nn.Conv2d(self.in_channels * self.num_dist_params, self.in_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.in_channels, output_channels, 1),
        )

        for name, param in self.topo_group_predictor.named_parameters():
            nn.init.constant_(param, 0.01)

        # self.topo_group_predictor = nn.Conv2d(self.in_channels * self.num_dist_params, output_channels, 1)
        # self.topo_group_predictor.weight.data[0] = 0.1
        # self.topo_group_predictor.weight.data[1] = 0.05
        # nn.init.constant_(self.topo_group_predictor.bias, 0.0)
        # self.topo_group_predictor.bias.data = torch.tensor([0.0 * self.in_channels, 0.05 * self.in_channels])

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

    def _get_pe(self, prior):
        if self.predictor_pe_div_trainable:
            pe = torch.zeros(self.max_len, self.in_channels * self.num_dist_params).type_as(self.pe_div_term)
            self.update_cache("moniter_dict", pe_div_term_mean=self.pe_div_term.mean())
            pe[:, 0::2] = torch.sin(self.position * self.pe_div_term)
            pe[:, 1::2] = torch.cos(self.position * self.pe_div_term)
            pe = pe.unsqueeze(0).movedim(1, 2)
        else:
            pe = self.pe

        if self.predictor_use_2d_pe:
            pe = 2 * pe[:, :, :prior.shape[-2], None] + pe[:, :, None, :prior.shape[-1]]
        else:
            spatial_size = np.prod(prior.shape[2:])
            assert spatial_size <= pe.shape[2]
            pe = pe[:, :, :spatial_size].reshape(1, *prior.shape[1:])
        if self.predictor_pe_preprocess:
            pe = self.pe_preprosser(pe)
        return pe

    def _get_pgm(self, input : Union[torch.Tensor, bytes], *args, prior : Optional[torch.Tensor] = None, input_shape : Optional[torch.Size] = None, pgm : Any = None, **kwargs) -> Any:
        mask, token_mask, input_mask, output_mask = super()._get_pgm(input, *args, prior=prior, input_shape=input_shape, **kwargs)
        # NOTE: tmp fix for no prior
        if prior is None:
            prior = torch.zeros_like(input).repeat(1, self.num_dist_params, 1, 1)

        if self.predictor_use_pe_only:
            prior = torch.zeros_like(prior) + self._get_pe(prior)
        else:
            if self.predictor_add_pe:
                prior = prior + self._get_pe(prior)
            if self.predictor_use_attention:
                prior_attention = prior.reshape(prior.shape[0], prior.shape[1], -1).movedim(-1, 0)
                prior_attention, _ = self.attention_module(prior_attention, prior_attention, prior_attention, need_weights=False)
                prior = prior_attention.movedim(0, -1).reshape_as(prior)
        topo_group_logits = self.topo_group_predictor(prior)
        if self.predictor_output_weights:
            topo_group_logits, topo_group_weights = torch.split(topo_group_logits, (self.num_topo_groups, 1), dim=1)

        if self.training and self.predictor_regularizer_weight > 0:
            topo_group_probs_mean = torch.softmax(topo_group_logits.movedim(1, -1).reshape(-1, self.num_topo_groups), dim=-1).mean(dim=0)
            topo_group_predictor_regularizer = F.mse_loss(topo_group_probs_mean, torch.zeros_like(topo_group_probs_mean) + 1 / self.num_topo_groups)
            self.update_cache("loss_dict", topo_group_predictor_regularizer=topo_group_predictor_regularizer * self.predictor_regularizer_weight)

        if self.training and self.predictor_soft_sampling:
            if self.predictor_training_skip_sampling:
                topo_group_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
            else:
                topo_group_dist = D.RelaxedOneHotCategorical(self.gs_temp, logits=topo_group_logits.movedim(1, -1))
                topo_group_probs = topo_group_dist.rsample()
        else:
            # TODO: faster hard-sampling?
            if self.predictor_mc_sampling:
                topo_group_dist = D.Categorical(logits=topo_group_logits.movedim(1, -1))
                topo_group_indices = topo_group_dist.sample().long()
                # NOTE: does not work with deterministic backend!
                # topo_group_indices_logprob = topo_group_dist.log_prob(topo_group_indices)
            else:
                topo_group_indices = topo_group_logits.argmax(1)
            # logging
            # self.update_cache("image_dict", input_images=input[:, [2,1,0]]) # BGR to RGB
            topo_group_probs = F.one_hot(topo_group_indices, self.num_topo_groups).type_as(topo_group_logits)
            # self.update_cache("hist_dict", topo_group_counts=topo_group_probs.view(-1, self.num_topo_groups).sum(0))
            # straight through
            if self.training:
                if self.predictor_mc_sampling:
                    topo_group_indices_logprob = topo_group_dist.logits * topo_group_probs
                    self.update_cache("common", pgm_log_prob=topo_group_indices_logprob)
                else:
                    topo_group_soft_probs = torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)
                    topo_group_probs = (topo_group_probs + topo_group_soft_probs) - topo_group_soft_probs.detach()
            else:
                self.update_cache("image_dict", topo_group_indices=(topo_group_indices.unsqueeze(1).float() / (self.num_topo_groups-1)))

        # from torchvision.utils import save_image
        # save_image((topo_group_indices.unsqueeze(1).float() / (self.num_topo_groups-1))[0], "topo_group_indices.png")
        # save_image(torch.softmax(topo_group_logits.movedim(1, -1), dim=-1)[0, ..., 0], "topo_group_probs0.png")

        self.update_cache("moniter_dict", topo_group_probs_mean0=topo_group_probs[..., 0].mean())

        batch_size = topo_group_logits.shape[0]
        spatial_size = np.prod(topo_group_logits.shape[2:])
        topo_group_probs_repeat = topo_group_probs.reshape(batch_size, 1, spatial_size, self.num_topo_groups, 1)
        # NOTE : ********************************************************************
        # calculate p(kernel_topo_group_idx < topo_group_idx) 
        # = sum_{kernel_topo_group_idx, topo_group_idx}(triu(diagonal=1) * p[kernel_topo_group_idx, topo_group_idx])
        # E.g. num_topo_groups=2, p(kernel_topo_group_idx < topo_group_idx) = p(0,1)
        # [[p(0,0), p(0,1)],    *   [[0,    1]  =   [[0,    p(0,1) ],
        # [ p(1,0), p(1,1)]]        [ 0,    0]]     [ 0,    0      ]]
        # NOTE : ********************************************************************
        topo_group_probs_matrices = topo_group_probs.reshape(batch_size, spatial_size, 1, 1, self.num_topo_groups) * topo_group_probs_repeat
        topo_group_mask_prob = torch.triu(topo_group_probs_matrices, diagonal=1).sum((-2, -1)).reshape(batch_size, spatial_size, spatial_size)
        # remove self
        topo_group_mask_prob = topo_group_mask_prob * (1 - torch.eye(spatial_size).type_as(topo_group_mask_prob).unsqueeze(0))
        if self.topo_group_mask_prob_thres is not None:
            topo_group_mask = (topo_group_mask_prob > self.topo_group_mask_prob_thres).type_as(topo_group_mask_prob)
            # straight through
            if self.training:
                topo_group_mask = (topo_group_mask + topo_group_mask_prob) - topo_group_mask_prob.detach()
        else:
            topo_group_mask = topo_group_mask_prob
        self.update_cache("moniter_dict", topo_group_mask_mean=topo_group_mask.mean())

        if self.training and self.topo_group_mask_regularizer_weight > 0:
            topo_group_mask_regularizer = 1 - topo_group_mask_prob.mean()
            self.update_cache("loss_dict", topo_group_mask_regularizer=topo_group_mask_regularizer * self.topo_group_mask_regularizer_weight)

        if self.topo_group_mask_normalize:
            topo_group_mask = topo_group_mask / topo_group_mask.sum(dim=(-2, -1), keepdim=True)

        # if self.predictor_output_weights:
        #     topo_group_weights_unfold = F.unfold(topo_group_weights, (self.kernel_size, self.kernel_size), padding=self.padding).permute(0, 2, 1)\
        #         .reshape(batch_size, *prior.shape[-2:], self.kernel_size, self.kernel_size)
        #     topo_group_mask = topo_group_mask * topo_group_weights_unfold

        # if not self.training:
        #     valid_mask_idx = topo_group_probs.view(topo_group_probs.shape[0], -1, self.num_topo_groups).sum(1)[:, 0].argmax()
        #     self.update_cache("image_dict", **{f"topo_group_mask_{valid_mask_idx}" : make_grid(topo_group_mask[valid_mask_idx].reshape(-1, 1, *topo_group_mask_prob.shape[-2:]), nrow=topo_group_mask_prob.shape[1], padding=1)})
        # annealing

        # from torchvision.utils import save_image
        # save_image(topo_group_mask[0].reshape(-1, 1, 5, 5), "topo_group_mask.png", nrow=topo_group_mask.shape[1])

        if self.detach_pgm_mask:
            topo_group_mask = topo_group_mask.detach()
        
        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    gs_temp=self.gs_temp
                )

        return torch.ones_like(mask).type_as(topo_group_mask) * topo_group_mask, token_mask, input_mask, output_mask
