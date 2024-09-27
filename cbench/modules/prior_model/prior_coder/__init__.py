import math
import struct
import copy
import itertools
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.distributed as distributed
from torch.distributions.utils import clamp_probs, probs_to_logits

from entmax import entmax_bisect

from cbench.nn.base import NNTrainableModule
from cbench.nn.layers import Upsample2DLayer, Downsample2DLayer, MaskedConv2d, MaskedConv3d
from cbench.nn.distributions.kumaraswamy import Kumaraswamy
from cbench.nn.distributions.relaxed import CategoricalRSample, RelaxedOneHotCategorical, AsymptoticRelaxedOneHotCategorical, DoubleRelaxedOneHotCategorical, InvertableGaussianSoftmaxppRelaxedOneHotCategorical
from cbench.modules.entropy_coder.utils import BinaryHeadConstructor
from cbench.modules.entropy_coder.rans import pmf_to_quantized_cdf_serial, pmf_to_quantized_cdf_batched
from cbench.utils.bytes_ops import encode_shape, decode_shape
from cbench.utils.bytes_ops import merge_bytes, split_merged_bytes
from cbench.utils.ar_utils import create_ar_offsets

from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder

from cbench.rans import BufferedRansEncoder, RansEncoder, RansDecoder, pmf_to_quantized_cdf
# TANS_AVAILABLE = True
# try:
#     from cbench.tans import BufferedTansEncoder, TansEncoder, TansDecoder, create_ctable_using_cnt, create_dtable_using_cnt
#     # from cbench.zstd_wrapper import fse_create_ctable_using_cnt, fse_create_dtable_using_cnt, \
#     #     fse_compress_using_ctable, fse_decompress_using_dtable
# except:
#     print("cbench.tans is not compiled! tans coding will not be available!")
#     TANS_AVAILABLE = False

from .base import PriorCoder
from .sqvae_coder import GaussianVectorQuantizer, VmfVectorQuantizer

def gaussian_pyramid_init(num_embeddings, embedding_dim, kl_level_base=2, invert_logprob=False):
    embedding = torch.zeros(num_embeddings, embedding_dim)
    embedding_logprob = torch.zeros(num_embeddings)
    # num_embeddings = (kl_level_base ** num_kl_levels - 1) / (kl_level_base - 1)
    num_kl_levels = math.floor(math.log(num_embeddings * (kl_level_base - 1) + 1) / math.log(kl_level_base))
    cur_idx = 0
    for i in range(num_kl_levels):
        cur_num_embeddings = kl_level_base ** i
        cur_kl = math.log(cur_num_embeddings)
        # kl = -0.5 * (1 + logvar - mean ** 2 - logvar.exp())
        with torch.no_grad():
            trial_step = 0.1
            max_trial = math.exp(1 + cur_kl * 2)
            trial_num_steps = math.floor(max_trial / trial_step)
            logvar_trials = torch.linspace(0, math.exp(1 + cur_kl * 2), steps=trial_num_steps)
            max_logvar_eq_trials = logvar_trials + 1 - logvar_trials.exp() + cur_kl * 2
            max_logvar_eq_trials[max_logvar_eq_trials < 0] = np.inf
            max_logvar = torch.abs(max_logvar_eq_trials).argmin() * trial_step
            min_logvar_eq_trials = -logvar_trials + 1 - (-logvar_trials).exp() + cur_kl * 2
            min_logvar_eq_trials[min_logvar_eq_trials < 0] = np.inf
            min_logvar = -torch.abs(min_logvar_eq_trials).argmin() * trial_step
            if cur_num_embeddings > 1:
                logvars = torch.linspace(min_logvar, max_logvar, cur_num_embeddings // 2)
                means = torch.sqrt(1 + logvars - logvars.exp() + 2 * cur_kl)
                means_negative = -means
                embedding_init_positive = torch.stack([means, logvars], dim=-1)
                embedding_init_negative = torch.stack([means_negative, logvars], dim=-1)
                embedding_init = torch.cat([embedding_init_positive, embedding_init_negative], dim=0).repeat_interleave(embedding_dim // 2, dim=-1)
            else:
                embedding_init = torch.zeros(cur_num_embeddings, embedding_dim)
            embedding[cur_idx:(cur_idx+cur_num_embeddings)] = embedding_init
            embedding_logprob[cur_idx:(cur_idx+cur_num_embeddings)] = cur_kl if invert_logprob else -cur_kl
        cur_idx += cur_num_embeddings
    return embedding, embedding_logprob


class NNPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self):
        super().__init__()
        NNTrainableModule.__init__(self)

    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        raise NotImplementedError()

    def set_vamp_posterior(self, posterior):
        raise NotImplementedError()

    def encode(self, input : torch.Tensor, *args, **kwargs) -> bytes:
        raise NotImplementedError()

    def decode(self, byte_string : bytes, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    """
    A quick function that combines encode and decode. May be overrided to skip latent decoding for faster training.
    """
    def encode_and_decode(self, input : torch.Tensor, *args, **kwargs) -> Tuple[bytes, torch.Tensor]:
        byte_string = self.encode(input, *args, **kwargs)
        return byte_string, self.decode(byte_string, *args, **kwargs)

class HierarchicalNNPriorCoder(NNPriorCoder):
    def __init__(self, 
        encoders : List[nn.Module],
        decoders : List[nn.Module],
        prior_coders : List[NNPriorCoder],
        freeze_encoder=False,
        freeze_decoder=False,
        **kwargs):
        super().__init__()

        self.num_layers = len(prior_coders)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.prior_coders = nn.ModuleList(prior_coders)
        assert len(self.encoders) == self.num_layers-1
        assert len(self.decoders) == self.num_layers-1
        assert len(self.prior_coders) == self.num_layers

        if freeze_encoder:
            for param in self.encoders.parameters():
                param.requires_grad = False
        if freeze_decoder:
            for param in self.decoders.parameters():
                param.requires_grad = False

    def forward(self, input: torch.Tensor, **kwargs):
        latent = input
        latent_enc_nlayer = []
        latent_enc = latent
        for i in range(self.num_layers-1):
            latent_enc = self.encoders[i](latent_enc)
            latent_enc_nlayer.append(latent_enc)

        # prior decoding in inverse direction
        latent_dec = None
        for i in range(self.num_layers-2, -1, -1):
            latent_dec = self.prior_coders[i+1](latent_enc_nlayer[i], prior=latent_dec)
            # save latent_dec as the prior dist for the prev prior_coder
            latent_dec = self.decoders[i](latent_dec) 

        # final output
        latent = self.prior_coders[0](latent, prior=latent_dec)

        # collect all loss_rate
        if self.training:
            loss_rate = 0
            for idx, prior_coder in enumerate(self.prior_coders):
                loss_dict = prior_coder.get_raw_cache("loss_dict")
                if loss_dict.get("loss_rate") is not None:
                    # pop the loss_rate in submodule to avoid multiple losses
                    loss_rate += loss_dict.pop("loss_rate")
                else:
                    print(f"No loss_rate found in self.prior_coders[{idx}]! Check the implementation!")

            self.update_cache("loss_dict",
                loss_rate = loss_rate,
            )

        # collect all prior_entropy
        prior_entropy = 0
        for idx, prior_coder in enumerate(self.prior_coders):
            metric_dict = prior_coder.get_raw_cache("metric_dict")
            if metric_dict.get("prior_entropy") is not None:
                prior_entropy += metric_dict["prior_entropy"]
            else:
                print(f"No prior_entropy found in self.prior_coders[{idx}]! Check the implementation!")

        self.update_cache("metric_dict",
            prior_entropy = prior_entropy,
        )

        return latent

    def encode(self, input: torch.Tensor, **kwargs):
        latent = input
        latent_enc_nlayer = []
        latent_enc = latent
        for i in range(self.num_layers-1):
            latent_enc = self.encoders[i](latent_enc)
            latent_enc_nlayer.append(latent_enc)

        # prior decoding in inverse direction
        latent_byte_strings = []
        latent_dec = None
        for i in range(self.num_layers-2, -1, -1):
            latent_bytes, latent_dec = self.prior_coders[i+1].encode_and_decode(latent_enc_nlayer[i], prior=latent_dec)
            latent_byte_strings.append(latent_bytes)
            # save latent_dec as the prior dist for the prev prior_coder
            latent_dec = self.decoders[i](latent_dec) 

        # final output
        latent_bytes = self.prior_coders[0].encode(latent, prior=latent_dec)
        latent_byte_strings.append(latent_bytes)

        return merge_bytes(latent_byte_strings, num_segments=len(self.prior_coders))

    def decode(self, byte_string: bytes, *args, **kwargs) -> torch.Tensor:
        latent_byte_strings = split_merged_bytes(byte_string, num_segments=len(self.prior_coders))
        
        # prior decoding in inverse direction
        latent_dec = None
        for i in range(self.num_layers-2, -1, -1):
            latent_bytes = latent_byte_strings.pop(0)
            latent_dec = self.prior_coders[i+1].decode(latent_bytes, prior=latent_dec)
            # save latent_dec as the prior dist for the prev prior_coder
            latent_dec = self.decoders[i](latent_dec) 

        # final output
        latent_dec = self.prior_coders[0].decode(latent_byte_strings[-1], prior=latent_dec)
        return latent_dec
    
    # TODO:
    def encode_and_decode(self, input: torch.Tensor, *args, **kwargs) -> Tuple[bytes, torch.Tensor]:
        return super().encode_and_decode(input, *args, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        for prior_coder in self.prior_coders:
            prior_coder.update_state(*args, **kwargs)


class Hierarchical2LayerNNPriorCoder(HierarchicalNNPriorCoder):
    def __init__(self, prior_coders: List[NNPriorCoder] = None, in_channels=256, mid_channels=64, **kwargs):
        encoders = [
            Downsample2DLayer(in_channels, mid_channels),
        ]
        decoders = [
            Upsample2DLayer(mid_channels, in_channels),
        ]

        if prior_coders is None:
            prior_coders = [
                GaussianPriorCoder(in_channels),
                GaussianPriorCoder(mid_channels),
            ]

        super().__init__(encoders, decoders, prior_coders, in_channels=in_channels, **kwargs)


class NNPriorCoderFlatLinearTransform(NNPriorCoder):
    def __init__(self, in_channels=256, 
                 skip_layers_if_equal_channels=False,
                 freeze_input_layer=False,
                 freeze_output_layer=False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.skip_layers_if_equal_channels = skip_layers_if_equal_channels

        self.input_layer = nn.Linear(self.in_channels, self.latent_channels_in)
        if self.skip_layers_if_equal_channels and self.in_channels == self.latent_channels_in:
            self.input_layer = nn.Identity()
        # self.input_layer.weight.data = torch.eye(self.in_channels)
        # self.input_layer.bias.data = torch.zeros(self.in_channels)
        # self.input_layer_mu = nn.Linear(self.in_channels, self.latent_channels)
        # self.input_layer_logvar = nn.Linear(self.in_channels, self.latent_channels)
        self.output_layer = nn.Linear(self.latent_channels_out, self.in_channels)
        # self.output_layer.weight.data = torch.eye(self.in_channels)
        # self.output_layer.bias.data = torch.zeros(self.in_channels)
        if self.skip_layers_if_equal_channels and self.in_channels == self.latent_channels_out:
            self.output_layer = nn.Identity()

        if freeze_input_layer:
            for p in self.input_layer.parameters():
                p.lr_modifier = 0.0

        if freeze_output_layer:
            for p in self.output_layer.parameters():
                p.lr_modifier = 0.0

    @property
    def latent_channels_in(self):
        return self.in_channels

    @property
    def latent_channels_out(self):
        return self.in_channels

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        raise NotImplementedError()

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        raise NotImplementedError()

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs):
        input_shape = input.shape
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        assert(channel_size == self.in_channels)

        if prior is not None:
            assert(prior.shape == input.shape)
            prior = prior.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
            prior = self.input_layer(prior)
        input = input.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
        # input = torch.flatten(input, start_dim=1)
        input = self.input_layer(input) #.reshape(batch_size, -1, self.latent_channels * 2).permute(0, 2, 1)

        output = self._forward_flat(input, input_shape, prior=prior, **kwargs)

        # if prior is not None:
        #     prior = self.output_layer(prior)
        #     prior = prior.reshape(batch_size, -1, channel_size).permute(0, 2, 1).reshape(*input_shape).contiguous()
        #     assert(prior.shape == output.shape)
        output = self.output_layer(output)
        output = output.reshape(batch_size, -1, channel_size).permute(0, 2, 1).reshape(*input_shape).contiguous()

        return output

    def encode(self, input : torch.Tensor, *args, prior : torch.Tensor = None, **kwargs) -> bytes:
        input_shape = input.shape
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        assert(channel_size == self.in_channels)
        spatial_shape = input.shape[2:]

        if prior is not None:
            assert(prior.shape == input.shape)
            prior = prior.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
            prior = self.input_layer(prior)
            prior = prior.reshape(batch_size, -1, self.latent_channels_in).permute(0, 2, 1)\
                .reshape(batch_size, self.latent_channels_in, *spatial_shape).contiguous()
        input = input.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
        # input = torch.flatten(input, start_dim=1)
        input = self.input_layer(input)
        input = input.reshape(batch_size, -1, self.latent_channels_in).permute(0, 2, 1)\
            .reshape(batch_size, self.latent_channels_in, *spatial_shape).contiguous()

        return self._encode_transformed(input, prior=prior, **kwargs)

    def decode(self, byte_string : bytes, *args, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if prior is not None:
            batch_size = prior.shape[0]
            channel_size = prior.shape[1]
            assert(channel_size == self.in_channels)
            spatial_shape = prior.shape[2:]
            prior = prior.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
            prior = self.input_layer(prior)
            prior = prior.reshape(batch_size, -1, self.latent_channels_in).permute(0, 2, 1)\
                .reshape(batch_size, self.latent_channels_in, *spatial_shape).contiguous()
        
        output = self._decode_transformed(byte_string, prior=prior, *args, **kwargs)
        
        batch_size = output.shape[0]
        channel_size = output.shape[1]
        assert(channel_size == self.latent_channels_in)
        spatial_shape = output.shape[2:]
        output = output.reshape(batch_size, channel_size, -1).permute(0, 2, 1)\
            .reshape(-1, channel_size).contiguous()
        output = self.output_layer(output)
        output = output.reshape(batch_size, -1, self.in_channels).permute(0, 2, 1)\
            .reshape(batch_size, self.in_channels, *spatial_shape).contiguous()

        return output


class GaussianPriorCoder(NNPriorCoderFlatLinearTransform):
    def __init__(self, in_channels=256, latent_channels=None,
        coder_type="rans", # current support "rans", "tans"
        coding_sampler="importance",
        coding_seed=0,
        coding_max_index=8,
        coding_max_aux_index=8,
        **kwargs
        ):
        self.latent_channels = (in_channels // 2) if latent_channels is None else latent_channels
        super().__init__(in_channels=in_channels)

        self.coder_type = coder_type
        self.coding_sampler = coding_sampler
        self.coding_seed = coding_seed
        self.coding_max_index = coding_max_index
        self.coding_max_aux_index = coding_max_aux_index

    @property
    def latent_channels_in(self):
        return self.latent_channels * 2

    @property
    def latent_channels_out(self):
        return self.latent_channels

    # TODO: update with new ans interface!
    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        # Relative Entropy Coding (for bits-back, see entropy_coder implementation)
        mean, logvar = input.chunk(2, dim=1)
        # mean = self.input_layer_mu(input)
        # logvar = self.input_layer_logvar(input)
        std = torch.exp(0.5 * logvar)

        dist = distributions.Normal(mean, std)

        kl_divergence = -0.5 * (1 + logvar - mean ** 2 - logvar.exp())

        num_samples = torch.exp(kl_divergence).long()
        rng = torch.Generator().manual_seed(self.coding_seed)
        # TODO: how to deal with outer prior?
        samples = torch.empty(num_samples.sum()).normal_(0, 1, generator=rng)
        if self.coding_sampler == "importance":
            sample_log_importance = dist.log_prob(samples) + (samples ** 2) / 2
            best_indices = sample_log_importance.argmax()
        else:
            raise NotImplementedError(f"coding_sampler {self.coding_sampler} not implemented!")

        # divide indices
        aux_indices = (torch.log(best_indices.float()) / math.log(self.coding_max_index)).floor().long()
        assert (aux_indices < self.coding_max_aux_index).all()
        remain_indices = best_indices - self.coding_max_index ** aux_indices
        
        if self.coder_type == "rans":
            rans_encoder = BufferedRansEncoder()
            indexes = np.zeros(input.shape)
            
            # prepare for coding
            data = data.astype(np.int32).reshape(-1)
            indexes = indexes.astype(np.int32).reshape(-1)
            cdfs = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_index))])
            cdfs_aux = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_aux_index))])
            cdf_sizes = np.ones(1) * self.coding_max_index
            cdf_sizes_aux = np.ones(1) * self.coding_max_index
            offsets = np.zeros(1)
            with self.profiler.start_time_profile("time_rans_encoder"):
                rans_encoder.encode_with_indexes_np(
                    remain_indices, indexes, cdfs, cdf_sizes, offsets
                )
                rans_encoder.encode_with_indexes_np(
                    aux_indices, indexes, cdfs_aux, cdf_sizes_aux, offsets
                )
                data_bytes = rans_encoder.flush()
        
        bytes_shape = encode_shape((input.shape))
        data_bytes = b''.join([bytes_shape, data_bytes])
        
        return data_bytes
            

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        input_shape, byte_ptr = decode_shape(byte_string)

        if self.coder_type == "rans":
            rans_decoder = RansDecoder()
            indexes = np.zeros(input_shape)
            
            # prepare for coding
            indexes = indexes.astype(np.int32).reshape(-1)
            cdfs = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_index))])
            cdfs_aux = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_aux_index))])
            cdf_sizes = np.ones(1) * self.coding_max_index
            cdf_sizes_aux = np.ones(1) * self.coding_max_index
            offsets = np.zeros(1)
            with self.profiler.start_time_profile("time_rans_encoder"):
                remain_indices = rans_decoder.decode_with_indexes_np(
                    byte_string[byte_ptr:], indexes, cdfs, cdf_sizes, offsets
                )
                byte_ptr += 0 # TODO:
                aux_indices = rans_decoder.decode_with_indexes_np(
                    byte_string[byte_ptr:], indexes, cdfs_aux, cdf_sizes_aux, offsets
                )

        # generate samples
        prior_dist = self.prior_distribution(prior=prior)
        torch.manual_seed(self.coding_seed)
        num_samples_max = self.coding_max_index ** aux_indices + remain_indices
        samples = prior_dist.rsample(num_samples_max)

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        mean, logvar = input.chunk(2, dim=1)
        # mean = self.input_layer_mu(input)
        # logvar = self.input_layer_logvar(input)
        std = torch.exp(0.5 * logvar)

        dist = distributions.Normal(mean, std)
        output = dist.rsample()
        # normal = torch.normal(torch.zeros(mean.shape), torch.ones(mean.shape)).type_as(mean)
        # normal = torch.randn_like(std)
        # output = mean + normal * std

        if prior is not None:
            prior_mean, prior_logvar = prior.chunk(2, dim=1)
            prior_std = torch.exp(0.5 * prior_logvar)
            prior_dist = distributions.Normal(prior_mean, prior_std)
            KLD = distributions.kl_divergence(dist, prior_dist).sum()
        else:
            KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
        
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        self.update_cache("metric_dict",
            prior_entropy = KLD / input_shape[0], # normalize by batch size
        )

        return output


class CategoricalGaussianPriorCoder(NNPriorCoderFlatLinearTransform):
    def __init__(self, in_channels=256, categorical_dim=128, latent_channels=8,
        sample_gmm_prior=True,
        gs_temp=0.5, gs_temp_anneal=False,
        freeze_logvar=False, var_scale=1.0, var_scale_anneal=False,
        **kwargs):
        self.categorical_dim = categorical_dim
        self.latent_channels = (in_channels // 2) if latent_channels is None else latent_channels
        
        self.sample_gmm_prior = sample_gmm_prior
        super().__init__(in_channels=in_channels, **kwargs)

        # prior params
        self.gaussian_priors_mean = nn.Parameter(torch.zeros(self.latent_channels, self.categorical_dim))
        nn.init.uniform_(self.gaussian_priors_mean, -1, 1)
        self.gaussian_priors_logvar = nn.Parameter(torch.zeros(self.latent_channels, self.categorical_dim))
        nn.init.constant_(self.gaussian_priors_logvar, -math.log(self.categorical_dim))
        if freeze_logvar:
            self.gaussian_priors_logvar.requires_grad = False
        
        self.cat_prior_logits = nn.Parameter(torch.zeros(self.latent_channels, self.categorical_dim))
        nn.init.constant_(self.cat_prior_logits, -math.log(self.categorical_dim))

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        self.var_scale_anneal = var_scale_anneal
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.var_scale = var_scale

    @property
    def latent_channels_in(self):
        return self.latent_channels * (2 + self.categorical_dim)

    @property
    def latent_channels_out(self):
        return self.latent_channels

    # def encode(self, input, *args, **kwargs) -> bytes:
    #     raise NotImplementedError("Variational prior are not directly encodable!")

    # def decode(self, byte_string, *args, **kwargs) -> Any:
    #     raise NotImplementedError("Variational prior are not directly decodable!")

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        gaussian_params, categorical_params = input.view(-1, self.latent_channels, 2 + self.categorical_dim).split([2, self.categorical_dim], dim=-1)
        mean, logvar = gaussian_params.chunk(2, dim=-1)
        # mean = self.input_layer_mu(input)
        # logvar = self.input_layer_logvar(input)
        std = torch.exp(0.5 * logvar)

        dist = distributions.Normal(mean.squeeze(-1), std.squeeze(-1))
        output = dist.rsample()
        # normal = torch.normal(torch.zeros(mean.shape), torch.ones(mean.shape)).type_as(mean)
        # normal = torch.randn_like(std)
        # output = mean + normal * std

        # KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
        cat_dist = distributions.RelaxedOneHotCategorical(self.gs_temp, logits=categorical_params)
        cat_sample = cat_dist.rsample() # batch * self.latent_channels * self.categorical_dim

        if self.sample_gmm_prior:
            gmm_weight_dist = distributions.Categorical(probs=cat_sample)
        else:
            gmm_weight_dist = distributions.Categorical(probs=cat_dist.probs)

        if prior is not None:
            prior_gaussian_params, prior_categorical_params = input.view(-1, self.latent_channels, 2 + self.categorical_dim).split([2, self.categorical_dim], dim=-1)
            prior_mean, prior_logvar = prior_gaussian_params.chunk(2, dim=-1)
            prior_std = torch.exp(0.5 * prior_logvar)
            prior_dists = distributions.Normal(prior_mean, prior_std)
            gmm_prior = distributions.MixtureSameFamily(gmm_weight_dist, prior_dists)
            cat_prior_logits = torch.log_softmax(prior_categorical_params, dim=-1)
        else:
            prior_dists = distributions.Normal(
                self.gaussian_priors_mean.unsqueeze(0).repeat(input.shape[0], 1, 1), 
                torch.exp(0.5 * self.gaussian_priors_logvar).unsqueeze(0).repeat(input.shape[0], 1, 1) * self.var_scale
            )
            gmm_prior = distributions.MixtureSameFamily(gmm_weight_dist, prior_dists)
            cat_prior_logits = torch.log_softmax(self.cat_prior_logits, dim=-1).unsqueeze(0)
        
        gaussian_kl = (dist.log_prob(output) - gmm_prior.log_prob(output)).sum()
        
        cat_kl = cat_dist.probs * (cat_dist.logits - cat_prior_logits)
        cat_kl[(cat_dist.probs == 0).expand_as(cat_kl)] = 0
        cat_kl = cat_kl.sum()

        prior_entropy = gaussian_kl + cat_kl
        if self.training:
            self.update_cache("loss_dict",
                loss_rate= prior_entropy / input_shape[0], # normalize by batch size
            )
        self.update_cache("metric_dict",
            prior_entropy = prior_entropy / input_shape[0],
            gaussian_kl = gaussian_kl / input_shape[0],
            cat_kl = cat_kl / input_shape[0],
        )

        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    gs_temp=self.gs_temp
                )

        if self.var_scale_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    var_scale=self.var_scale
                )

        return output


class GaussianEmbeddingCategoricalPriorCoder(NNPriorCoderFlatLinearTransform):
    def __init__(self, in_channels=256, categorical_dim=128, embedding_dim=1, latent_channels=8,
        output_sample_from_embedding=False,
        gs_temp=0.5, gs_temp_anneal=False,
        **kwargs):
        self.categorical_dim = categorical_dim
        self.embedding_dim = embedding_dim
        self.latent_channels = (in_channels // 2) if latent_channels is None else latent_channels
        
        self.output_sample_from_embedding = output_sample_from_embedding
        super().__init__(in_channels=in_channels, **kwargs)

        # prior params
        self.embedding_means = nn.Parameter(torch.zeros(self.latent_channels, self.categorical_dim, self.embedding_dim))
        nn.init.uniform_(self.embedding_means, -1, 1)
        self.embedding_logvars = nn.Parameter(torch.zeros(self.latent_channels, self.categorical_dim, self.embedding_dim))
        # nn.init.constant_(self.embedding_logvars, -math.log(self.categorical_dim))
        
        self.cat_prior_logits = nn.Parameter(torch.zeros(self.latent_channels, self.categorical_dim))
        nn.init.constant_(self.cat_prior_logits, -math.log(self.categorical_dim))

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

    @property
    def latent_channels_in(self):
        return self.latent_channels * (self.categorical_dim + self.categorical_dim * self.embedding_dim)

    @property
    def latent_channels_out(self):
        return self.latent_channels * self.embedding_dim

    # def encode(self, input, *args, **kwargs) -> bytes:
    #     raise NotImplementedError("Variational prior are not directly encodable!")

    # def decode(self, byte_string, *args, **kwargs) -> Any:
    #     raise NotImplementedError("Variational prior are not directly decodable!")

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        # TODO: prior
        input_batch_size = input.shape[0]
        categorical_params, embedding_params = input.view(
            input_batch_size, 
            self.latent_channels, 
            self.categorical_dim + self.categorical_dim * self.embedding_dim
            ).split([self.categorical_dim, self.categorical_dim * self.embedding_dim], dim=-1)

        cat_logits = torch.log_softmax(categorical_params, dim=1)
        cat_dist = distributions.Categorical(logits=cat_logits / self.gs_temp)

        cat_prior_logits = torch.log_softmax(self.cat_prior_logits, dim=-1).unsqueeze(0)
        cat_kl = cat_dist.probs * (cat_dist.logits - cat_prior_logits)
        cat_kl[(cat_dist.probs == 0).expand_as(cat_kl)] = 0
        cat_kl = cat_kl.sum()

        embedding_scales = torch.exp(0.5 * self.embedding_logvars)
        embedding_dist = distributions.LowRankMultivariateNormal(self.embedding_means, torch.zeros_like(embedding_scales).unsqueeze(-1), embedding_scales)
        embedding_kl = -0.5 * (1 + self.embedding_logvars - self.embedding_means ** 2 - self.embedding_logvars.exp()).sum() * input_batch_size

        embedding_params = embedding_params.reshape(input_batch_size, self.latent_channels, self.categorical_dim, self.embedding_dim)
        quantize_logprob = embedding_dist.log_prob(embedding_params)
        quantize_kl = -cat_dist.probs * quantize_logprob
        quantize_kl[(cat_dist.probs == 0).expand_as(quantize_kl)] = 0
        quantize_kl = quantize_kl.sum()

        if self.output_sample_from_embedding:
            embedding_sample = embedding_dist.rsample(embedding_params.shape[:1])
        else:
            embedding_sample = embedding_params
        output = (cat_dist.probs.unsqueeze(-1) * embedding_sample).sum(-2)
        output = output.reshape(input_batch_size, self.latent_channels_out)

        total_kl = cat_kl + embedding_kl + quantize_kl
        prior_entropy = -(cat_dist.probs * cat_prior_logits).sum()
        if self.training:
            self.update_cache("loss_dict",
                loss_rate= total_kl / input_shape[0], # normalize by batch size
            )
        self.update_cache("metric_dict",
            prior_entropy = prior_entropy / input_shape[0],
            cat_kl = cat_kl / input_shape[0],
            embedding_kl = embedding_kl / input_shape[0],
            quantize_kl = quantize_kl / input_shape[0],
        )

        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    gs_temp=self.gs_temp
                )

        return output


class DistributionPriorCoder(NNPriorCoderFlatLinearTransform):
    def __init__(self, in_channels=256, latent_channels=None, 
        # prior_trainable=False,
        train_em_update=False,
        coder_type="rans", # current support "rans", "tans"
        coder_freq_precision=16,
        coding_sampler="importance",
        coding_seed=0,
        coding_max_samples=256,
        coding_max_index=8,
        coding_max_aux_index=8,
        fixed_input_shape=None,
        **kwargs):
        self.latent_channels = (in_channels // self.num_posterior_params) if latent_channels is None else latent_channels
        super().__init__(in_channels=in_channels, **kwargs)
        
        # TODO: prior should be defined here?
        # self.prior_trainable = prior_trainable
        # prior_params = torch.zeros(self.latent_channels, self.num_prior_params)
        # if prior_trainable:
        #     self.prior_params = nn.Parameter(prior_params)
        # else:
        #     self.register_buffer("prior_params", prior_params, persistent=False)

        self.train_em_update = train_em_update
        if train_em_update:
            self.em_state = True

        self.coder_type = coder_type
        self.coder_freq_precision = coder_freq_precision
        self.coding_sampler = coding_sampler
        self.coding_seed = coding_seed
        self.coding_max_samples = coding_max_samples
        self.coding_max_index = coding_max_index
        self.coding_max_aux_index = coding_max_aux_index
        self.fixed_input_shape = fixed_input_shape

    @property
    def latent_channels_in(self):
        return self.latent_channels * self.num_posterior_params

    @property
    def latent_channels_out(self):
        return self.latent_channels * self.num_sample_params
    
    @property
    def num_posterior_params(self):
        return 1

    @property
    def num_prior_params(self):
        # Usually prior is the same type as posterior
        return self.num_posterior_params

    @property
    def num_sample_params(self):
        return 1

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        raise NotImplementedError("Variational prior are not directly encodable!")

        # Relative Entropy Coding (for bits-back, see entropy_coder implementation)
        posterior_dist = self.posterior_distribution(input)
        prior_dist = self.prior_distribution(prior=prior)

        # TODO: some implementations use kl_divergence as a loss calculator. This may not be appropriate!
        kl_divergence = self.kl_divergence(prior_dist, posterior_dist)
        if kl_divergence.shape != input.shape: 
            raise ValueError("kl_divergence Implementation should not reduce kl divergence result!")

        num_samples = self.coding_max_samples # torch.exp(kl_divergence).long().max().item()
        torch.manual_seed(self.coding_seed)
        # TODO: how to efficiently generate samples in batch with different num_samples?
        samples = prior_dist.rsample(num_samples).unsqueeze(1)
        if self.coding_sampler == "importance":
            sample_log_importance = posterior_dist.log_prob(samples) - prior_dist.log_prob(samples)
            best_indices = sample_log_importance.argmax()
        else:
            raise NotImplementedError(f"coding_sampler {self.coding_sampler} not implemented!")

        # divide indices
        # aux_indices = (torch.log(best_indices.float()) / math.log(self.coding_max_index)).floor().long()
        # assert (aux_indices < self.coding_max_aux_index).all()
        # remain_indices = best_indices - self.coding_max_index ** aux_indices
        
        if self.coder_type == "rans":
            rans_encoder = BufferedRansEncoder()
            indexes = np.zeros(input.shape)
            
            # prepare for coding
            data = best_indices.numpy().astype(np.int32).reshape(-1)
            indexes = indexes.astype(np.int32).reshape(-1)
            cdfs = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_samples))])
            # cdfs_aux = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_aux_index))])
            cdf_sizes = np.ones(1) * self.coding_max_index
            # cdf_sizes_aux = np.ones(1) * self.coding_max_index
            offsets = np.zeros(1)
            with self.profiler.start_time_profile("time_rans_encoder"):
                rans_encoder.encode_with_indexes_np(
                    data, best_indices, indexes, cdfs, cdf_sizes, offsets
                )
                # rans_encoder.encode_with_indexes_np(
                #     aux_indices, indexes, cdfs_aux, cdf_sizes_aux, offsets
                # )
                data_bytes = rans_encoder.flush()
        
        bytes_shape = encode_shape((input.shape))
        data_bytes = b''.join([bytes_shape, data_bytes])
        
        return data_bytes
            

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Variational prior are not directly decodable!")
        input_shape, byte_ptr = decode_shape(byte_string)

        if self.coder_type == "rans":
            rans_decoder = RansDecoder()
            indexes = np.zeros(input_shape)
            
            # prepare for coding
            indexes = indexes.astype(np.int32).reshape(-1)
            cdfs = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_index))])
            cdfs_aux = np.array([pmf_to_quantized_cdf(np.ones(self.coding_max_aux_index))])
            cdf_sizes = np.ones(1) * self.coding_max_index
            cdf_sizes_aux = np.ones(1) * self.coding_max_index
            offsets = np.zeros(1)
            with self.profiler.start_time_profile("time_rans_encoder"):
                remain_indices = rans_decoder.decode_with_indexes_np(
                    byte_string[byte_ptr:], indexes, cdfs, cdf_sizes, offsets
                )
                byte_ptr += 0 # TODO:
                aux_indices = rans_decoder.decode_with_indexes_np(
                    byte_string[byte_ptr:], indexes, cdfs_aux, cdf_sizes_aux, offsets
                )

        # generate samples
        prior_dist = self.prior_distribution(prior=prior)
        torch.manual_seed(self.coding_seed)
        num_samples_max = self.coding_max_index ** aux_indices + remain_indices
        samples = prior_dist.rsample(num_samples_max)

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    def prior_distribution(self, prior=None, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, posterior_samples=None, **kwargs):
        """
        Default KL Divergence is calculated by stochastic sampling, rather than closed-form.
        Overwrite this to implement a closed-form kl divergence.

        Args:
            prior_dist (distributions.Distribution)
            posterior_dist (distributions.Distribution)

        Returns:
            _type_: kl_divergence post || prior
        """        
        if posterior_samples is None:
            posterior_samples = posterior_dist.rsample()
        logp = prior_dist.log_prob(posterior_samples)
        logq = posterior_dist.log_prob(posterior_samples)
        return logq - logp

    def sample_from_posterior(self, posterior_dist : distributions.Distribution, **kwargs):
        if self.training:
            output = posterior_dist.rsample()
        else:
            # TODO: use REC samples?
            output = posterior_dist.sample()
        return output
    
    def postprocess_samples(self, samples):
        return samples.reshape(-1, self.latent_channels_out)

    def set_custom_state(self, state: str = None):
        if state == "EM-E":
            self.em_state = True
        elif state == "EM-M":
            self.em_state = False
        # else:
        #     raise ValueError()
        return super().set_custom_state(state)

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        if self.train_em_update:
            if self.em_state:
                posterior_dist = self.posterior_distribution(input)
                prior_dist = self.prior_distribution(prior=prior.detach())
            else:
                posterior_dist = self.posterior_distribution(input.detach())
                prior_dist = self.prior_distribution(prior=prior)
        else:
            posterior_dist = self.posterior_distribution(input)
            prior_dist = self.prior_distribution(prior=prior)

        samples = self.sample_from_posterior(posterior_dist)

        # TODO: check add posterior_samples?
        KLD = torch.sum(self.kl_divergence(prior_dist, posterior_dist, input_shape=input_shape, posterior_samples=samples))
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        # if implementation has not provide prior_entropy, use kl as prior_entropy instead
        if not "prior_entropy" in self.get_raw_cache("metric_dict"):
            self.update_cache("metric_dict",
                prior_entropy = KLD / input_shape[0], # normalize by batch size
            )

        return self.postprocess_samples(samples)


class AutoregressivePriorDistributionPriorCoder(DistributionPriorCoder):

    def _autoregressive_prior(self, prior : torch.Tensor = None, input_shape : torch.Size = None, posterior_samples : torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def _autoregressive_posterior(self, input : torch.Tensor = None, **kwargs) -> torch.Tensor:
        return input

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        posterior_ar = self._autoregressive_posterior(input)
        if self.train_em_update and self.em_state == False:
            posterior_ar = posterior_ar.detach()

        posterior_dist = self.posterior_distribution(posterior_ar)
        samples = self.sample_from_posterior(posterior_dist)

        prior_ar = self._autoregressive_prior(prior=prior, input_shape=input_shape, posterior_samples=samples)

        if self.train_em_update and self.em_state == True:
            prior_ar = posterior_ar.detach()
        prior_dist = self.prior_distribution(prior=prior_ar)

        # TODO: check add posterior_samples?
        KLD = torch.sum(self.kl_divergence(prior_dist, posterior_dist, input_shape=input_shape, posterior_samples=samples))
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        # if implementation has not provide prior_entropy, use kl as prior_entropy instead
        if not "prior_entropy" in self.get_raw_cache("metric_dict"):
            self.update_cache("metric_dict",
                prior_entropy = KLD / input_shape[0], # normalize by batch size
            )

        return self.postprocess_samples(samples)


class AutoregressivePriorImplDistributionPriorCoder(AutoregressivePriorDistributionPriorCoder):
    def __init__(self, in_channels=256, latent_channels=None, 
                 prior_trainable=False,
                 # ar prior
                 use_autoregressive_prior=False, 
                 ar_method="finitestate", ar_input_detach=False, # ar_input_sample=True, ar_input_straight_through=False,
                 ar_window_size=1, ar_offsets=None,
                 ar_fs_method="table", # deprecated
                 ## for table based fsar
                #  ar_prior_decomp_method="sum", ar_prior_decomp_dim=None,
                 ## for MLP based fsar
                 ar_mlp_per_channel=False, ar_mlp_bottleneck_expansion=2,
                 # ar posterior
                 use_autoregressive_posterior=False,
                 posterior_ar_window_size=1,
                 **kwargs):
        super().__init__(in_channels, latent_channels, **kwargs)
        self.prior_trainable = prior_trainable
        prior_params = torch.zeros(self.latent_channels, self.num_prior_params)
        if prior_trainable:
            self.prior_params = nn.Parameter(prior_params)
        else:
            self.register_buffer("prior_params", prior_params, persistent=False)

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_method = ar_method
        self.ar_input_detach = ar_input_detach
        # self.ar_input_sample = ar_input_sample
        # self.ar_input_straight_through = ar_input_straight_through
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        # self.ar_fs_method = ar_fs_method
        # self.ar_prior_decomp_method = ar_prior_decomp_method
        # self.ar_prior_decomp_dim = ar_prior_decomp_dim
        self.ar_mlp_per_channel = ar_mlp_per_channel
        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_channels - 1
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = [(-offset,) for offset in range(1, self.ar_window_size+1)]
        else:
            self.ar_window_size = len(ar_offsets)

        if use_autoregressive_prior:
            ar_input_channels = self.num_sample_params
            self.ar_input_channels = ar_input_channels
            if self.ar_method == "finitestate":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(ar_input_channels * self.ar_window_size, int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels)),
                                nn.LeakyReLU(),
                                nn.Linear(int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels), int(ar_mlp_bottleneck_expansion * self.num_prior_params)),
                                nn.LeakyReLU(),
                                nn.Linear(int(ar_mlp_bottleneck_expansion * self.num_prior_params), self.num_prior_params),
                            )
                            for _ in range(self.latent_channels)
                        ]
                    )
                else:
                    self.fsar_mlp = nn.Sequential(
                        nn.Linear(ar_input_channels * self.ar_window_size, int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels)),
                        nn.LeakyReLU(),
                        nn.Linear(int(ar_mlp_bottleneck_expansion * self.ar_window_size * ar_input_channels), int(ar_mlp_bottleneck_expansion * self.num_prior_params)),
                        nn.LeakyReLU(),
                        nn.Linear(int(ar_mlp_bottleneck_expansion * self.num_prior_params), self.num_prior_params),
                    )

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(ar_input_channels, self.num_prior_params, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(ar_input_channels, self.num_prior_params, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_channels, self.num_prior_params * self.latent_channels, 5, padding=2, mask_type="Checkerboard")

            if ar_model is not None:
                self.ar_model = nn.Sequential(
                    ar_model,
                    # TODO: append a trans module
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 6 // 3, ar_input_channels * self.latent_dim * 5 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 5 // 3, ar_input_channels * self.latent_dim * 4 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 4 // 3, ar_input_channels * self.latent_dim * 3 // 3, 1),
                )

        self.use_autoregressive_posterior = use_autoregressive_posterior
        self.posterior_ar_window_size = posterior_ar_window_size

    def _default_sample(self, samples : torch.Tensor = None) -> torch.Tensor:
        return torch.zeros_like(samples)
    
    def _finite_state_to_samples(self, states : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError()

    def _autoregressive_prior(self, prior : torch.Tensor = None, input_shape : torch.Size = None, posterior_samples : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if prior is None:
            prior = self.prior_params.unsqueeze(0)
        # TODO: process prior parameter if exists!
        if self.use_autoregressive_prior:
            # if input_shape is None:
            #     input_shape = posterior_dist.logits.shape[:-1]
            assert input_shape is not None
            assert posterior_samples is not None
            batch_size = input_shape[0]
            spatial_shape = input_shape[2:]
            posterior_samples_shape = posterior_samples.shape # N*spatial_dim*C*num_sample_params
            if self.ar_input_detach:
                posterior_samples = posterior_samples.detach()
            if self.ar_method == "finitestate":
                # find samples for ar
                # reshape as input format (N*spatial_dim*C*num_sample_params -> N*C*spatial_dim*num_sample_params)
                posterior_samples_reshape = posterior_samples.reshape(batch_size, *spatial_shape, self.latent_channels, self.num_sample_params).movedim(-2, 1)
                # merge prior logits
                autoregressive_samples = []
                for ar_offset in self.ar_offsets:
                    default_samples = self._default_sample(posterior_samples_reshape)
                    ar_samples = posterior_samples_reshape
                    for data_dim, data_offset in enumerate(ar_offset):
                        if data_offset >= 0: continue
                        batched_data_dim = data_dim + 1
                        assert batched_data_dim != ar_samples.ndim - 1 # ar could not include ar_input_channels
                        ar_samples = torch.cat((
                            default_samples.narrow(batched_data_dim, 0, -data_offset),
                            ar_samples.narrow(batched_data_dim, 0, posterior_samples_reshape.shape[batched_data_dim]+data_offset)
                        ), dim=batched_data_dim)
                    autoregressive_samples.append(ar_samples)
                # [batch_size, self.latent_channels, *spatial_shape, self.ar_window_size*self.ar_input_channels]
                autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                if self.ar_mlp_per_channel:
                    autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                        .reshape(posterior_samples_shape[0], self.latent_channels, self.ar_window_size*self.ar_input_channels)
                    ar_logits = torch.stack([mlp(sample.squeeze(1)) for mlp, sample in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=1))], dim=1)
                else:
                    autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(-1, self.ar_window_size*self.ar_input_channels)
                    ar_logits = self.fsar_mlp(autoregressive_samples_flat)
                    # merge ar logits and prior logits
            else:
                assert len(spatial_shape) == 2
                posterior_samples_reshape = posterior_samples.reshape(batch_size, *spatial_shape, self.latent_channels * self.num_sample_params).movedim(-1, 1)
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        posterior_samples_reshape = posterior_samples_reshape.reshape(batch_size, self.latent_channels, self.num_sample_params, *spatial_shape)\
                            .permute(0, 2, 1, 3, 4)
                    ar_logits_reshape = self.ar_model(posterior_samples_reshape)
                    if self.ar_method.startswith("maskconv3d"):
                        ar_logits_reshape = ar_logits_reshape.permute(0, 2, 1, 3, 4)\
                            .reshape(batch_size, self.latent_channels, *spatial_shape)
                elif self.ar_method.startswith("checkerboard"):
                    ar_logits_reshape = self.ar_model(posterior_samples_reshape)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # ar_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # ar_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # TODO: default prior params?
                    ar_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = self.prior_params.reshape(1, -1, 1, 1).repeat(ar_logits_reshape.shape[0], 1, ar_logits_reshape.shape[2]//2, ar_logits_reshape.shape[3]//2)
                    ar_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = self.prior_params.reshape(1, -1, 1, 1).repeat(ar_logits_reshape.shape[0], 1, ar_logits_reshape.shape[2]//2, ar_logits_reshape.shape[3]//2)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                ar_logits = ar_logits_reshape.movedim(1, -1).reshape(posterior_samples_shape[0], self.latent_channels, self.num_prior_params)
            prior = ar_logits + prior

        return prior


class CategoricalAutoregressivePriorDistributionPriorCoder(AutoregressivePriorImplDistributionPriorCoder):
    def __init__(self, in_channels=256, latent_channels=8, categorical_dim=128, 
                 # KL
                 use_sample_kl=False, sample_kl_use_log_mixture=False, kl_prior_detach_posterior=False,
                 # sampling
                 use_gs_st_sample=False,
                 # category reduction
                 cat_reduce=False, cat_reduce_method="softminus", cat_reduce_channel_same=False,
                 cat_reduce_logit_thres=10.0, cat_reduce_logit_thres_low=None, 
                 cat_reduce_logit_bias=0.0,
                 cat_reduce_logit_init_range=0.1, cat_reduce_regularizer=0.0,
                 cat_reduce_entmax_alpha_trainable=False, cat_reduce_entmax_alpha=1.5, cat_reduce_entmax_alpha_min=1.0, cat_reduce_entmax_alpha_max=2.0,
                 # anneal
                 gs_temp=0.5, gs_temp_anneal=False,
                 relax_temp=0.5, relax_temp_anneal=False,
                 entropy_temp=1.0, entropy_temp_anneal=False, entropy_temp_threshold=0.0,
                 cat_reduce_temp=1.0, cat_reduce_temp_anneal=False,
                 **kwargs):
        self.categorical_dim = categorical_dim
        self.use_sample_kl = use_sample_kl
        self.sample_kl_use_log_mixture = sample_kl_use_log_mixture
        self.kl_prior_detach_posterior = kl_prior_detach_posterior
        self.use_gs_st_sample = use_gs_st_sample
        super().__init__(in_channels, latent_channels, **kwargs)

        self.cat_reduce = cat_reduce
        self.cat_reduce_method = cat_reduce_method
        self.cat_reduce_logit_thres = cat_reduce_logit_thres
        self.cat_reduce_logit_thres_low = cat_reduce_logit_thres\
            if cat_reduce_logit_thres_low is None else cat_reduce_logit_thres_low
        self.cat_reduce_logit_bias = cat_reduce_logit_bias
        self.cat_reduce_channel_same = cat_reduce_channel_same
        self.cat_reduce_regularizer = cat_reduce_regularizer
        self.cat_reduce_entmax_alpha_trainable = cat_reduce_entmax_alpha_trainable
        self.cat_reduce_entmax_alpha_min = cat_reduce_entmax_alpha_min
        self.cat_reduce_entmax_alpha_max = cat_reduce_entmax_alpha_max
        if self.cat_reduce_entmax_alpha_trainable:
            # inverse sigmoid
            self.cat_reduce_entmax_alpha = nn.Parameter(
                -torch.tensor([(1 / max(cat_reduce_entmax_alpha-self.cat_reduce_entmax_alpha_min, 1e-7) ) - 1]).log()
            )
        else:
            self.cat_reduce_entmax_alpha = cat_reduce_entmax_alpha
        if self.cat_reduce:
            cat_reduce_channel_dim = 1 if cat_reduce_channel_same else self.latent_channels
            cat_reduce_logprob = None
            if self.cat_reduce_method == "entmax":
                cat_reduce_logprob = torch.zeros(cat_reduce_channel_dim, categorical_dim) # - self.cat_reduce_logit_thres
            else:
                raise NotImplementedError(f"Unknown cat_reduce_method {cat_reduce_method}")
            if cat_reduce_logprob is not None:
                self.cat_reduce_logprob = nn.Parameter(cat_reduce_logprob)
                nn.init.uniform_(self.cat_reduce_logprob, -cat_reduce_logit_init_range, cat_reduce_logit_init_range) # add a small variation

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.register_buffer("gs_temp", torch.tensor(gs_temp), persistent=False)

        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.register_buffer("relax_temp", torch.tensor(relax_temp), persistent=False)

        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            # self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.register_buffer("entropy_temp", torch.tensor(entropy_temp), persistent=False)
            # self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_threshold = entropy_temp_threshold

        self.cat_reduce_temp_anneal = cat_reduce_temp_anneal
        if cat_reduce_temp_anneal:
            self.cat_reduce_temp = nn.Parameter(torch.tensor(cat_reduce_temp), requires_grad=False)
        else:
            self.cat_reduce_temp = cat_reduce_temp

        # TODO: tmp fix! remove this after implementing fsar for rans!
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            self.coder_type = "tans"
            self.coder_freq_precision = 10

    @property
    def num_posterior_params(self):
        return self.categorical_dim

    @property
    def num_sample_params(self):
        return self.categorical_dim

    def _get_entmax_probs(self):
        if self.cat_reduce_temp_anneal:
            alpha = self.cat_reduce_entmax_alpha_max - self.cat_reduce_temp * (self.cat_reduce_entmax_alpha_max - self.cat_reduce_entmax_alpha_min)
        else:
            if self.cat_reduce_entmax_alpha_trainable:
                alpha = self.cat_reduce_entmax_alpha_min + torch.sigmoid(self.cat_reduce_entmax_alpha) * \
                    (self.cat_reduce_entmax_alpha_max - self.cat_reduce_entmax_alpha_min)
                self.update_cache("metric_dict", 
                    cat_reduce_entmax_alpha=alpha,
                )
            else:
                alpha = self.cat_reduce_entmax_alpha

        if alpha <= 1.0:
            entmax_probs = torch.softmax(self.cat_reduce_logprob, dim=-1)
        else:
            entmax_probs = entmax_bisect(self.cat_reduce_logprob, alpha=alpha, dim=-1)

        return entmax_probs
    
    def _cat_reduce_entmax_probs(self, input):

        entmax_probs = self._get_entmax_probs()
        
        cat_reduce_percentage = (entmax_probs==0).sum() / self.cat_reduce_logprob.numel()
        self.update_cache("metric_dict", 
            cat_reduce_percentage=cat_reduce_percentage,
        )
        
        input_probs = torch.softmax(input, dim=-1) * entmax_probs.unsqueeze(0)
        return input_probs / (input_probs.sum(dim=-1, keepdim=True) + 1e-7) # renormalize

    def _cat_reduce_logits(self, logits):
        if self.cat_reduce_method == "entmax":
            probs = self._cat_reduce_entmax_probs(logits) # .unsqueeze(0)
            return (probs + 1e-9).log() / (probs + 1e-9).sum(dim=-1, keepdim=True)

    def _finite_state_to_samples(self, states: torch.LongTensor, add_default_samples=False) -> torch.Tensor:
        if add_default_samples:
            samples = F.one_hot((states-1).clamp(min=0), self.categorical_dim)
            return torch.where(states > 0 | states <= self.categorical_dim, samples, self._default_sample(samples))
        return F.one_hot(states, self.categorical_dim)

    def _latent_to_finite_state(self, latent: torch.Tensor) -> torch.LongTensor:
        if self.cat_reduce:
            latent = latent.index_select(-1, self._reduce_mask)
        return torch.argmax(latent, dim=-1)

    def prior_distribution(self, prior=None, **kwargs) -> distributions.Categorical:
        if prior is None:
            prior = self.prior_params.unsqueeze(0)
        prior = prior.view(-1, self.latent_channels, self.categorical_dim)
        if self.cat_reduce:
            prior = self._cat_reduce_logits(prior)
        return distributions.Categorical(logits=prior)

    def posterior_distribution(self, latent, **kwargs) -> distributions.RelaxedOneHotCategorical:
        latent_logits = latent.view(-1, self.latent_channels, self.categorical_dim)
        latent_logits = latent_logits / self.relax_temp
        if self.cat_reduce:
            latent_logits = self._cat_reduce_logits(latent_logits)
        return distributions.RelaxedOneHotCategorical(self.gs_temp, logits=latent_logits)
    
    def sample_from_posterior(self, posterior_dist: distributions.RelaxedOneHotCategorical, **kwargs):
        samples = super().sample_from_posterior(posterior_dist, **kwargs)
        if self.use_gs_st_sample:
            one_hot_samples = F.one_hot(samples.argmax(-1), samples.shape[-1])\
                .type_as(samples)
            samples = one_hot_samples + samples - samples.detach()
        return samples

    def kl_divergence(self, 
                      prior_dist: distributions.Categorical, 
                      posterior_dist: distributions.RelaxedOneHotCategorical, 
                      input_shape: torch.Size = None, posterior_samples=None, **kwargs):
        if self.use_sample_kl and posterior_samples is not None:                
            if self.sample_kl_use_log_mixture:
                posterior_entropy = (posterior_samples * posterior_dist.probs).sum(-1).clamp(min=1e-6).log()
                if self.kl_prior_detach_posterior:
                    prior_entropy = (posterior_samples.detach() * prior_dist.probs).sum(-1).clamp(min=1e-6).log()
                else:
                    prior_entropy = (posterior_samples * prior_dist.probs).sum(-1).clamp(min=1e-6).log()
            else:
                posterior_entropy = posterior_samples * posterior_dist.logits # posterior_samples.clamp(min=1e-6).log()
                posterior_entropy[posterior_samples == 0] = 0 # prevent nan
                if self.kl_prior_detach_posterior:
                    prior_entropy = posterior_samples.detach() * prior_dist.logits
                else:
                    prior_entropy = posterior_samples * prior_dist.logits
        else:
            posterior_entropy = posterior_dist.probs * posterior_dist.logits
            posterior_entropy[posterior_dist.probs == 0] = 0 # prevent nan
            if self.kl_prior_detach_posterior:
                prior_entropy = posterior_dist.probs.detach() * prior_dist.logits
            else:
                prior_entropy = posterior_dist.probs * prior_dist.logits

        entropy_temp = self.entropy_temp if self.entropy_temp >= self.entropy_temp_threshold else 0.0
        kld = posterior_entropy * entropy_temp - prior_entropy

        # moniter entropy gap for annealing
        if self.training:
            self.update_cache("moniter_dict", 
                qp_entropy_gap=(posterior_entropy.sum() / prior_entropy.sum()),
            )
            self.update_cache("moniter_dict", 
                posterior_entropy=posterior_entropy.sum(),
            )
            self.update_cache("moniter_dict", 
                prior_self_entropy=(prior_dist.probs * prior_dist.logits).sum(),
            )
            one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            self.update_cache("moniter_dict", 
                prior_one_hot_entropy=-(one_hot_samples * prior_dist.logits).sum(),
            )

        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    gs_temp=self.gs_temp
                )
        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    relax_temp=self.relax_temp
                )
        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    entropy_temp=self.entropy_temp
                )

        return kld

    def _normalize_prior_logits(self, prior_logits):
        if self.cat_reduce:
            if self.cat_reduce_method == "entmax":
                # cat_reduce_logprob = self.cat_reduce_logprob
                # if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
                #     cat_reduce_logprob = self.cat_reduce_logprob.unsqueeze(1)
                prior_logits = self._cat_reduce_logits(prior_logits) # .unsqueeze(0)
        prior_logits = torch.log_softmax(prior_logits, dim=-1)
        return prior_logits

    def _get_ar_params(self, indexes) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.use_autoregressive_prior:
            # indices_shape = indexes.shape
            # indices = torch.zeros(*indexes.shape).reshape(indexes.shape[0], indexes.shape[1], -1)
            # channel_indices = torch.arange(indexes.shape[1], dtype=torch.int32).reshape(1, indices.shape[1], 1).expand_as(indices)
            # ar_indices = channel_indices.reshape(*indices_shape).contiguous().numpy()
            ar_indices = np.zeros_like(indexes)
            ar_offsets = create_ar_offsets(indexes.shape, self.ar_offsets)
            return ar_indices, ar_offsets
        else:
            return None, None

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        spatial_shape = input.shape[2:]
        assert channel_size == self.latent_channels * self.num_posterior_params
        
        # posterior_dist = self.posterior_distribution(input.movedim(1, -1).reshape(-1, self.latent_channels, self.categorical_dim))
        # prior_dist = self.prior_distribution(prior=prior)

        # samples = self.sample_from_posterior(posterior_dist)

        # KLD = self.kl_divergence(prior_dist, posterior_dist, input_shape=(batch_size, self.latent_channels, *spatial_shape))

        input = input.movedim(1, -1).view(batch_size, *spatial_shape, self.latent_channels, self.num_sample_params)
        
        # non-finite autoregressive
        data_bytes = b''
        if self.use_autoregressive_prior and self.ar_method != "finitestate":
            samples = self._latent_to_finite_state(input)
            ar_input = self._finite_state_to_samples(samples).type_as(input)\
                .reshape(batch_size, *spatial_shape, self.latent_channels*self.num_sample_params).movedim(-1, 1)
            if self.ar_method.startswith("maskconv"):
                if self.ar_method.startswith("maskconv3d"):
                    ar_input = ar_input.reshape(batch_size, self.latent_channels, self.num_sample_params, *spatial_shape).movedim(2, 1)
                prior_logits_reshape = self.ar_model(ar_input)
                # move batched dimensions to last for correct decoding
                if self.ar_method.startswith("maskconv3d"):
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                    samples = samples.movedim(0, -2)
                else:
                    prior_logits_reshape = prior_logits_reshape.reshape(batch_size, self.latent_channels, self.num_prior_params, *spatial_shape)
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1).movedim(0, -1)
                    samples = samples.movedim(0, -2)
                # move categorical dim
                prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                
                rans_encoder = RansEncoder()

                data = samples.detach().cpu().numpy().astype(np.int32)
                prior_probs = torch.softmax(prior_logits_reshape, dim=-1)
                cdfs = pmf_to_quantized_cdf_batched(prior_probs.reshape(-1, prior_probs.shape[-1]))
                cdfs = cdfs.detach().cpu().numpy().astype(np.int32)

                data = data.reshape(-1)
                indexes = np.arange(len(data), dtype=np.int32)
                cdf_lengths = np.array([len(cdf) for cdf in cdfs])
                offsets = np.zeros(len(indexes)) # [0] * len(indexes)

                with self.profiler.start_time_profile("time_rans_encoder"):
                    data_bytes = rans_encoder.encode_with_indexes_np(
                        data, indexes,
                        cdfs, cdf_lengths, offsets
                    )

            elif self.ar_method.startswith("checkerboard"):
                samples = samples.movedim(-1, 1) # move latent channels after batch
                prior_logits_reshape = self.ar_model(ar_input)
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                # input_base = torch.cat([
                #     ar_input[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                #     ar_input[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                # ], dim=-1)
                # input_ar = torch.cat([
                #     ar_input[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                #     ar_input[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                # ], dim=-1)
                prior_logits_ar = torch.cat([
                    prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                prior_logits_ar = prior_logits_ar.reshape(batch_size, self.latent_channels, self.num_prior_params, *prior_logits_ar.shape[-2:]).movedim(2, -1)

                samples_base = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                ], dim=-1)
                data_base = samples_base.detach().cpu().numpy()
                indexes_base = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_size, 1, np.prod(spatial_shape) // 2).reshape_as(samples_base).numpy()
                samples_ar = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                data_ar = samples_ar.detach().cpu().numpy()
                
                # prepare for coding (base)
                data_base = data_base.astype(np.int32).reshape(-1)
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                # prepare for coding (ar)
                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                data_ar = data_ar.reshape(-1)
                indexes_ar = np.arange(len(data_ar), dtype=np.int32)
                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                rans_encoder = BufferedRansEncoder()
                with self.profiler.start_time_profile("time_rans_encoder"):
                    rans_encoder.encode_with_indexes_np(
                        data_base, indexes_base,
                        cdfs_base, cdf_sizes_base, offsets_base
                    )
                    rans_encoder.encode_with_indexes_np(
                        data_ar, indexes_ar,
                        cdfs_ar, cdf_sizes_ar, offsets_ar
                    )
                    data_bytes = rans_encoder.flush()
            else:
                pass

        
        if len(data_bytes) == 0:

            # TODO: use iterative autoregressive for overwhelmed states
            if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
                raise NotImplementedError("Overwhelmed states!")

            # if self.cat_reduce:
            #     input = input.index_select(2, self._reduce_mask).view(batch_size, self.latent_channels, -1, *spatial_shape)
                # if self.cat_reduce_method == "softminus":
                #     raise NotImplementedError()
                # elif self.cat_reduce_method == "sigmoid":
                #     if not self.cat_reduce_channel_same:
                #         # TODO: different transformation for different channels
                #         raise NotImplementedError()
                #     else:
                #         reduce_mask = (self.cat_reduce_logprob[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                #         input = input.index_select(2, reduce_mask).view(batch_size, self.latent_channels, -1, *spatial_shape)
                # elif self.cat_reduce_method == "entmax":
                #     reduce_mask = (entmax_bisect(self.cat_reduce_logprob) > 0)

            samples = self._latent_to_finite_state(input).movedim(-1, 1) # move latent channels back
            data = samples.contiguous().detach().cpu().numpy().astype(np.int32)
            # self._samples_cache = samples
            indexes = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
                .repeat(batch_size, 1, np.prod(spatial_shape)).reshape_as(samples)\
                .contiguous().numpy().astype(np.int32)

            ar_indexes, ar_offsets = self._get_ar_params(indexes)
            
            data_bytes = self._encoder.encode_with_indexes(
                data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets,
            )

        if len(data_bytes) == 0:
            return b''

        # store sample shape in header
        byte_strings = []
        if self.fixed_input_shape is not None:
            assert batch_size == self.fixed_input_shape[0]
            assert spatial_shape == self.fixed_input_shape[1:]
        else:
            byte_head = [struct.pack("B", len(spatial_shape)+1)]
            byte_head.append(struct.pack("<H", batch_size))
            for dim in spatial_shape:
                byte_head.append(struct.pack("<H", dim))
            byte_strings.extend(byte_head)
        byte_strings.append(data_bytes)
        return b''.join(byte_strings)

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        # if len(byte_string) == 0:
        #     return torch.zeros(1, self.latent_channels*self.categorical_dim, 8, 8, device=self.device)

        # decode shape from header
        if self.fixed_input_shape is not None:
            byte_ptr = 0
            batch_dim = self.fixed_input_shape[0]
            spatial_shape = self.fixed_input_shape[1:]
            spatial_dim = np.prod(spatial_shape)
        else:
            num_shape_dims = struct.unpack("B", byte_string[:1])[0]
            flat_shape = []
            byte_ptr = 1
            for _ in range(num_shape_dims):
                flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
                byte_ptr += 2
            batch_dim = flat_shape[0]
            spatial_shape = flat_shape[1:]
            spatial_dim = np.prod(spatial_shape)

        if self.use_autoregressive_prior and self.ar_method != "finitestate":
            if self.ar_method.startswith("maskconv"):
                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                samples = torch.zeros(batch_dim, *spatial_shape, self.latent_channels, dtype=torch.long, device=self.device)

                assert len(spatial_shape) == 2
                if self.ar_method.startswith("maskconv3d"):
                    c, h, w = (self.latent_channels, *spatial_shape)
                    for c_idx in range(c):
                        for h_idx in range(h):
                            for w_idx in range(w):
                                ar_input = self._finite_state_to_samples(samples).float().movedim(-1, 1)
                                prior_logits_ar = self.ar_model(ar_input).movedim(1, -1)[:, h_idx, w_idx, c_idx]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device)
                                samples[:, h_idx, w_idx, c_idx] = samples_ar
                else:
                    h, w = spatial_shape
                    error_flag = False
                    for h_idx in range(h):
                        for w_idx in range(w):
                                ar_input = self._finite_state_to_samples(samples).float().reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params).movedim(-1, 1)
                                prior_logits_ar = self.ar_model(ar_input).reshape(batch_dim, self.latent_channels, self.num_prior_params, *spatial_shape).movedim(2, -1)[:, :, h_idx, w_idx, :]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device).reshape(-1, self.latent_channels)
                                if samples_ar.max() >= self.categorical_dim or samples_ar.min() < 0:
                                    # NOTE: early exit to avoid gpu indicing error!
                                    print("Decode error detected! The decompressed data may be corrupted!")
                                    error_flag = True
                                    break
                                samples[:, h_idx, w_idx, :] = samples_ar
                        if error_flag:
                            break

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = self._finite_state_to_samples(samples.movedim(1, -1)).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
                    .movedim(-1, 1)

                return samples

            elif self.ar_method.startswith("checkerboard"):
                assert len(spatial_shape) == 2
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=self.device)
                checkerboard_index_h_00, checkerboard_index_w_00 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_0)
                checkerboard_index_h_11, checkerboard_index_w_11 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_1)
                checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)

                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                indexes_base = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_dim, 1, spatial_dim // 2).reshape(batch_dim, self.latent_channels, spatial_shape[0] // 2, spatial_shape[1])\
                    .numpy()

                # prepare for coding
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                samples = torch.zeros(batch_dim, self.latent_channels, *spatial_shape, dtype=torch.long, device=self.device)
                with self.profiler.start_time_profile("time_rans_decoder"):
                    samples_base = rans_decoder.decode_stream_np(
                        indexes_base, cdfs_base, cdf_sizes_base, offsets_base
                    )
                    samples_base = torch.as_tensor(samples_base, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.latent_channels, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_01, checkerboard_index_w_01] = samples_base[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_10, checkerboard_index_w_10] = samples_base[..., (spatial_shape[-1]//2):]
                    ar_input = self._finite_state_to_samples(samples.movedim(1, -1)).float()
                    ar_input = ar_input.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
                        .movedim(-1, 1)
                    
                    prior_logits_reshape = self.ar_model(ar_input)
                    prior_logits_ar = torch.cat([
                        prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                        prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                    ], dim=-1)
                    prior_logits_ar = prior_logits_ar.reshape(batch_dim, self.latent_channels, self.num_prior_params, *prior_logits_ar.shape[-2:]).movedim(2, -1)
                    
                    # prepare for coding (ar)
                    # NOTE: coding may be unstable on GPU!
                    prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                    cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                    cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                    data_length = samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0].numel() + samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1].numel()
                    indexes_ar = np.arange(data_length, dtype=np.int32)
                    cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                    offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                    samples_ar = rans_decoder.decode_stream_np(
                        indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                    )
                    samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.latent_channels, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_00, checkerboard_index_w_00] = samples_ar[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_11, checkerboard_index_w_11] = samples_ar[..., (spatial_shape[-1]//2):]

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = self._finite_state_to_samples(samples.movedim(1, -1)).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
                    .movedim(-1, 1)

                return samples

            else:
                pass

        # TODO: use iterative autoregressive for overwhelmed states
        if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
            raise NotImplementedError("Overwhelmed states!")

        indexes = torch.arange(self.latent_channels).unsqueeze(0).unsqueeze(-1)\
            .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.latent_channels, *spatial_shape)\
            .contiguous().numpy().astype(np.int32)

        ar_indexes, ar_offsets = self._get_ar_params(indexes)
        
        samples = self._decoder.decode_with_indexes(
            byte_string[byte_ptr:], indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets,
        )

        samples = torch.as_tensor(samples, dtype=torch.long, device=self.device)\
            .reshape(batch_dim, self.latent_channels, *spatial_shape)
        # assert (samples == self._samples_cache).all()

        # cat_reduce transform
        if self.cat_reduce:
            samples = self._reduce_mask[samples]
            # if self.cat_reduce_method == "softminus":
            #     raise NotImplementedError()
            # elif self.cat_reduce_method == "sigmoid":
            #     if not self.cat_reduce_channel_same:
            #         # TODO: different transformation for different channels
            #         raise NotImplementedError()
            #     else:
            #         reduce_mask = (self.cat_reduce_logprob[0] > 0).nonzero(as_tuple=False).squeeze(-1)
            #         # samples_full = torch.zeros(*samples.shape, self.categorical_dim).type_as(self.cat_reduce_logprob)
            #         samples = reduce_mask[samples]

        # merge categorical dim back to latent dim
        # samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
        samples = self._finite_state_to_samples(samples.movedim(1, -1)).float()
        samples = samples.reshape(batch_dim, *spatial_shape, self.latent_channels*self.num_sample_params)\
            .movedim(-1, 1)

        return samples

    def update_state(self, *args, **kwargs) -> None:
        with torch.no_grad():
            if self.prior_trainable:
                prior_logits = self._normalize_prior_logits(self.prior_params)#.unsqueeze(-1)
            else:
                prior_logits = (torch.ones(self.latent_channels, self.categorical_dim) / self.categorical_dim).log().to(device=self.device)
            
            categorical_dim = self.categorical_dim # cat reduce moved after fsar
            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                # TODO: this is a hard limit! may could be improved!
                if len(self.ar_offsets) > 2:
                    return
                else:
                    lookup_table_shape = [self.latent_channels] + [categorical_dim+1] * len(self.ar_offsets) + [categorical_dim]
                    ar_idx_all = list(itertools.product(range(self.categorical_dim+1), repeat=self.ar_window_size))
                    ar_idx_all = torch.tensor(ar_idx_all, device=self.device).reshape(-1, 1).repeat(1, self.latent_channels)
                    ar_input_all = self._finite_state_to_samples(ar_idx_all, add_default_samples=True).type_as(prior_logits)\
                        .reshape(-1, self.ar_window_size, self.latent_channels, self.num_sample_params).movedim(1, -2)\
                        .reshape(-1, self.latent_channels, self.ar_window_size*self.num_sample_params).movedim(1, 0)
                    if self.ar_mlp_per_channel:
                        ar_logits_reshape = torch.stack([mlp(ar_input) for (mlp, ar_input) in zip(self.fsar_mlps_per_channel, ar_input_all)], dim=0)
                    else:
                        ar_logits_reshape = self.fsar_mlp(ar_input_all)
                    prior_logits = prior_logits.unsqueeze(-2) + ar_logits_reshape
                    prior_logits = self._normalize_prior_logits(prior_logits)
                    prior_logits = prior_logits.reshape(*lookup_table_shape)

            prior_pmfs = None

            if self.cat_reduce:
                if self.cat_reduce_method == "softminus":
                    # raise NotImplementedError()
                    pass
                elif self.cat_reduce_method == "sigmoid":
                    if not self.cat_reduce_channel_same:
                        # TODO: different transformation for different channels
                        # raise NotImplementedError()
                        pass
                    else:
                        reduce_mask = (self.cat_reduce_logprob[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                        categorical_dim = reduce_mask.shape[0]
                        prior_logits_reduced = prior_logits
                        if self.use_autoregressive_prior and self.ar_method == "finitestate":
                            # if self.ar_prior_decomp_dim is None:
                            #     prior_logits = prior_logits.reshape(self.latent_channels, self.ar_window_size, self.categorical_dim, self.categorical_dim)
                            # else:
                            #     prior_logits = prior_logits.reshape(self.latent_channels, self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                            # prior_logits = prior_logits.index_select(-2, reduce_mask)
                            for i in range(-1, -len(self.ar_offsets)-1, -1):
                                prior_logits_reduced = prior_logits_reduced.index_select(i, reduce_mask)
                        else:
                            prior_logits = self.cat_reduce_logit_thres * torch.sigmoid(prior_logits).index_select(-1, reduce_mask)
                        prior_logits = torch.log_softmax(prior_logits_reduced, dim=-1)
                        prior_pmfs = prior_logits.exp()
                        # self._reduce_mask = reduce_mask
                        self.register_buffer("_reduce_mask", reduce_mask, persistent=False)
                elif self.cat_reduce_method == "entmax":
                    if not self.cat_reduce_channel_same:
                        # TODO: different transformation for different channels
                        # raise NotImplementedError()
                        pass
                    else:
                        reduce_mask = (self._get_entmax_probs()[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                        categorical_dim = reduce_mask.shape[0]
                        prior_logits_reduced = prior_logits
                        if self.use_autoregressive_prior and self.ar_method == "finitestate":
                            # if self.ar_prior_decomp_dim is None:
                            #     prior_logits = prior_logits.reshape(self.latent_channels, self.ar_window_size, self.categorical_dim, self.categorical_dim)
                            # else:
                            #     prior_logits = prior_logits.reshape(self.latent_channels, self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                            # prior_logits = prior_logits.index_select(-2, reduce_mask)
                            prior_logits_reduced = prior_logits_reduced.index_select(-1, reduce_mask)
                            for i in range(-2, -len(self.ar_offsets)-2, -1):
                                reduce_mask_ar = torch.cat([torch.zeros(1).type_as(reduce_mask), reduce_mask+1], dim=0)
                                prior_logits_reduced = prior_logits_reduced.index_select(i, reduce_mask_ar)
                        else:
                            prior_logits = self.cat_reduce_logit_thres * torch.sigmoid(prior_logits).index_select(-1, reduce_mask)
                        prior_logits = torch.log_softmax(prior_logits_reduced, dim=-1)
                        # self._reduce_mask = reduce_mask
                        self.register_buffer("_reduce_mask", reduce_mask, persistent=False)

            if prior_pmfs is None:
                prior_pmfs = prior_logits.exp()

            # TODO: customize freq precision
            if self.coder_type == "rans" or self.coder_type == "rans64":
                # for compability
                self._prior_cdfs = pmf_to_quantized_cdf_serial(prior_pmfs.reshape(-1, categorical_dim))
                self._encoder = Rans64Encoder(freq_precision=self.coder_freq_precision)
                self._decoder = Rans64Decoder(freq_precision=self.coder_freq_precision)
            elif self.coder_type == "tans":
                self._encoder = TansEncoder(table_log=self.coder_freq_precision, max_symbol_value=categorical_dim-1)
                self._decoder = TansDecoder(table_log=self.coder_freq_precision, max_symbol_value=categorical_dim-1)
            else:
                raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")

            prior_cnt = (prior_pmfs * (1<<self.coder_freq_precision)).clamp_min(1).reshape(-1, categorical_dim)
            prior_cnt = prior_cnt.detach().cpu().numpy().astype(np.int32)
            num_symbols = np.zeros(len(prior_cnt), dtype=np.int32) + categorical_dim
            offsets = np.zeros(len(prior_cnt), dtype=np.int32)

            self._encoder.init_params(prior_cnt, num_symbols, offsets)
            self._decoder.init_params(prior_cnt, num_symbols, offsets)

            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                ar_indexes = np.arange(len(prior_cnt), dtype=np.int32).reshape(1, *prior_pmfs.shape[:-1])

                self._encoder.init_ar_params(ar_indexes, [self.ar_offsets])
                self._decoder.init_ar_params(ar_indexes, [self.ar_offsets])


class StochasticVQAutoregressivePriorDistributionPriorCoder(CategoricalAutoregressivePriorDistributionPriorCoder):
    def __init__(self, in_channels=256, latent_channels=8, categorical_dim=128, embedding_dim=32,
                 channels_share_codebook=False,
                 force_hardmax=False, use_st_hardmax=False, hardmax_st_use_logits=False, use_st_prior_embedding=False,
                 force_st=False, st_weight=1.0, use_st_below_entropy_threshold=False,
                 fix_embedding=False, ema_update_embedding=False, ema_decay=0.999, ema_epsilon=1e-5,
                 train_em_mstep_samples=1,
                 initialization_scale=1.0, # NOTE: for compability
                 embedding_variance=1.0, embedding_variance_per_channel=False,
                 embedding_variance_trainable=True, embedding_variance_lr_modifier=1.0,
                 fix_prior_variance=False, prior_embedding_variance=1.0,
                 posterior_detach_embedding=False, posterior_use_sampled_distance=False,
                 distance_method="gaussian", 
                 cont_loss_weight=1.0, seperate_update_cont_loss=False, vq_loss_weight=1.0, commit_loss_weight=1.0,
                 output_use_variance=False,
                 one_hot_initialization=False, embedding_init_method="uniform",
                 relax_temp=1.0, # fix default value
                 var_scale=1.0, var_scale_anneal=False,
                 **kwargs):
        self.embedding_dim = embedding_dim
        self.channels_share_codebook = channels_share_codebook
        self.force_hardmax = force_hardmax
        self.use_st_hardmax = use_st_hardmax
        self.hardmax_st_use_logits = hardmax_st_use_logits
        self.use_st_prior_embedding = use_st_prior_embedding
        self.force_st = force_st
        self.st_weight = st_weight
        self.use_st_below_entropy_threshold = use_st_below_entropy_threshold
        
        self.posterior_detach_embedding = posterior_detach_embedding
        self.posterior_use_sampled_distance = posterior_use_sampled_distance
        self.distance_method = distance_method
        self.cont_loss_weight = cont_loss_weight
        self.seperate_update_cont_loss = seperate_update_cont_loss
        self.vq_loss_weight = vq_loss_weight
        self.commit_loss_weight = commit_loss_weight
        self.output_use_variance = output_use_variance

        self.train_em_mstep_samples = train_em_mstep_samples

        super().__init__(in_channels, latent_channels, categorical_dim, relax_temp=relax_temp, **kwargs)
        
        embedding = torch.zeros(1 if self.channels_share_codebook else latent_channels, categorical_dim, embedding_dim)
        if one_hot_initialization:
            self.embedding_dim = categorical_dim # force embedding dim equal to categorical_dim
            embedding = torch.eye(categorical_dim).unsqueeze(0).repeat(latent_channels, 1, 1)
        else:
            if embedding_init_method == "normal":
                nn.init.normal_(embedding, 0, initialization_scale)
            else:
                nn.init.uniform_(embedding, -initialization_scale, initialization_scale)
        
        if initialization_scale is None:
            initialization_scale = 1/categorical_dim
        embedding.uniform_(-initialization_scale, initialization_scale)

        self.fix_embedding = fix_embedding
        self.ema_update_embedding = ema_update_embedding
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        
        if self.fix_embedding:
            self.register_buffer("embedding", embedding)
        else:
            if self.ema_update_embedding:
                self.register_buffer("embedding", embedding)
                self.register_buffer("ema_count", torch.zeros(latent_channels, categorical_dim))
                self.register_buffer("ema_weight", self.embedding.clone())
            else:
                self.embedding = nn.Parameter(embedding)

        self.embedding_variance_trainable = embedding_variance_trainable
        self.embedding_variance_per_channel = embedding_variance_per_channel
        self.fix_prior_variance = fix_prior_variance
        if fix_prior_variance:
            self.prior_embedding_variance = prior_embedding_variance
        if embedding_variance > 0:
            if self.embedding_variance_per_channel:
                embedding_variance = torch.ones(self.latent_channels) * np.log(embedding_variance) # exponential reparameterization
            else:
                embedding_variance = torch.ones(1) * np.log(embedding_variance) # exponential reparameterization
            if embedding_variance_trainable:
                self.embedding_variance = nn.Parameter(embedding_variance)
                self.embedding_variance.lr_modifier = embedding_variance_lr_modifier
            else:
                self.register_buffer("embedding_variance", embedding_variance)
        else:
            self.embedding_variance = 1e-6

        self.var_scale_anneal = var_scale_anneal
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.var_scale = var_scale

        if self.use_autoregressive_posterior:
            self.posterior_ar_model = nn.Sequential(
                        nn.Linear(2 * self.embedding_dim, 3 * self.embedding_dim),
                        nn.LeakyReLU(),
                        nn.Linear(3 * self.embedding_dim, 2 * self.embedding_dim),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            )

    def _get_embedding_variance(self):
        return self.embedding_variance.exp() * self.var_scale # exponential reparameterization

    def _calc_distances(self, codewords, codebook, embedding_variance=None):
        if embedding_variance is None:
            embedding_variance = self._get_embedding_variance()
        if self.embedding_variance_per_channel:
            embedding_variance = embedding_variance.view(1, -1, 1, 1)
        if self.distance_method == "gaussian":
            # distance = ((codewords - codebook) ** 2).sum(-1) / embedding_variance
            distance = torch.sum(codewords**2, dim=-1, keepdim=True) \
                + torch.sum(codebook**2, dim=-1).unsqueeze(-2) \
                - 2 * torch.matmul(codewords, codebook.transpose(-2, -1))
            distance = distance / embedding_variance / 2
        elif self.distance_method == "vmf":
            # embedding_variance = self.embedding_variance.exp()
            codewords = F.normalize(codewords, p=2.0, dim=-1)
            codebook = F.normalize(codebook, p=2.0, dim=-1)
            distance = torch.matmul(codewords, codebook.transpose(-2, -1)) / embedding_variance
        else:
            raise NotImplementedError(f"Unknown distance method {self.distance_method}")

        if self.training:
            if isinstance(embedding_variance, torch.Tensor):
                self.update_cache("moniter_dict",
                    embedding_variance_mean = embedding_variance.mean(),
                )

        return distance

    def _calc_dist_logits(self, codewords, codebook, embedding_variance=None):
        distance = self._calc_distances(codewords, codebook, embedding_variance=embedding_variance)
        return torch.log_softmax(-distance, dim=-1)

    def _calc_cont_loss(self, latent, samples):
        embedding_variance = self._get_embedding_variance()
        if self.embedding_variance_per_channel:
            latent = latent.view(-1, self.latent_channels, self.embedding_dim)
            samples = samples.view(-1, self.latent_channels, self.embedding_dim)
            embedding_variance = embedding_variance.view(1, -1, 1)
        if self.distance_method == "gaussian":
            # embedding_variance = 2 * (embedding_variance ** 2)
            if self.fix_prior_variance:
                # NOTE: this is basically kl divergence of normal, the entropy part is annealed with entropy_temp
                if self.embedding_variance_trainable:
                    return torch.sum(
                        ((latent - samples) ** 2 + embedding_variance) / self.prior_embedding_variance - 
                        torch.log(embedding_variance / self.prior_embedding_variance) * self.entropy_temp, 
                        dim=-1
                    ) / 2
                else:
                    return torch.sum((latent - samples) ** 2 / self.prior_embedding_variance / 2, dim=-1)
            else:
                return torch.sum((latent - samples) ** 2 / embedding_variance / 2, dim=-1)
        elif self.distance_method == "vmf":
            # TODO: if self.fix_prior_variance:
            # embedding_variance = self.embedding_variance.exp()
            latent = F.normalize(latent, p=2.0, dim=-1)
            samples = F.normalize(samples, p=2.0, dim=-1)
            return torch.sum(latent * (latent - samples) / embedding_variance, dim=-1)
        else:
            raise NotImplementedError(f"Unknown distance method {self.distance_method}")

    def _ema_update_embedding(self, input, samples):
            with torch.no_grad():
                input = input.view(-1, self.latent_channels, self.embedding_dim)
                samples = samples.view(-1, self.latent_channels, self.categorical_dim)
                total_count = samples.sum(dim=0)
                dw = torch.bmm(samples.permute(1, 2, 0), input.permute(1, 0, 2))
                if distributed.is_initialized():
                    distributed.all_reduce(total_count)
                    distributed.all_reduce(dw)
                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * total_count
                n = torch.sum(self.ema_count, dim=-1, keepdim=True)
                self.ema_count = (self.ema_count + self.ema_epsilon) / (n + self.categorical_dim * self.ema_epsilon) * n
                self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * dw
                self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

    @property
    def num_sample_params(self):
        return self.embedding_dim
    
    @property
    def num_posterior_params(self):
        return self.embedding_dim

    @property
    def num_prior_params(self):
        return self.categorical_dim

    def _finite_state_to_samples(self, states: torch.LongTensor, add_default_samples=False) -> torch.Tensor:
        # TODO: use indexing to optimize! 
        def _index_samples(states):
            # NOTE: this is just a try to accelerate! not sure which is faster.
            # or is batched index select possible?
            # if self.latent_channels > 10:
            #     samples = F.one_hot(states, self.categorical_dim).type_as(self.embedding)
            #     samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
            # else:
            #     samples = torch.stack([v.index_select(0, s.reshape(-1)) for s, v in zip(states.split(1, dim=-1), self.embedding)], dim=-1)\
            #         .reshape(*states.shape, self.embedding_dim)
            indexes = states.movedim(-1, 0).reshape(self.latent_channels, -1, 1).expand(-1, -1, self.embedding_dim)
            samples = torch.gather(self.embedding, 1, indexes).movedim(0,1)\
                .reshape(*states.shape, self.embedding_dim)
            # samples = torch.stack([v.index_select(0, s.reshape(-1)) for s, v in zip(states.split(1, dim=-1), self.embedding)], dim=-1)\
            #     .reshape(*states.shape, self.embedding_dim)
            return samples

        if add_default_samples:
            # samples = F.one_hot((states-1).clamp(min=0), self.categorical_dim).type_as(self.embedding)
            # samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
            samples = _index_samples((states-1).clamp(min=0))
            return torch.where(torch.logical_and(states > 0, states <= self.categorical_dim).unsqueeze(-1), samples, self._default_sample(samples))
        else:
            # samples = F.one_hot(states, self.categorical_dim).type_as(self.embedding)
            # samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
            samples = _index_samples(states)
            return samples

    def _latent_to_finite_state(self, latent: torch.Tensor) -> torch.LongTensor:
        # NOTE: different from forward process, this implementation only use 3 dims for distance calculation which is faster!
        embedding = self.embedding# .unsqueeze(0)
        latent_shape = latent.shape
        latent = latent.reshape(-1, self.latent_channels, self.embedding_dim).movedim(0, 1)
        # logits = self._calc_dist_logits(latent, embedding).squeeze(-2)
        distance = self._calc_distances(latent, embedding).movedim(0, 1)#.squeeze(-2)
        return super()._latent_to_finite_state(-distance).reshape(*latent_shape[:-1]) # last dim (embedding_dim) is removed

    def posterior_distribution(self, latent, **kwargs) -> distributions.RelaxedOneHotCategorical:
        latent = latent.reshape(-1, self.latent_channels, 1, self.embedding_dim)
        embedding = self.embedding.unsqueeze(0)
        if self.posterior_detach_embedding:
            embedding = embedding.detach()
        logits = self._calc_dist_logits(latent, embedding).squeeze(-2)
        return super().posterior_distribution(logits, **kwargs)
        # if self.cat_reduce:
        #     logits = self._cat_reduce_logits(logits)
        # return distributions.RelaxedOneHotCategorical(self.gs_temp, logits=logits)

    def sample_from_posterior(self, posterior_dist: distributions.Distribution, **kwargs):
        if self.training and not self.force_hardmax:
            if self.train_em_update and self.em_state == False:
                # draw samples for M step
                dist_cat = distributions.Categorical(logits=posterior_dist.logits)
                samples = dist_cat.sample((self.train_em_mstep_samples, ))
                samples = F.one_hot(samples, posterior_dist.logits.shape[-1])\
                    .type_as(posterior_dist.logits)
                samples = samples.mean(dim=0)
            else:
                samples = super().sample_from_posterior(posterior_dist, **kwargs)
                if self.use_st_hardmax:
                    one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                        .type_as(posterior_dist.logits)
                    if self.hardmax_st_use_logits:
                        samples = posterior_dist.probs
                    samples = samples + one_hot_samples - samples.detach()                    
        else:
            samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)

        # NOTE: workaround for sample_kl                
        if self.use_sample_kl:
            self.update_cache(samples_gumbel=samples)            

        samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
        return samples

    def kl_divergence(self, prior_dist: distributions.Categorical, posterior_dist: distributions.RelaxedOneHotCategorical, input_shape: torch.Size = None, posterior_samples=None, **kwargs):
        # NOTE: workaround for sample_kl: we pass gumbel sample to super here
        if self.use_sample_kl:
            posterior_samples = self.get_raw_cache()["samples_gumbel"]
        kld = super().kl_divergence(prior_dist, posterior_dist, input_shape, posterior_samples, **kwargs)

        # use one_hot sample prior entropy as prior entropy during testing
        if not self.training:
            samples_one_hot = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            sample_prior_entropy = -(samples_one_hot * prior_dist.logits).sum()
            self.update_cache("metric_dict",
                prior_entropy=sample_prior_entropy / input_shape[0]
            )

        return kld

    def postprocess_samples(self, samples):
        samples = super().postprocess_samples(samples)
        # if self.output_use_variance:
        #     samples += torch.normal(torch.zeros_like(samples)) * self.embedding_variance            
        return samples

    def _forward_flat(self, input: torch.Tensor, input_shape: torch.Size, prior: torch.Tensor = None, **kwargs):
        # output = super()._forward_flat(input, input_shape, prior, **kwargs)
        input = self._autoregressive_posterior(input)
        if self.posterior_use_sampled_distance:
            input = input + torch.normal(torch.zeros_like(input)) * self.embedding_variance

        if self.train_em_update and self.em_state == False:
            input = input.detach()

        posterior_dist = self.posterior_distribution(input)
        if self.training and self.use_st_prior_embedding:
            prior_var_logits = self._calc_dist_logits(
                input.view(-1, self.latent_channels, 1, self.embedding_dim), 
                self.embedding.unsqueeze(0),
                embedding_variance=self.prior_embedding_variance,
            ).squeeze(-2)
            U = torch.rand(prior_var_logits.shape).type_as(prior_var_logits)
            gumbel_noise = -torch.log(-torch.log(U + 1e-7) + 1e-7)
            prior_var_samples = F.softmax((prior_var_logits + gumbel_noise) / self.gs_temp, dim=-1)
            posterior_samples = F.softmax((posterior_dist.logits + gumbel_noise) / self.gs_temp, dim=-1)
            samples = posterior_samples + prior_var_samples - prior_var_samples.detach()
            samples = torch.sum(samples.unsqueeze(-1) * self.embedding.unsqueeze(0), dim=-2)
        else:
            samples = self.sample_from_posterior(posterior_dist)

        prior_ar = self._autoregressive_prior(prior=prior, input_shape=input_shape, posterior_samples=samples)

        if self.train_em_update and self.em_state == True:
            prior_ar = prior_ar.detach()
        prior_dist = self.prior_distribution(prior=prior_ar)

        # TODO: check add posterior_samples?
        KLD = torch.sum(self.kl_divergence(prior_dist, posterior_dist, input_shape=input_shape, posterior_samples=samples))
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input_shape[0], # normalize by batch size
            )
        # if implementation has not provide prior_entropy, use kl as prior_entropy instead
        if not "prior_entropy" in self.get_raw_cache("metric_dict"):
            self.update_cache("metric_dict",
                prior_entropy = KLD / input_shape[0], # normalize by batch size
            )

        # perplexity
        avg_probs = torch.mean(posterior_dist.probs, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )
        
        # centriod distance variance
        embedding_centriod = self.embedding.unsqueeze(0).mean(-2, keepdim=True)
        embedding_centriod_distances = self._calc_distances(embedding_centriod, self.embedding.unsqueeze(0)).squeeze(-2)
        self.update_cache("metric_dict", 
            centriod_distance_variance=embedding_centriod_distances.var(dim=-1).sum()
        )

        output = self.postprocess_samples(samples)

        # add continuous vq loss to loss_rate?
        if self.training:
            # for output_use_variance, cont loss is calculated in kl_divergence
            # calculate cont loss for output
            if self.output_use_variance:
                code_probs = posterior_dist.probs
                if self.distance_method == "gaussian":
                    input_mean = input.view(-1, self.latent_channels, 1, self.embedding_dim)
                    distances = self._calc_distances(input_mean, self.embedding.unsqueeze(0)).squeeze(-2)
                    cont_loss = (code_probs * distances).sum() / input_shape[0]
                    output = output + torch.normal(torch.zeros_like(output)) * self._get_embedding_variance()
                else:
                    raise NotImplementedError(f"Unknown distance method {self.distance_method} for cont loss with output variance!")
                cont_loss *= self.cont_loss_weight
            else:
                if self.seperate_update_cont_loss:
                    vq_loss = self._calc_cont_loss(input.detach(), output).sum() / input_shape[0] * self.vq_loss_weight
                    commit_loss = self._calc_cont_loss(input, output.detach()).sum() / input_shape[0] * self.commit_loss_weight
                    cont_loss = (vq_loss + commit_loss) * self.cont_loss_weight
                else:
                    cont_loss = self._calc_cont_loss(input, output).sum() / input_shape[0] \
                        * self.cont_loss_weight
            loss_rate = self.get_raw_cache("loss_dict").get('loss_rate')
            self.update_cache("loss_dict",
                loss_rate=(loss_rate + cont_loss),
            )
            self.update_cache("moniter_dict",
                cont_loss=cont_loss,
            )
            if self.var_scale_anneal:
                self.update_cache("moniter_dict", 
                    var_scale=self.var_scale
                )

            if self.force_st or (self.use_st_below_entropy_threshold and self.entropy_temp < self.entropy_temp_threshold):
                output = output * (1 - self.st_weight) + input * self.st_weight + (output * self.st_weight - input * self.st_weight).detach()

            if self.ema_update_embedding:
                # ema update codebook only during M-step
                if not self.train_em_update or (self.train_em_update and self.em_state == False):
                    self._ema_update_embedding(input, posterior_dist.probs)

        # def _grad(grad):
        #     self._input_grad = grad
        # def _grad2(grad):
        #     self._output_grad = grad
        # def _grad3(grad):
        #     self._embedding_grad = grad
        # if input.requires_grad:
        #     input.register_hook(_grad)
        #     output.register_hook(_grad2)
        #     self.embedding.register_hook(_grad3)

        # if input.requires_grad:
        #     jacobian_io = torch.zeros(input.shape[0], input.shape[1], output.shape[1])
        #     for i in range(output.shape[-1]):
        #         grad_output = torch.zeros_like(output)
        #         grad_output[..., i] = 1
        #         grad = torch.autograd.grad(output, input, grad_output, create_graph=True)
        #         jacobian_io[..., i] = grad[0]
        #     self._jacobian_io = jacobian_io
        #     jacobian_bo = torch.zeros(*self.embedding.shape, output.shape[1])
        #     for i in range(output.shape[-1]):
        #         grad_output = torch.zeros_like(output)
        #         grad_output[..., i] = 1
        #         grad = torch.autograd.grad(output, self.embedding, grad_output, create_graph=True)
        #         jacobian_bo[..., i] = grad[0]
        #     self._jacobian_bo = jacobian_bo
        return output

    def _autoregressive_posterior(self, input : torch.Tensor = None, **kwargs) -> torch.Tensor:
        if self.use_autoregressive_posterior:
            input = input.view(-1, self.latent_channels, self.embedding_dim)
            ar_output = []
            for idx in range(self.latent_channels):
                if idx==0: 
                    ar_input = torch.cat([torch.zeros_like(input[:, 0]), input[:, 0]], dim=-1)
                else:
                    ar_input = input[:, (idx-1):(idx+1)].reshape(-1, 2*self.embedding_dim)
                ar_output.append(self.posterior_ar_model(ar_input))
            return torch.stack(ar_output, dim=1)
        else:
            return super()._autoregressive_posterior(input, **kwargs)


class VQPriorStochasticVQAutoregressivePriorDistributionPriorCoder(StochasticVQAutoregressivePriorDistributionPriorCoder):
    def __init__(self, num_prior_codes=8, **kwargs):
        self.num_prior_codes = num_prior_codes
        super().__init__(**kwargs)

        # TODO: reinitialize prior_params?

    @property
    def num_prior_params(self):
        return self.num_prior_codes * self.embedding_dim
    
    def prior_distribution(self, prior=None, **kwargs) -> distributions.Categorical:
        if prior is None:
            prior = self.prior_params.unsqueeze(0)
        prior = prior.view(-1, self.latent_channels, self.num_prior_codes, self.embedding_dim)
        prior_logits = self._calc_dist_logits(prior, self.embedding.unsqueeze(0)).mean(-2)
        return distributions.Categorical(logits=prior_logits)


class ContinuousBernoulliAutoregressivePriorDistributionPriorCoder(AutoregressivePriorImplDistributionPriorCoder):
    def __init__(self, in_channels=256, latent_channels=64,
                 entropy_temp=1.0, entropy_temp_anneal=False, entropy_temp_threshold=0.0,
                #  cat_reduce_temp=1.0, cat_reduce_temp_anneal=False,
                 **kwargs):
        super().__init__(in_channels, latent_channels, **kwargs)
        
        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            # self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.register_buffer("entropy_temp", torch.tensor(entropy_temp), persistent=False)
            # self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_threshold = entropy_temp_threshold

        # self.cat_reduce_temp_anneal = cat_reduce_temp_anneal
        # if cat_reduce_temp_anneal:
        #     self.cat_reduce_temp = nn.Parameter(torch.tensor(cat_reduce_temp), requires_grad=False)
        # else:
        #     self.cat_reduce_temp = cat_reduce_temp

    @property
    def num_posterior_params(self):
        return 1

    @property
    def num_sample_params(self):
        return 1
    
    def prior_distribution(self, prior=None, **kwargs) -> distributions.ContinuousBernoulli:
        if prior is None:
            prior = self.prior_params.unsqueeze(0)
        prior = prior.view(-1, self.latent_channels)
        return distributions.ContinuousBernoulli(logits=prior)

    def posterior_distribution(self, latent, **kwargs) -> distributions.ContinuousBernoulli:
        latent = latent.view(-1, self.latent_channels)
        return distributions.ContinuousBernoulli(logits=latent)

    def sample_from_posterior(self, posterior_dist: distributions.ContinuousBernoulli, **kwargs):
        if not self.training and self.entropy_temp_anneal:
            # get deterministic {0, 1} sample
            return posterior_dist.probs.round()
        else:
            return super().sample_from_posterior(posterior_dist, **kwargs)

    def kl_divergence(self, 
                      prior_dist: distributions.ContinuousBernoulli, 
                      posterior_dist: distributions.ContinuousBernoulli, 
                      input_shape: torch.Size = None, posterior_samples=None, **kwargs):
        # kld = distributions.kl_divergence(posterior_dist, prior_dist)
        def _kl(p, q):
            t1 = p.mean * (probs_to_logits(p.probs) - probs_to_logits(q.probs))
            t2 = p._cont_bern_log_norm() + torch.log1p(-p.probs)
            t3 = - q._cont_bern_log_norm() - torch.log1p(-q.probs)
            return t1 + t2 + t3
        
        def _entropy(p):
            log_probs0 = torch.log1p(-p.probs)
            log_probs1 = torch.log(p.probs)
            return p.mean * (log_probs0 - log_probs1) - p._cont_bern_log_norm() - log_probs0

        posterior_entropy = posterior_dist.entropy()
        prior_entropy = posterior_entropy - _kl(posterior_dist, prior_dist)

        kld = posterior_entropy * self.entropy_temp - prior_entropy

        # moniter entropy gap for annealing
        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    qp_entropy_gap=(posterior_entropy.sum() / prior_entropy.sum()),
                )
                self.update_cache("moniter_dict", 
                    entropy_temp=self.entropy_temp
                )

        return kld


class IGRCategoricalAutoregressivePriorDistributionPriorCoder(AutoregressivePriorImplDistributionPriorCoder):
    def __init__(self, in_channels=256, latent_channels=8, categorical_dim=128, 
                 relax_temp=0.5, relax_temp_anneal=False,
                 var_scale=1.0, var_scale_anneal=False, fix_prior_var_scale=False,
                 entropy_temp=1.0, entropy_temp_anneal=False, entropy_temp_threshold=0.0,
                #  cat_reduce_temp=1.0, cat_reduce_temp_anneal=False,
                 **kwargs):
        self.categorical_dim = categorical_dim
        super().__init__(in_channels, latent_channels, **kwargs)
        
        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.register_buffer("relax_temp", torch.tensor(relax_temp), persistent=False)

        self.var_scale_anneal = var_scale_anneal
        self.fix_prior_var_scale = fix_prior_var_scale
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.register_buffer("var_scale", torch.tensor(var_scale), persistent=False)

        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            # self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.register_buffer("entropy_temp", torch.tensor(entropy_temp), persistent=False)
            # self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_threshold = entropy_temp_threshold

        # self.cat_reduce_temp_anneal = cat_reduce_temp_anneal
        # if cat_reduce_temp_anneal:
        #     self.cat_reduce_temp = nn.Parameter(torch.tensor(cat_reduce_temp), requires_grad=False)
        # else:
        #     self.cat_reduce_temp = cat_reduce_temp

    @property
    def num_posterior_params(self):
        return 2 * (self.categorical_dim - 1)

    @property
    def num_sample_params(self):
        return self.categorical_dim
    
    def prior_distribution(self, prior=None, **kwargs) -> InvertableGaussianSoftmaxppRelaxedOneHotCategorical:
        if prior is None:
            mean, logvar = self.prior_params.unsqueeze(0).chunk(2, dim=-1)
        else:
            mean, logvar = prior.reshape(-1, self.latent_channels, self.num_posterior_params).chunk(2, dim=-1)
        if self.fix_prior_var_scale:
            logvar = torch.zeros_like(logvar)
        return InvertableGaussianSoftmaxppRelaxedOneHotCategorical(mean, torch.exp(logvar), self.relax_temp)

    def posterior_distribution(self, latent, **kwargs) -> InvertableGaussianSoftmaxppRelaxedOneHotCategorical:
        mean, logvar = latent.reshape(-1, self.latent_channels, self.num_posterior_params).chunk(2, dim=-1)
        if self.var_scale_anneal:
            # manual control var
            scale = self.var_scale
        else:
            scale = torch.exp(logvar)
        return InvertableGaussianSoftmaxppRelaxedOneHotCategorical(mean, scale, self.relax_temp)

    def sample_from_posterior(self, posterior_dist: distributions.Distribution, **kwargs):
        if self.training:
            return super().sample_from_posterior(posterior_dist, **kwargs)
        else:
            posterior_max_idx = posterior_dist.mean.argmax(-1)
            return F.one_hot(posterior_max_idx, self.categorical_dim).type_as(posterior_dist.mean)

    def kl_divergence(self, 
                      prior_dist: InvertableGaussianSoftmaxppRelaxedOneHotCategorical, 
                      posterior_dist: InvertableGaussianSoftmaxppRelaxedOneHotCategorical, 
                      input_shape: torch.Size = None, posterior_samples=None, **kwargs):
        
        if self.entropy_temp_anneal:
            if posterior_samples is None:
                posterior_samples = posterior_dist.rsample()
            logp = prior_dist.log_prob(posterior_samples)
            logq = posterior_dist.log_prob(posterior_samples)
            kld = logq * self.entropy_temp - logp
        else:
            # closed form gaussian kl
            kld = distributions.kl_divergence(
                distributions.Normal(posterior_dist.mean, posterior_dist.stddev),
                distributions.Normal(prior_dist.mean, prior_dist.stddev)
            )

        # calculate categorical kl during testing
        if not self.training:
            # NOTE: May have closed form. See https://arxiv.org/pdf/1912.09588.pdf Appendix B
            prior_cat = prior_dist.to_categorical()
            posterior_max, posterior_max_idx = posterior_dist.mean.max(-1)
            posterior_max_idx = torch.where(posterior_max > 0, posterior_max_idx, posterior_dist.mean.shape[-1])
            kl_cat = -prior_cat.logits * F.one_hot(posterior_max_idx, self.categorical_dim)
            self.update_cache("metric_dict", 
                prior_entropy=kl_cat.sum() / input_shape[0],
            )

        # moniter entropy gap for annealing
        # if self.relax_temp_anneal or self.entropy_temp_anneal:
        #     if self.training:
        #         self.update_cache("moniter_dict", 
        #             qp_entropy_gap=(posterior_entropy.sum() / prior_entropy.sum()),
        #         )

        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    relax_temp=self.relax_temp
                )
        if self.var_scale_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    var_scale=self.var_scale
                )
        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    entropy_temp=self.entropy_temp
                )

        return kld

# TODO: to automatically construct latent variables, we need a new latent distribution interface!
class MultiLatentDistributionPriorCoder(DistributionPriorCoder):
    @property
    def latent_channel_mapping() -> Dict[str, int]:
        raise NotImplementedError()

    @property
    def num_posterior_params(self):
        return sum(self.latent_channel_mapping.values())

    def posterior_distribution(self, latent) -> distributions.Distribution:
        param_dict = latent.split(sum(self.latent_channel_mapping.values()), dim=-1)
        return self._construct_posterior_distributions(**param_dict)

    def _construct_posterior_distributions(self, **kwargs) -> List[distributions.Distribution]:
        raise NotImplementedError()


class ContinuousToDiscreteDistributionPriorCoder(DistributionPriorCoder):
    def __init__(self, in_channels=256, 
        latent_dim=8, 
        num_discrete_embeddings=128, 
        embedding_dim=32,
        sample_blend_mode="interpolate", # "interpolate", "random"
        initialization_scale=None,
        discrete_prior_trainable=False,
        gs_temp=0.5, gs_temp_anneal=False,
        relax_temp=0.5, relax_temp_anneal=False,
        **kwargs):
        self.latent_dim = latent_dim
        self.num_discrete_embeddings = num_discrete_embeddings
        self.embedding_dim = embedding_dim
        self.sample_blend_mode = sample_blend_mode
        
        super().__init__(in_channels, latent_channels=latent_dim, **kwargs)

        # init discrete embeddings
        embedding = torch.zeros(latent_dim, num_discrete_embeddings, embedding_dim)
        if initialization_scale is None:
            initialization_scale = 1/num_discrete_embeddings
        embedding.uniform_(-initialization_scale, initialization_scale)
        self.discrete_embeddings = nn.Parameter(embedding)

        self.discrete_prior_trainable = discrete_prior_trainable
        if discrete_prior_trainable:
            self.discrete_prior_logprob = nn.Parameter(torch.zeros(latent_dim, num_discrete_embeddings))
        else:
            self.register_buffer("discrete_prior_logprob", torch.zeros(latent_dim, num_discrete_embeddings), persistent=False)

        # relax temp
        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.relax_temp = relax_temp

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

    @property
    def latent_channels_out(self):
        return self.latent_dim * self.embedding_dim

    @property
    def num_posterior_params(self):
        return self.num_continuous_posterior_params * self.embedding_dim + self.num_discrete_embeddings

    @property
    def num_continuous_posterior_params(self):
        raise NotImplementedError()

    def continuous_prior_distribution(self, prior=None, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    def prior_distribution(self, prior=None, **kwargs) -> distributions.Distribution:
        return [
            self.continuous_prior_distribution(prior=prior, **kwargs), 
            distributions.Categorical(logits=self.discrete_prior_logprob),
        ]

    def continuous_posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        raise NotImplementedError()

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        latent_continuous, latent_discrete = latent.reshape(-1, self.latent_dim, self.num_posterior_params)\
            .split([self.num_continuous_posterior_params * self.embedding_dim, self.num_discrete_embeddings], dim=-1)
        
        latent_discrete = latent_discrete.view(-1, self.latent_dim, self.num_discrete_embeddings)
        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    gs_temp=self.gs_temp
                )
        return [
            self.continuous_posterior_distribution(latent_continuous, **kwargs), 
            distributions.RelaxedOneHotCategorical(self.gs_temp, logits=latent_discrete)
        ]

    def continuous_sample(self, posterior_dist: distributions.Distribution, **kwargs):
        return super().sample_from_posterior(posterior_dist, **kwargs)

    def continuous_vq_sample(self, posterior_dist: distributions.Distribution, straight_through=False, **kwargs):
        x = posterior_dist.mean
        B = x.shape[0]
        N, M, D = self.discrete_embeddings.size()

        x_flat = x.permute(1, 0, 2)

        pairwise_distances = (x_flat.unsqueeze(2) - self.discrete_embeddings.unsqueeze(1)).pow(2).mean(dim=-1)

        samples_idxs = torch.argmin(pairwise_distances, dim=-1)
        if not self.training:
            self.update_cache("hist_dict",
                code_hist=samples_idxs.view(N, -1).float().cpu().detach_()
            )
        samples_onehot = F.one_hot(samples_idxs, self.num_discrete_embeddings).float()
        samples_onehot = samples_onehot.view(N, B, M)
        
        # N * B * D
        embedding_samples = torch.bmm(samples_onehot, self.discrete_embeddings)
        embedding_samples = embedding_samples.permute(1, 0, 2)
        if straight_through:
            embedding_samples = x + (embedding_samples - x).detach() # straight through

        return samples_idxs, embedding_samples

    def discrete_sample(self, posterior_dist: distributions.Distribution, **kwargs):
        output = super().sample_from_posterior(posterior_dist, **kwargs)
        output = output.view(-1, self.latent_dim, self.num_discrete_embeddings)
        samples = output.permute(1, 0, 2).reshape(self.latent_dim, -1, self.num_discrete_embeddings)
        embedding_samples = torch.bmm(samples, self.discrete_embeddings)
        return embedding_samples.permute(1, 0, 2)

    def continuous_loss(self, prior_dist: distributions.Distribution, posterior_dist: distributions.Distribution, input_shape: torch.Size, **kwargs):
        return super().kl_divergence(prior_dist, posterior_dist, input_shape, **kwargs)

    def discrete_loss(self, prior_dist: distributions.Distribution, posterior_dist: distributions.Distribution, input_shape: torch.Size, **kwargs):        
        prior_entropy = posterior_dist.probs * (posterior_dist.logits - prior_dist.logits.unsqueeze(0))
        
        # perplexity
        avg_probs = torch.mean(posterior_dist.probs, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )

        return prior_entropy

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, **kwargs):
        continuous_prior, discrete_prior = prior_dist
        continuous_post, discrete_post = posterior_dist

        # samples = self.sample_from_posterior(posterior_dist)
        # p_prob = continuous_prior.log_prob(samples).exp() * self.relax_temp + \
        #     discrete_prior.log_prob(samples).exp() * (1 - self.relax_temp) + \
        # q_prob = continuous_post.log_prob(samples).exp() * self.relax_temp + \
        #     discrete_post.log_prob(samples).exp() * (1 - self.relax_temp) + \
        # return q_prob.log() - p_prob.log()
        
        continuous_kl = self.continuous_loss(continuous_prior, continuous_post, input_shape, **kwargs)
        discrete_loss = self.discrete_loss(discrete_prior, discrete_post, input_shape, **kwargs)

        dist_means = continuous_post.mean 
        flat_dim = dist_means.shape[0]
        samples_idxs, embedding_samples = self.continuous_vq_sample(continuous_post)

        loss_quant = (dist_means.detach() - embedding_samples).pow(2)\
            .reshape(flat_dim, self.latent_dim, self.embedding_dim) # .sum(-1) / self.latent_dim

        # self.update_cache("metric_dict", 
        #     prior_entropy = prior_entropy / input_shape[0],
        # )

        if self.training:
            self.update_cache("loss_dict", 
                loss_quant = loss_quant.mean(),
            )


        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    relax_temp=self.relax_temp
                )
        return continuous_kl.sum() * self.relax_temp + discrete_loss.sum() * (1 - self.relax_temp)

    def sample_from_posterior(self, posterior_dist: distributions.Distribution):
        continuous_post, discrete_post = posterior_dist
        continuous_sample = continuous_post.rsample()
        discrete_sample = self.discrete_sample(discrete_post)
        if self.sample_blend_mode == "interpolate":
            output = continuous_sample * self.relax_temp + discrete_sample * (1 - self.relax_temp)
        elif self.sample_blend_mode == "random":
            rand_sample = distributions.RelaxedBernoulli(self.gs_temp, probs=self.relax_temp).sample(continuous_sample.shape)
            output = continuous_sample * rand_sample + discrete_sample * (1 - rand_sample)
        else:
            raise NotImplementedError(f"Unknown sample blend mode {self.sample_blend_mode}")
        return output.reshape(-1, self.latent_channels_out)


class ContinuousToVQDistributionPriorCoder(DistributionPriorCoder):
    def __init__(self, in_channels=256, 
        latent_dim=8, 
        num_discrete_embeddings=128, 
        embedding_dim=32,
        sample_blend_mode="interpolate", # "interpolate", "random"
        commitment_cost=0.25,
        initialization_scale=None,
        discrete_prior_trainable=False,
        discrete_prior_ema_decay=0.9,
        gs_temp=0.5, gs_temp_anneal=False,
        relax_temp=0.5, relax_temp_anneal=False,
        **kwargs):
        self.latent_dim = latent_dim
        self.num_discrete_embeddings = num_discrete_embeddings
        self.embedding_dim = embedding_dim
        self.sample_blend_mode = sample_blend_mode
        self.commitment_cost = commitment_cost
        
        super().__init__(in_channels, latent_channels=latent_dim*embedding_dim, **kwargs)

        # init discrete embeddings
        embedding = torch.zeros(latent_dim, num_discrete_embeddings, embedding_dim)
        if initialization_scale is None:
            initialization_scale = 1/num_discrete_embeddings
        embedding.uniform_(-initialization_scale, initialization_scale)
        self.discrete_embeddings = nn.Parameter(embedding)

        self.discrete_prior_trainable = discrete_prior_trainable
        self.discrete_prior_ema_decay = discrete_prior_ema_decay
        if discrete_prior_trainable:
            self.discrete_prior_logprob = nn.Parameter(torch.zeros(latent_dim, num_discrete_embeddings))
        else:
            self.register_buffer("discrete_prior_logprob", torch.zeros(latent_dim, num_discrete_embeddings), persistent=False)

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        # relax temp
        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.relax_temp = relax_temp

    def continuous_sample(self, posterior_dist: distributions.Distribution, **kwargs):
        return super().sample_from_posterior(posterior_dist, **kwargs)

    def discrete_sample(self, posterior_dist: distributions.Distribution, straight_through=False, **kwargs):
        x = posterior_dist.mean
        B, C = x.shape
        N, M, D = self.discrete_embeddings.size()
        x_embedding_dim = C // N
        assert C == N * x_embedding_dim

        x_flat = x.view(B, N, x_embedding_dim).permute(1, 0, 2)

        pairwise_distances = (x_flat.unsqueeze(2) - self.discrete_embeddings.unsqueeze(1)).pow(2).mean(dim=-1)

        samples_idxs = torch.argmin(pairwise_distances, dim=-1)
        if not self.training:
            self.update_cache("hist_dict",
                code_hist=samples_idxs.view(N, -1).float().cpu().detach_()
            )
        samples_onehot = F.one_hot(samples_idxs, self.num_discrete_embeddings).float()
        samples_onehot = samples_onehot.view(N, B, M)
        
        # N * B * D
        embedding_samples = torch.bmm(samples_onehot, self.discrete_embeddings)
        embedding_samples = embedding_samples.permute(1, 0, 2).reshape(B, N*D)
        if straight_through:
            embedding_samples = x + (embedding_samples - x).detach() # straight through

        return samples_idxs, embedding_samples

    def continuous_loss(self, prior_dist: distributions.Distribution, posterior_dist: distributions.Distribution, input_shape: torch.Size, **kwargs):
        return super().kl_divergence(prior_dist, posterior_dist, input_shape, **kwargs)

    def discrete_loss(self, posterior_dist: distributions.Distribution, input_shape: torch.Size, **kwargs):
        # posterior_dist.log_prob(self.discrete_embeddings.unsqueeze(0))
        dist_means = posterior_dist.mean 
        flat_dim = dist_means.shape[0]
        samples_idxs, embedding_samples = self.discrete_sample(posterior_dist, **kwargs)
        samples_onehot = F.one_hot(samples_idxs, self.num_discrete_embeddings).float()

        loss_quant = (dist_means.detach() - embedding_samples).pow(2)\
            .reshape(flat_dim, self.latent_dim, self.embedding_dim) # .sum(-1) / self.latent_dim
        loss_commit = (dist_means - embedding_samples.detach()).pow(2)\
            .reshape(flat_dim, self.latent_dim, self.embedding_dim) # .sum(-1) / self.latent_dim
        loss_commit *= self.commitment_cost
        
        if self.discrete_prior_trainable:
            if self.training:
                with torch.no_grad():
                    # NOTE: should soft samples be allowed here?
                    total_count = samples_onehot.sum(dim=1)
                    # sum over all gpus
                    if distributed.is_initialized():
                        distributed.all_reduce(total_count)

                    # normalize to probability.
                    normalized_freq = total_count / total_count.sum(-1, keepdim=True)

                    # ema update
                    # ema = (1 - self.update_code_freq_ema_decay) * normalized_freq + self.update_code_freq_ema_decay * self.embedding_freq
                    # self.embedding_freq.copy_(ema)

                    ema = (1 - self.discrete_prior_ema_decay) * normalized_freq + \
                        self.discrete_prior_ema_decay * torch.softmax(self.discrete_prior_logprob, dim=-1)
                    self.discrete_prior_logprob.copy_(torch.log(ema))

            prior_entropy = torch.bmm(samples_onehot, -torch.log_softmax(self.discrete_prior_logprob, dim=-1).unsqueeze(-1)).sum()
        else:
            prior_entropy = math.log(self.num_discrete_embeddings) * samples_idxs.numel()
        
        self.update_cache("metric_dict", 
            prior_entropy = prior_entropy / input_shape[0],
        )

        if self.training:
            self.update_cache("loss_dict", 
                loss_quant = loss_quant.mean(),
                loss_commit = loss_commit.mean() * (1 - self.relax_temp),
            )

        # perplexity
        avg_probs = torch.mean(samples_onehot, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )

        # vq losses has no rate loss
        return torch.zeros_like(dist_means).sum() # (loss_quant + loss_commit).mean() * input_shape[0]

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, **kwargs):
        continuous_kl = self.continuous_loss(prior_dist, posterior_dist, input_shape, **kwargs)
        discrete_loss = self.discrete_loss(posterior_dist, input_shape, **kwargs)
        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    relax_temp=self.relax_temp
                )
        return continuous_kl.sum() * self.relax_temp + discrete_loss.sum() * (1 - self.relax_temp)

    def sample_from_posterior(self, posterior_dist: distributions.Distribution, **kwargs):
        continuous_sample = self.continuous_sample(posterior_dist, **kwargs)
        _, discrete_sample = self.discrete_sample(posterior_dist, straight_through=True, **kwargs)
        if self.sample_blend_mode == "interpolate":
            output = continuous_sample * self.relax_temp + discrete_sample * (1 - self.relax_temp)
        elif self.sample_blend_mode == "random":
            if self.gs_temp_anneal:
                if self.training:
                    self.update_cache("metric_dict", 
                        gs_temp=self.gs_temp
                    )
            rand_sample = distributions.RelaxedBernoulli(self.gs_temp, probs=self.relax_temp).sample(continuous_sample.shape)
            output = continuous_sample * rand_sample + discrete_sample * (1 - rand_sample)
        else:
            raise NotImplementedError(f"Unknown sample blend mode {self.sample_blend_mode}")
        return output

class GaussianDistributionPriorCoder(DistributionPriorCoder):
    def __init__(self, in_channels=256, latent_channels=None, **kwargs):
        super().__init__(in_channels, latent_channels=latent_channels, **kwargs)
        self.register_buffer("prior_means", torch.zeros(1))
        self.register_buffer("prior_scales", torch.ones(1))
        self._prior_dist = distributions.Normal(self.prior_means, self.prior_scales)
    
    @property
    def num_posterior_params(self):
        return 2

    def prior_distribution(self, prior=None, **kwargs):
        if prior is not None:
            return self.posterior_distribution(prior)
        else:
            mix = distributions.Categorical(torch.ones_like(self.prior_means) / self.prior_means.size(0))
            # TODO: per-element vamp
            comp = distributions.Normal(self.prior_means, self.prior_scales)
            return distributions.MixtureSameFamily(mix, comp)

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        mean, logvar = latent.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        return distributions.Normal(mean, std)

    def set_vamp_posterior(self, posterior):
        batch_size = posterior.shape[0]
        channel_size = posterior.shape[1]
        posterior = posterior.reshape(batch_size, channel_size, -1).permute(0, 2, 1)
        posterior = posterior.reshape(-1, self.num_posterior_params).contiguous()

        mean, logvar = posterior.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        self.prior_means = mean[:, 0]
        self.prior_scales = std[:, 0]


class CategoricalDistributionPriorCoder(DistributionPriorCoder):
    def __init__(self, in_channels=256, 
        latent_dim=8, categorical_dim=128,
        # posterior
        use_gs_dist_kl=False, use_gs_prior_kl=False, 
        gs_temp_prior_trainable=True, gs_temp_post_trainable=False,
        use_gs_prior_blend_kl=False,
        posterior_fix_first_logit=False,
        gs_noise_temp_trainable=False, gs_noise_temp_lr_modifier=1.0,
        testing_one_hot_sample=False,
        # prior
        prior_trainable=False, prior_ema_update=False, prior_ema_update_decay=0.9,
        regularize_prior_entropy=False, regularize_prior_entropy_factor=0.01,
        # category reduction
        cat_reduce=False, cat_reduce_method="softminus", cat_reduce_channel_same=False,
        cat_reduce_logit_thres=10.0, cat_reduce_logit_thres_low=None, 
        cat_reduce_logit_bias=0.0,
        cat_reduce_logit_init_range=0.1, cat_reduce_regularizer=0.0,
        cat_reduce_entmax_alpha_trainable=False, cat_reduce_entmax_alpha=1.5, cat_reduce_entmax_alpha_min=1.0, cat_reduce_entmax_alpha_max=2.0,
        eps=1e-8,
        # autoregressive prior
        use_autoregressive_prior=False, 
        ar_method="finitestate", ar_input_sample=True, ar_input_straight_through=False,
        ar_window_size=1, ar_offsets=None,
        ar_fs_method="table",
        ## for table based fsar
        ar_prior_decomp_method="sum", ar_prior_decomp_dim=None,
        ## for MLP based fsar
        ar_mlp_per_channel=False,
        # coding
        coder_type="rans", # current support "rans", "tans"
        fixed_input_shape : Optional[Tuple[int]] = None,
        # annealing
        gs_temp=0.5, gs_temp_anneal=False, gs_temp_min=0.0, gs_temp_threshold=0.0,
        relax_temp=1.0, relax_temp_anneal=False, 
        relax_on_normalized_logits=False, 
        relax_temp_rsample_threshold=0.0, relax_temp_threshold=0.0, sample_straight_through=False,
        entropy_temp=1.0, entropy_temp_anneal=False, entropy_temp_bias=0.0, entropy_temp_threshold=0.0, entropy_temp_reparam="identity", entropy_temp_neginv=False,
        cat_reduce_temp=1.0, cat_reduce_temp_anneal=False,
        **kwargs):
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        latent_channels = latent_dim * categorical_dim
        super().__init__(in_channels, latent_channels=latent_channels, **kwargs)

        self.use_gs_dist_kl = use_gs_dist_kl
        self.use_gs_prior_kl = use_gs_prior_kl
        self.gs_temp_prior_trainable = gs_temp_prior_trainable
        if use_gs_prior_kl:
            if self.gs_temp_prior_trainable:
                self.prior_gs_temp = nn.Parameter(torch.tensor(math.log(gs_temp))) # exp reparam to ensure >0
        self.gs_temp_post_trainable = gs_temp_post_trainable
        if gs_temp_post_trainable:
            self.posterior_gs_temp = nn.Parameter(torch.tensor(math.log(gs_temp)))
        self.use_gs_prior_blend_kl = use_gs_prior_blend_kl
        self.posterior_fix_first_logit = posterior_fix_first_logit
        self.gs_noise_temp_trainable = gs_noise_temp_trainable
        self.gs_noise_temp_lr_modifier = gs_noise_temp_lr_modifier
        if gs_noise_temp_trainable:
            self.gs_noise_temp = nn.Parameter(torch.zeros(1))
            if gs_noise_temp_lr_modifier != 1.0:
                self.gs_noise_temp.lr_modifier = gs_noise_temp_lr_modifier
        self.testing_one_hot_sample = testing_one_hot_sample

        self.prior_trainable = prior_trainable
        self.prior_ema_update = prior_ema_update
        self.prior_ema_update_decay = prior_ema_update_decay
        self.regularize_prior_entropy = regularize_prior_entropy
        self.regularize_prior_entropy_factor = regularize_prior_entropy_factor
        
        self.cat_reduce = cat_reduce
        self.cat_reduce_method = cat_reduce_method
        self.cat_reduce_logit_thres = cat_reduce_logit_thres
        self.cat_reduce_logit_thres_low = cat_reduce_logit_thres\
            if cat_reduce_logit_thres_low is None else cat_reduce_logit_thres_low
        self.cat_reduce_logit_bias = cat_reduce_logit_bias
        self.cat_reduce_channel_same = cat_reduce_channel_same
        self.cat_reduce_regularizer = cat_reduce_regularizer
        self.cat_reduce_entmax_alpha_trainable = cat_reduce_entmax_alpha_trainable
        self.cat_reduce_entmax_alpha_min = cat_reduce_entmax_alpha_min
        self.cat_reduce_entmax_alpha_max = cat_reduce_entmax_alpha_max
        if self.cat_reduce_entmax_alpha_trainable:
            # inverse sigmoid
            self.cat_reduce_entmax_alpha = nn.Parameter(
                -torch.tensor([(1 / max(cat_reduce_entmax_alpha-self.cat_reduce_entmax_alpha_min, eps) ) - 1]).log()
            )
        else:
            self.cat_reduce_entmax_alpha = cat_reduce_entmax_alpha
        self.eps = eps
        if self.cat_reduce:
            cat_reduce_channel_dim = 1 if cat_reduce_channel_same else latent_dim
            cat_reduce_logprob = None
            if self.cat_reduce_method == "softminus":
                cat_reduce_logprob = torch.zeros(cat_reduce_channel_dim, categorical_dim)
            elif self.cat_reduce_method == "sigmoid":
                cat_reduce_logprob = torch.zeros(cat_reduce_channel_dim, categorical_dim) # - self.cat_reduce_logit_thres
            elif self.cat_reduce_method == "entmax":
                cat_reduce_logprob = torch.zeros(cat_reduce_channel_dim, categorical_dim) # - self.cat_reduce_logit_thres
            else:
                raise NotImplementedError(f"Unknown cat_reduce_method {cat_reduce_method}")
            if cat_reduce_logprob is not None:
                self.cat_reduce_logprob = nn.Parameter(cat_reduce_logprob)
                nn.init.uniform_(self.cat_reduce_logprob, -cat_reduce_logit_init_range, cat_reduce_logit_init_range) # add a small variation

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_method = ar_method
        self.ar_input_sample = ar_input_sample
        self.ar_input_straight_through = ar_input_straight_through
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        self.ar_fs_method = ar_fs_method
        self.ar_prior_decomp_method = ar_prior_decomp_method
        self.ar_prior_decomp_dim = ar_prior_decomp_dim
        self.ar_mlp_per_channel = ar_mlp_per_channel
        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_dim - 1
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = [(-offset,) for offset in range(1, self.ar_window_size+1)]
        else:
            self.ar_window_size = len(ar_offsets)


        self.coder_type = coder_type
        # TODO: temp fix for no rans fsar impl! Remove this after fsar-rans is done!
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            if self.coder_type == "rans":
                print("Warning! rans fsar is not implemented! switching to tans!")
                self.coder_type = "tans"
        self.fixed_input_shape = fixed_input_shape

        if use_autoregressive_prior and self.ar_method == "finitestate":
            if self.ar_fs_method == "table":
                if self.ar_prior_decomp_dim is not None:
                    # TODO: directly set ar_prior_categorical_dim?
                    self.ar_prior_categorical_dim = int(self.categorical_dim * self.ar_prior_decomp_dim)
                    # TODO: non-integer ar_prior_decomp_dim?
                    assert isinstance(self.ar_prior_decomp_dim, int)
                    ar_prior_dim = self.ar_prior_categorical_dim * self.ar_window_size
                    if self.ar_prior_decomp_method == "tucker":
                        self.ar_prior_tucker_core = nn.Parameter(torch.ones(self.ar_prior_decomp_dim ** self.ar_window_size) / self.ar_prior_decomp_dim)
                    elif self.ar_prior_decomp_method == "MLP3":
                        self.ar_mlps = nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(self.ar_prior_categorical_dim * self.ar_window_size, 2 * self.ar_prior_categorical_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(2 * self.ar_prior_categorical_dim, self.ar_prior_categorical_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.ar_prior_categorical_dim, self.categorical_dim),
                                )
                                for _ in range(self.latent_dim)
                            ]
                        )
                else:
                    self.ar_prior_categorical_dim = self.categorical_dim
                    # exponential lookup table
                    ar_prior_dim = categorical_dim
                    for _ in range(self.ar_window_size - 1):
                        ar_prior_dim *= categorical_dim
                prior_logprob = torch.zeros(latent_dim, ar_prior_dim, categorical_dim)
                # randomize to enable decomp matrix optimization
                if self.ar_prior_decomp_dim is not None:
                    prior_logprob.uniform_(-1, 1)
            # TODO: per-channel MLP may perform better
            elif self.ar_fs_method == "MLP3":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear((self.categorical_dim + 1) * self.ar_window_size, 2 * self.ar_window_size * (self.categorical_dim + 1)),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.ar_window_size * (self.categorical_dim + 1), 2 * self.categorical_dim),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                            )
                            for _ in range(self.latent_dim)
                        ]
                    )
                else:
                    self.fsar_mlp = nn.Sequential(
                        nn.Linear((self.categorical_dim + 1) * self.ar_window_size, 2 * self.ar_window_size * (self.categorical_dim + 1)),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.ar_window_size * (self.categorical_dim + 1), 2 * self.categorical_dim),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                    )

                # we do not really need this here, just a dummy for compability
                prior_logprob = torch.zeros(latent_dim, categorical_dim)
            else:
                raise NotImplementedError(f"Unknown self.ar_fs_method {self.ar_fs_method}")
        else:
            prior_logprob = torch.zeros(latent_dim, categorical_dim)

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(self.latent_channels, self.latent_channels, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(self.latent_channels, self.latent_channels, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(self.categorical_dim, self.categorical_dim, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(self.categorical_dim, self.categorical_dim, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(self.latent_channels, self.latent_channels, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(self.latent_channels, self.latent_channels, 5, padding=2, mask_type="Checkerboard")

            if ar_model is not None:
                self.ar_model = nn.Sequential(
                    ar_model,
                    # nn.Conv2d(self.latent_channels * 6 // 3, self.latent_channels * 5 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(self.latent_channels * 5 // 3, self.latent_channels * 4 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(self.latent_channels * 4 // 3, self.latent_channels * 3 // 3, 1),
                )

        if prior_trainable:
            self.prior_logprob = nn.Parameter(prior_logprob)
            if prior_ema_update:
                self.prior_logprob.requires_grad = False
        else:
            self.register_buffer("prior_logprob", prior_logprob, persistent=False)

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            # for log_prob of RelaxedOneHotDistribution to work
            # self.gs_temp = gs_temp
            self.register_buffer("gs_temp", torch.tensor(gs_temp), persistent=False)
        self.register_buffer("gs_temp_min", torch.tensor(gs_temp_min, requires_grad=False), persistent=False)
        self.gs_temp_threshold = gs_temp_threshold

        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.relax_temp = relax_temp
        self.relax_on_normalized_logits = relax_on_normalized_logits
        self.relax_temp_rsample_threshold = relax_temp_rsample_threshold
        self.relax_temp_threshold = relax_temp_threshold
        self.sample_straight_through = sample_straight_through

        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            # self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.entropy_temp = entropy_temp
            # self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_bias = entropy_temp_bias
        self.entropy_temp_threshold = entropy_temp_threshold
        self.entropy_temp_reparam = entropy_temp_reparam
        # backward compability
        if entropy_temp_neginv:
            self.entropy_temp_reparam = "neginv"
        # self.entropy_temp_neginv = entropy_temp_neginv

        self.cat_reduce_temp_anneal = cat_reduce_temp_anneal
        if cat_reduce_temp_anneal:
            self.cat_reduce_temp = nn.Parameter(torch.tensor(cat_reduce_temp), requires_grad=False)
        else:
            self.cat_reduce_temp = cat_reduce_temp

        # initalize members for coding
        # self.update_state()

    @property
    def latent_channels_in(self):
        return self.latent_channels

    @property
    def latent_channels_out(self):
        return self.latent_channels

    @property
    def num_posterior_params(self):
        return self.categorical_dim

    def prior_distribution(self, prior=None, **kwargs):
        if prior is not None:
            prior_logits = prior.view(-1, self.latent_dim, self.categorical_dim)
        else:
            prior_logits = self.prior_logprob.unsqueeze(0)
        return distributions.Categorical(logits=prior_logits)

        # NOTE: this is done later in _normalize_prior_logits!
        # if self.cat_reduce:
        #     if self.cat_reduce_method == "softminus":
        #         cat_reduce_logprob = torch.log_softmax(self.cat_reduce_logprob, dim=-1)
        #         prior_soft_thres = cat_reduce_logprob + self.cat_reduce_logit_thres - F.softplus(cat_reduce_logprob + self.cat_reduce_logit_thres, beta=(1./self.cat_reduce_temp)) 
        #         prior_logits = torch.log_softmax(torch.log_softmax(prior_logits, dim=-1) + prior_soft_thres, dim=-1)
        #         prior_probs = prior_logits.exp()
        #     elif self.cat_reduce_method == "sigmoid":
        #         # cat_reduce_logprob = self.cat_reduce_logprob
        #         # if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
        #         #     cat_reduce_logprob = self.cat_reduce_logprob.unsqueeze(1)
        #         prior_logits = self._cat_reduce_sigmoid_logits(prior_logits)
        #         prior_probs = prior_logits.exp()
        #     elif self.cat_reduce_method == "entmax":
        #         prior_probs = self._cat_reduce_entmax_probs(prior_logits)
        # else:
        #     prior_probs = torch.softmax(prior_logits)
        # return distributions.Categorical(probs=prior_probs)

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        latent = latent.view(-1, self.latent_dim, self.categorical_dim)

        latent_probs = None
        latent_logits = None
        if self.cat_reduce:
            if self.cat_reduce_method == "softminus":
                cat_reduce_logprob = torch.log_softmax(self.cat_reduce_logprob, dim=-1)
                prior_soft_thres = cat_reduce_logprob + self.cat_reduce_logit_thres - F.softplus(cat_reduce_logprob + self.cat_reduce_logit_thres, beta=(1./self.cat_reduce_temp)) 
                latent_logits = torch.log_softmax(torch.log_softmax(latent, dim=-1) + prior_soft_thres, dim=-1)
            elif self.cat_reduce_method == "sigmoid":
                latent_logits = self._cat_reduce_sigmoid_logits(latent)#.unsqueeze(0)
                self.update_cache("metric_dict", 
                    cat_reduce_percentage=(self.cat_reduce_logprob<0).sum() / self.cat_reduce_logprob.numel()
                )
                if self.cat_reduce_regularizer > 0 and self.training:
                    # push cat_reduce_logprob away from 0
                    cat_reduce_regularizer = (-self.cat_reduce_logprob.abs()).exp()
                    self.update_cache("loss_dict", 
                        cat_reduce_regularizer=self.cat_reduce_regularizer * cat_reduce_regularizer.mean()
                    )
            elif self.cat_reduce_method == "entmax":
                # NOTE: probs cannot be relaxed, so we do normalization before entmax
                if self.relax_on_normalized_logits:
                    latent_logits = torch.log_softmax(latent, dim=-1) / self.relax_temp
                else:
                    latent_logits = latent / self.relax_temp

                latent_probs = self._cat_reduce_entmax_probs(latent_logits)
                # renormalize with eps to avoid nan
                latent_probs = (latent_probs + self.eps) / (latent_probs + self.eps).sum(dim=-1, keepdim=True)
        else:
            latent_logits = latent

        gs_temp = self.posterior_gs_temp.exp() if self.gs_temp_post_trainable else max(self.gs_temp, self.gs_temp_min)
        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    gs_temp=gs_temp
                )
        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    relax_temp=self.relax_temp
                )
        if self.cat_reduce_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    cat_reduce_temp=self.cat_reduce_temp
                )

        if self.gs_temp_post_trainable and self.training:
            self.update_cache("moniter_dict", 
                posterior_gs_temp = gs_temp
            )

        # categorical_params = dict()
        if latent_probs is not None:
            if self.relax_temp <= self.relax_temp_threshold:
                prob_ont_hot = F.one_hot(latent_probs.argmax(-1), latent_probs.shape[-1])
                latent_probs = latent_probs * prob_ont_hot
                latent_probs = latent_probs / latent_probs.sum(-1, keepdim=True)
            categorical_params = dict(probs=latent_probs)
        else:
            if self.posterior_fix_first_logit:
                latent_logits = torch.cat([
                    torch.zeros_like(latent_logits)[..., :1],
                    latent_logits[..., 1:]
                ], dim=-1)
            if self.relax_on_normalized_logits:
                latent_logits = torch.log_softmax(latent_logits, dim=-1) / self.relax_temp
            else:
                latent_logits = latent_logits / self.relax_temp
            if self.relax_temp <= self.relax_temp_threshold:
                prob_ont_hot = F.one_hot(latent_logits.argmax(-1), latent_logits.shape[-1])
                latent_probs = latent_logits * prob_ont_hot
                latent_probs = latent_probs / latent_probs.sum(-1, keepdim=True)
                categorical_params = dict(probs=latent_probs)
            else:
                categorical_params = dict(logits=latent_logits)
                # return distributions.RelaxedOneHotCategorical(gs_temp, probs=torch.softmax(latent_logits, dim=-1))
                # from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
                # return ExpRelaxedCategorical(gs_temp, logits=latent_logits)
            
            if self.gs_noise_temp_trainable:
                gs_noise_temp = self.gs_noise_temp.exp() # 0 clamped reparam
                if self.training:
                    self.update_cache("moniter_dict", 
                        gs_noise_temp=gs_noise_temp
                    )
                return DoubleRelaxedOneHotCategorical(gs_temp, gs_noise_temp, **categorical_params)
            else:
                return distributions.RelaxedOneHotCategorical(gs_temp, **categorical_params)

    def _get_reparam_entropy_temp(self):
        entropy_temp = self.entropy_temp if self.entropy_temp >= self.entropy_temp_threshold else 0.0
        if self.entropy_temp_reparam == "neginv":
            entropy_temp = -1 / (entropy_temp if entropy_temp > self.eps else self.eps)
        elif self.entropy_temp_reparam == "log":
            entropy_temp = torch.log(entropy_temp) if entropy_temp > self.eps else torch.log(self.eps)
        entropy_temp = entropy_temp + self.entropy_temp_bias
        return entropy_temp

    def _cat_reduce_sigmoid_logits(self, input):
        return torch.log_softmax(
            (
                self.cat_reduce_logit_thres * torch.sigmoid(input) \
                - self.cat_reduce_logit_thres_low * torch.sigmoid(-self.cat_reduce_logprob / self.cat_reduce_temp) \
                + self.cat_reduce_logit_bias
            ), dim=-1)
    
    def _get_entmax_probs(self):
        if self.cat_reduce_temp_anneal:
            alpha = self.cat_reduce_entmax_alpha_max - self.cat_reduce_temp * (self.cat_reduce_entmax_alpha_max - self.cat_reduce_entmax_alpha_min)
        else:
            if self.cat_reduce_entmax_alpha_trainable:
                alpha = self.cat_reduce_entmax_alpha_min + torch.sigmoid(self.cat_reduce_entmax_alpha) * \
                    (self.cat_reduce_entmax_alpha_max - self.cat_reduce_entmax_alpha_min)
                self.update_cache("metric_dict", 
                    cat_reduce_entmax_alpha=alpha,
                )
            else:
                alpha = self.cat_reduce_entmax_alpha

        if alpha <= 1.0:
            entmax_probs = torch.softmax(self.cat_reduce_logprob, dim=-1)
        else:
            entmax_probs = entmax_bisect(self.cat_reduce_logprob, alpha=alpha, dim=-1)

        return entmax_probs
    
    def _cat_reduce_entmax_probs(self, input):

        entmax_probs = self._get_entmax_probs()
        
        cat_reduce_percentage = (entmax_probs==0).sum() / self.cat_reduce_logprob.numel()
        self.update_cache("metric_dict", 
            cat_reduce_percentage=cat_reduce_percentage,
        )
        
        input_probs = torch.softmax(input, dim=-1) * entmax_probs.unsqueeze(0)
        return input_probs / (input_probs.sum(dim=-1, keepdim=True) + self.eps) # renormalize

    def _merge_prior_logits_ar(self, prior_logits_ar):
        # prior_logits_ar : [batch_size, latent_dim, ar_dim, decomp_dim, categorical_dim]
        # TODO: consider cat_reduce (MLP3 refuse reduced category!)
        categorical_dim = self.categorical_dim # prior_logits_ar.shape[-1]
        # aggregate samples
        if self.ar_prior_decomp_method == "tucker":
            tucker_matrix = self.ar_prior_tucker_core.reshape(-1, self.ar_prior_decomp_dim).unsqueeze(0).unsqueeze(0)
            prior_logits_ar = prior_logits_ar.transpose(-2, -1)
            for w_offset in range(self.ar_window_size):
                tucker_matrix = torch.matmul(tucker_matrix, prior_logits_ar.select(2, w_offset).unsqueeze(-1)).squeeze(-1)
                if w_offset != self.ar_window_size -1:
                    tucker_matrix = tucker_matrix.reshape(*tucker_matrix.shape[:-1], -1, self.ar_prior_decomp_dim)
            prior_logits = tucker_matrix.squeeze(-1)
        elif self.ar_prior_decomp_method == "MLP3":
            prior_logits_ar = prior_logits_ar.reshape(-1, self.latent_dim, self.ar_window_size * self.ar_prior_decomp_dim * categorical_dim)
            prior_logits = torch.stack([mlp(logits.squeeze(1)) for mlp, logits in zip(self.ar_mlps, prior_logits_ar.split(1, dim=1))], dim=1)
        else:
            prior_logits = prior_logits_ar.sum(-2).sum(-2)
        return prior_logits

    def _normalize_prior_logits(self, prior_logits):
        if self.cat_reduce:
            if self.cat_reduce_method == "softminus":
                cat_reduce_logprob = torch.log_softmax(self.cat_reduce_logprob, dim=-1)
                prior_soft_thres = cat_reduce_logprob + self.cat_reduce_logit_thres - F.softplus(cat_reduce_logprob + self.cat_reduce_logit_thres, beta=(1./self.cat_reduce_temp)) 
                prior_logits = torch.log_softmax(prior_logits, dim=-1) + prior_soft_thres
                # prior_logits = torch.log_softmax(prior_logits, dim=-1)
            elif self.cat_reduce_method == "sigmoid":
                # cat_reduce_logprob = self.cat_reduce_logprob
                # if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
                #     cat_reduce_logprob = self.cat_reduce_logprob.unsqueeze(1)
                prior_logits = self._cat_reduce_sigmoid_logits(prior_logits) # .unsqueeze(0)
                # prior_logits = torch.log_softmax(prior_logits, dim=-1)
            elif self.cat_reduce_method == "entmax":
                # cat_reduce_logprob = self.cat_reduce_logprob
                # if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
                #     cat_reduce_logprob = self.cat_reduce_logprob.unsqueeze(1)
                prior_probs = self._cat_reduce_entmax_probs(prior_logits) # .unsqueeze(0)
                prior_logits = (prior_probs + self.eps).log() / (prior_probs + self.eps).sum(dim=-1, keepdim=True)
        prior_logits = torch.log_softmax(prior_logits, dim=-1)
        return prior_logits

    def kl_divergence(self, prior_dist : distributions.Categorical, posterior_dist : distributions.RelaxedOneHotCategorical, input_shape : torch.Size = None, posterior_samples=None, **kwargs):
        prior_logits = prior_dist.logits # self.prior_logprob.unsqueeze(0) # prior_dist.logits # N * latent_dim * categorical_dim
        if posterior_samples is None:
            posterior_samples = posterior_dist.rsample()
        else:
            posterior_samples = posterior_samples.reshape(-1, self.latent_dim, self.categorical_dim)
        if self.use_autoregressive_prior:
            # if input_shape is None:
            #     input_shape = posterior_dist.logits.shape[:-1]
            batch_size = input_shape[0]
            spatial_shape = input_shape[2:]
            posterior_samples_shape = posterior_samples.shape # N*spatial_dim*C*categorical_dim
            if self.ar_method == "finitestate":
                # find samples for ar
                # reshape as input format (N*spatial_dim*C*categorical_dim -> N*C*spatial_dim*categorical_dim)
                posterior_samples_reshape = posterior_samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.categorical_dim).movedim(-2, 1)
                # merge prior logits
                if self.ar_fs_method == "table":
                    autoregressive_samples = []
                    # for w_offset in range(self.ar_window_size):
                    for ar_offset in self.ar_offsets:
                        # take ar samples
                        if self.ar_input_sample:
                            ar_samples = posterior_samples_reshape
                        else:
                            ar_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1]).type_as(posterior_dist.logits)
                            # gradient trick
                            if self.ar_input_straight_through:
                                ar_samples = posterior_dist.probs + (ar_samples - posterior_dist.probs).detach() # straight through
                            # ar_samples = posterior_dist.probs * one_hot_samples
                            # ar_samples = ar_samples / ar_samples.sum(-1, keepdim=True)
                            # reshape
                            ar_samples = ar_samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.categorical_dim).movedim(-2, 1)
                        default_samples = torch.zeros_like(posterior_samples_reshape)
                        default_samples[..., 0] = 1
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                # TODO: leave 0 as unknown sample, let total categories categorical_dim+1 (left for compability)
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, posterior_samples_reshape.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        # reshape back to sample format
                        ar_samples = ar_samples.movedim(1, -2).reshape_as(posterior_samples)
                        # ar_samples = torch.cat((
                        #     # first latent dim use default prior 0
                        #     F.one_hot(torch.zeros(posterior_dist.probs.shape[0], w_offset+1, dtype=torch.long).to(device=posterior_samples.device), self.categorical_dim),
                        #     # dims > 1 access prior according to the previous posterior
                        #     posterior_samples[:, :(self.latent_dim-1-w_offset)],
                        # ), dim=1)
                        autoregressive_samples.append(ar_samples)

                    if self.ar_prior_decomp_dim is not None:
                        # normalize logits to 0 mean
                        prior_logits = prior_logits - prior_logits.mean(-1, keepdim=True)
                        autoregressive_samples = torch.stack(autoregressive_samples, dim=-2)
                        prior_logits_ar = prior_logits.reshape(*prior_logits.shape[:-2], self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                        prior_logits_ar = (prior_logits_ar * autoregressive_samples.unsqueeze(-2).unsqueeze(-1)).sum(-2)
                        # aggregate samples
                        prior_logits = self._merge_prior_logits_ar(prior_logits_ar)
                        # if self.ar_prior_decomp_method == "tucker":
                        #     tucker_matrix = self.ar_prior_tucker_core.reshape(-1, self.ar_prior_decomp_dim).unsqueeze(0).unsqueeze(0)
                        #     prior_logits_ar = prior_logits_ar.transpose(-2, -1)
                        #     for w_offset in range(self.ar_window_size):
                        #         tucker_matrix = torch.matmul(tucker_matrix, prior_logits_ar.select(2, w_offset).unsqueeze(-1)).squeeze(-1)
                        #         if w_offset != self.ar_window_size -1:
                        #             tucker_matrix = tucker_matrix.reshape(*tucker_matrix.shape[:-1], -1, self.ar_prior_decomp_dim)
                        #     prior_logits = tucker_matrix.squeeze(-1)
                        # elif self.ar_prior_decomp_method == "MLP3":
                        #     prior_logits_ar = prior_logits_ar.reshape(-1, self.latent_dim, self.ar_window_size * self.ar_prior_categorical_dim)
                        #     prior_logits = torch.stack([mlp(logits.squeeze(1)) for mlp, logits in zip(self.ar_mlps, prior_logits_ar.split(1, dim=1))], dim=1)
                        # else:
                        #     prior_logits = prior_logits_ar.sum(-2).sum(-2)
                    else:
                        aggregated_samples = autoregressive_samples[0]
                        for w_offset in range(self.ar_window_size-1):
                            cur_sample = autoregressive_samples[w_offset+1]
                            for _ in range(w_offset+1):
                                cur_sample = cur_sample.unsqueeze(-2)
                            aggregated_samples = aggregated_samples.unsqueeze(-1) * cur_sample
                        aggregated_samples = aggregated_samples.reshape(*posterior_samples.shape[:-1], -1)
                        prior_logits = torch.matmul(prior_logits.transpose(-2, -1), aggregated_samples.unsqueeze(-1)).squeeze(-1)
                elif self.ar_fs_method == "MLP3":
                    autoregressive_samples = []
                    for ar_offset in self.ar_offsets:
                        default_sample = torch.zeros_like(posterior_samples_reshape)[..., :1]
                        if self.ar_input_sample:
                            ar_samples = posterior_samples_reshape
                        else:
                            ar_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1]).type_as(posterior_dist.logits)
                            # gradient trick
                            if self.ar_input_straight_through:
                                ar_samples = posterior_dist.probs + (ar_samples - posterior_dist.probs).detach() # straight through
                            # ar_samples = posterior_dist.probs * one_hot_samples
                            # ar_samples = ar_samples / ar_samples.sum(-1, keepdim=True)
                            # reshape
                            ar_samples = ar_samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.categorical_dim).movedim(-2, 1)

                        # take ar samples
                        ar_samples = torch.cat(
                            [
                                default_sample,
                                ar_samples,
                            ], dim=-1
                        )
                        # leave 0 as unknown sample, let total categories categorical_dim+1
                        default_samples = torch.cat(
                            [
                                default_sample + 1,
                                torch.zeros_like(posterior_samples_reshape),
                            ], dim=-1
                        )
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, posterior_samples_reshape.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        autoregressive_samples.append(ar_samples)
                    # [batch_size, self.latent_dim, *spatial_shape, self.ar_window_size*(self.categorical_dim+1)]
                    autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                    if self.ar_mlp_per_channel:
                        autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                            .reshape(posterior_samples_shape[0], self.latent_dim, self.ar_window_size*(self.categorical_dim+1))
                        ar_logits_reshape = torch.stack([mlp(sample.squeeze(1)) for mlp, sample in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=1))], dim=1)
                        prior_logits = ar_logits_reshape + prior_logits
                    else:
                        autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(-1, self.ar_window_size*(self.categorical_dim+1))
                        ar_logits_reshape = self.fsar_mlp(autoregressive_samples_flat)
                        # merge ar logits and prior logits
                        prior_logits = ar_logits_reshape.reshape_as(posterior_samples) + prior_logits
                # normalize logits
                prior_logits = self._normalize_prior_logits(prior_logits)
                # prior_logits = torch.cat((prior_logits[:, :1, 0], prior_logits), dim=1)
            else:
                assert len(spatial_shape) == 2
                posterior_samples_reshape = posterior_samples.reshape(batch_size, *spatial_shape, self.latent_channels).movedim(-1, 1)
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        posterior_samples_reshape = posterior_samples_reshape.reshape(batch_size, self.latent_dim, self.categorical_dim, *spatial_shape)\
                            .permute(0, 2, 1, 3, 4)
                    prior_logits_reshape = self.ar_model(posterior_samples_reshape)
                    if self.ar_method.startswith("maskconv3d"):
                        prior_logits_reshape = prior_logits_reshape.permute(0, 2, 1, 3, 4)\
                            .reshape(batch_size, self.latent_channels, *spatial_shape)
                elif self.ar_method.startswith("checkerboard"):
                    prior_logits_reshape = self.ar_model(posterior_samples_reshape)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=posterior_samples_reshape.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                prior_logits = prior_logits_reshape.movedim(1, -1).reshape(posterior_samples_shape[0], self.latent_dim, self.categorical_dim)
                prior_logits = self._normalize_prior_logits(prior_logits)
        else:
            prior_logits = prior_dist.logits
        
        entropy_temp = self._get_reparam_entropy_temp()
        posterior_entropy = posterior_dist.probs * posterior_dist.logits
        posterior_entropy[posterior_dist.probs == 0] = 0 # prevent nan
        prior_entropy = posterior_dist.probs * prior_logits

        # calculate one_hot sample kl divergence if specified
        if self.testing_one_hot_sample and not self.training:
            one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            kld = one_hot_samples * -prior_logits
        else:
            if self.use_gs_dist_kl:
                # posterior_samples_eps = (posterior_samples + self.eps) / (posterior_samples + self.eps).sum(-1, keepdim=True)
                # posterior_probs_eps = (posterior_dist.probs + self.eps) / (posterior_dist.probs + self.eps).sum(-1, keepdim=True)
                # posterior_entropy_gs = posterior_dist.log_prob(clamp_probs(posterior_samples))
                
                # posterior_samples_logits = posterior_samples_eps.log()
                # posterior_dist_logits = posterior_probs_eps.log()
                # log_scale = (torch.full_like(posterior_dist.temperature, float(self.categorical_dim)).lgamma() -
                #             posterior_dist.temperature.log().mul(-(self.categorical_dim - 1)))
                # score = posterior_dist_logits - posterior_samples_logits.mul(posterior_dist.temperature)
                # score = (score - posterior_samples_logits - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
                # posterior_entropy_gs = score + log_scale
                if self.use_gs_prior_kl:
                    posterior_logits = posterior_dist.logits
                    uniforms = clamp_probs(torch.rand(posterior_logits.shape, dtype=posterior_logits.dtype, device=posterior_logits.device))
                    gumbels = -((-(uniforms.log())).log())
                    scores = (posterior_logits + gumbels) / posterior_dist.temperature
                    posterior_samples_logits =  scores - scores.logsumexp(dim=-1, keepdim=True)
                    # posterior_samples_logits = posterior_samples
                    log_scale = (torch.full_like(posterior_dist.temperature, float(self.categorical_dim)).lgamma() -
                                posterior_dist.temperature.log().mul(-(self.categorical_dim - 1)))
                    
                    posterior_score = posterior_logits - posterior_samples_logits.mul(posterior_dist.temperature)
                    posterior_score = (posterior_score - posterior_samples_logits - posterior_score.logsumexp(dim=-1, keepdim=True)).sum(-1)
                    posterior_entropy_gs = posterior_score + log_scale

                    # prior_dist_gs = distributions.RelaxedOneHotCategorical(posterior_dist.temperature, logits=prior_logits)
                    # prior_entropy_gs = prior_dist_gs.log_prob(clamp_probs(posterior_samples))
                    
                    # prior_probs_eps = (prior_dist_gs.probs + self.eps) / (posterior_dist.probs + self.eps).sum(-1, keepdim=True)
                    # prior_logits_eps = prior_probs_eps.log()
                    prior_gs_temp = self.prior_gs_temp.exp() if self.gs_temp_prior_trainable else posterior_dist.temperature
                    prior_log_scale = (torch.full_like(prior_gs_temp, float(self.categorical_dim)).lgamma() -
                                prior_gs_temp.log().mul(-(self.categorical_dim - 1)))
                    prior_score = prior_logits - posterior_samples_logits.mul(prior_gs_temp)
                    prior_score = (prior_score - posterior_samples_logits - prior_score.logsumexp(dim=-1, keepdim=True)).sum(-1)
                    prior_entropy_gs = prior_score + prior_log_scale
                    # posterior_score = (posterior_dist.logits - posterior_samples.log().mul(posterior_dist.temperature)).logsumexp(dim=-1, keepdim=True)
                    # prior_score = (prior_logits - posterior_samples.log().mul(posterior_dist.temperature)).logsumexp(dim=-1, keepdim=True)
                    # kld = ((posterior_dist.logits - prior_logits) - (posterior_score - prior_score)).sum(-1)
                    if self.gs_temp_prior_trainable and self.training:
                        self.update_cache("moniter_dict", 
                            prior_gs_temp=prior_gs_temp
                        )
                    if self.use_gs_prior_blend_kl:
                        posterior_entropy_discrete = (posterior_samples * posterior_dist.logits).sum(-1)
                        prior_entropy_discrete = (posterior_samples * prior_logits).sum(-1)
                        blend_param = 1 - torch.exp(-prior_gs_temp)
                        prior_entropy_gs = blend_param * prior_entropy_gs + (1 - blend_param) * prior_entropy_discrete
                        posterior_entropy_gs = blend_param * posterior_entropy_gs + (1 - blend_param) * posterior_entropy_discrete
                else:
                    posterior_entropy_gs = (posterior_samples * posterior_dist.logits).sum(-1)
                    prior_entropy_gs = (posterior_samples * prior_logits).sum(-1)
                kld = posterior_entropy_gs * entropy_temp - prior_entropy_gs
            else:
                kld = posterior_entropy * entropy_temp - prior_entropy

        kld = kld.sum()

        # moniter entropy gap for annealing
        # TODO: move some keys of metric_dict to moniter_dict
        # if self.relax_temp_anneal or self.entropy_temp_anneal:
        if self.training:
            self.update_cache("moniter_dict", 
                qp_entropy_gap=(posterior_entropy.sum() / prior_entropy.sum()),
            )
            self.update_cache("moniter_dict", 
                posterior_entropy=-posterior_entropy.sum(),
            )
            self.update_cache("moniter_dict", 
                prior_cross_entropy=-prior_entropy.sum(),
            )
            self.update_cache("moniter_dict", 
                prior_self_entropy=-(prior_logits.exp() * prior_logits).sum(),
            )
            one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            self.update_cache("moniter_dict", 
                prior_one_hot_entropy=-(one_hot_samples * prior_logits).sum(),
            )

        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("moniter_dict", 
                    entropy_temp=entropy_temp
                )

        if self.regularize_prior_entropy:
            prior_self_entropy = -(prior_logits * prior_logits.exp()).sum()
            kld += prior_self_entropy * self.regularize_prior_entropy_factor
            self.update_cache("metric_dict", 
                prior_self_entropy=prior_self_entropy,
            )
            self.update_cache("hist_dict", 
                prior_probs=prior_logits.exp(),
            )

        return kld

    def sample_from_posterior(self, posterior_dist: distributions.RelaxedOneHotCategorical):
        if self.testing_one_hot_sample and not self.training:
            # one hot sample
            one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            return one_hot_samples
            # output = super().sample_from_posterior(posterior_dist)
            # output[posterior_dist.logits.argmax(-1)] = 1.0
            # one_hot_samples = F.one_hot(posterior_dist.logits.argmax(-1), posterior_dist.logits.shape[-1])\
            #     .type_as(posterior_dist.logits)
            # one_hot_samples[posterior_dist.logits.argmax(-1)] = 1.0
            # return posterior_dist.probs

        if self.relax_temp >= self.relax_temp_rsample_threshold:
            output = super().sample_from_posterior(posterior_dist)
        else:
            # one hot sample
            samples = distributions.Categorical(probs=posterior_dist.probs).sample()
            output = F.one_hot(samples, posterior_dist.logits.shape[-1])\
                .type_as(posterior_dist.logits)
            if self.sample_straight_through:
                output = posterior_dist.probs + (output - posterior_dist.probs).detach() # straight through

        if self.training:
            sample_entropy = -output * output.log()
            sample_entropy[output == 0] = 0
            self.update_cache("moniter_dict", 
                sample_entropy=sample_entropy.sum(),
            )


        # hard one hot sample if gs temp is below threshold
        if posterior_dist.temperature < self.gs_temp_threshold:
            one_hot_samples = F.one_hot(output.argmax(-1), output.shape[-1])\
                .type_as(output)
            if self.sample_straight_through:
                output = output + (one_hot_samples - output).detach() # straight through
            else:
                output = one_hot_samples
        
        # ema update
        if self.prior_ema_update:
            with torch.no_grad():
                total_count = output.sum(dim=0)
                # sum over all gpus
                if distributed.is_initialized():
                    distributed.all_reduce(total_count)

                # normalize to probability.
                normalized_freq = total_count / total_count.sum(-1, keepdim=True)

                # ema update
                ema = (1 - self.prior_ema_update_decay) * normalized_freq + \
                    self.prior_ema_update_decay * torch.softmax(self.prior_logprob, dim=-1)
                self.prior_logprob.copy_(torch.log(ema))

        return output.view(-1, self.latent_channels)

    def _encode_transformed(self, input : torch.Tensor, prior : torch.Tensor = None, **kwargs) -> bytes:
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        spatial_shape = input.shape[2:]
        assert channel_size == self.latent_dim * self.categorical_dim
        
        # posterior_dist = self.posterior_distribution(input.movedim(1, -1).reshape(-1, self.latent_dim, self.categorical_dim))
        # prior_dist = self.prior_distribution(prior=prior)

        # samples = self.sample_from_posterior(posterior_dist)

        # KLD = self.kl_divergence(prior_dist, posterior_dist, input_shape=(batch_size, self.latent_dim, *spatial_shape))

        input = input.view(batch_size, self.latent_dim, self.categorical_dim, *spatial_shape)
        
        # non-finite autoregressive
        data_bytes = b''
        if self.use_autoregressive_prior:
            samples = torch.argmax(input, dim=2)
            input_one_hot = F.one_hot(samples, self.categorical_dim).type_as(input).movedim(-1, 2)\
                .reshape(batch_size, self.latent_dim*self.categorical_dim, *spatial_shape)
            if self.ar_method.startswith("maskconv"):
                if self.ar_method.startswith("maskconv3d"):
                    input_one_hot = input_one_hot.reshape(batch_size, self.latent_dim, self.categorical_dim, *spatial_shape)\
                        .permute(0, 2, 1, 3, 4)
                prior_logits_reshape = self.ar_model(input_one_hot)
                # move batched dimensions to last for correct decoding
                if self.ar_method.startswith("maskconv3d"):
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                    samples = samples.movedim(0, -1)
                else:
                    prior_logits_reshape = prior_logits_reshape.reshape(batch_size, self.latent_dim, self.categorical_dim, *spatial_shape)
                    prior_logits_reshape = prior_logits_reshape.movedim(0, -1).movedim(0, -1)
                    samples = samples.movedim(0, -1).movedim(0, -1)
                # move categorical dim
                prior_logits_reshape = prior_logits_reshape.movedim(0, -1)
                
                rans_encoder = RansEncoder()

                data = samples.detach().cpu().numpy().astype(np.int32)
                prior_probs = torch.softmax(prior_logits_reshape, dim=-1)
                cdfs = pmf_to_quantized_cdf_batched(prior_probs.reshape(-1, prior_probs.shape[-1]))
                cdfs = cdfs.detach().cpu().numpy().astype(np.int32)

                data = data.reshape(-1)
                indexes = np.arange(len(data), dtype=np.int32)
                cdf_lengths = np.array([len(cdf) for cdf in cdfs])
                offsets = np.zeros(len(indexes)) # [0] * len(indexes)

                with self.profiler.start_time_profile("time_rans_encoder"):
                    data_bytes = rans_encoder.encode_with_indexes_np(
                        data, indexes,
                        cdfs, cdf_lengths, offsets
                    )

            elif self.ar_method.startswith("checkerboard"):
                prior_logits_reshape = self.ar_model(input_one_hot)
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=input.device)
                # input_base = torch.cat([
                #     input_one_hot[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                #     input_one_hot[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                # ], dim=-1)
                # input_ar = torch.cat([
                #     input_one_hot[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                #     input_one_hot[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                # ], dim=-1)
                prior_logits_ar = torch.cat([
                    prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                prior_logits_ar = prior_logits_ar.reshape(batch_size, self.latent_dim, self.categorical_dim, *prior_logits_ar.shape[-2:]).movedim(2, -1)

                samples_base = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0],
                ], dim=-1)
                data_base = samples_base.detach().cpu().numpy()
                indexes_base = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_size, 1, np.prod(spatial_shape) // 2).reshape_as(samples_base).numpy()
                samples_ar = torch.cat([
                    samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                    samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                ], dim=-1)
                data_ar = samples_ar.detach().cpu().numpy()
                
                # prepare for coding (base)
                data_base = data_base.astype(np.int32).reshape(-1)
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                # prepare for coding (ar)
                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                data_ar = data_ar.reshape(-1)
                indexes_ar = np.arange(len(data_ar), dtype=np.int32)
                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)


                rans_encoder = BufferedRansEncoder()
                with self.profiler.start_time_profile("time_rans_encoder"):
                    rans_encoder.encode_with_indexes_np(
                        data_base, indexes_base,
                        cdfs_base, cdf_sizes_base, offsets_base
                    )
                    rans_encoder.encode_with_indexes_np(
                        data_ar, indexes_ar,
                        cdfs_ar, cdf_sizes_ar, offsets_ar
                    )
                    data_bytes = rans_encoder.flush()
            else:
                pass

        
        if len(data_bytes) == 0:

            # TODO: use iterative autoregressive for overwhelmed states
            if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
                raise NotImplementedError("Overwhelmed states!")

            if self.cat_reduce:
                input = input.index_select(2, self._reduce_mask).view(batch_size, self.latent_dim, -1, *spatial_shape)
                # if self.cat_reduce_method == "softminus":
                #     raise NotImplementedError()
                # elif self.cat_reduce_method == "sigmoid":
                #     if not self.cat_reduce_channel_same:
                #         # TODO: different transformation for different channels
                #         raise NotImplementedError()
                #     else:
                #         reduce_mask = (self.cat_reduce_logprob[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                #         input = input.index_select(2, reduce_mask).view(batch_size, self.latent_dim, -1, *spatial_shape)
                # elif self.cat_reduce_method == "entmax":
                #     reduce_mask = (entmax_bisect(self.cat_reduce_logprob) > 0)
            samples = torch.argmax(input, dim=2)
            data = samples.detach().cpu().numpy()
            # self._samples_cache = samples

            if self.coder_type == "rans":
                # TODO: autoregressive rans
                # if self.use_autoregressive_prior and self.ar_method == "finitestate":
                #     raise NotImplementedError()
                rans_encoder = RansEncoder()
                indexes = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_size, 1, np.prod(spatial_shape)).reshape_as(samples).numpy()
                
                # prepare for coding
                data = data.astype(np.int32).reshape(-1)
                indexes = indexes.astype(np.int32).reshape(-1)
                cdfs = self._prior_cdfs
                cdf_sizes = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets = np.zeros(len(self._prior_cdfs))
                with self.profiler.start_time_profile("time_rans_encoder"):
                    data_bytes = rans_encoder.encode_with_indexes_np(
                        data, indexes, cdfs, cdf_sizes, offsets
                    )
            elif self.coder_type == "tans":
                assert TANS_AVAILABLE
                assert self.categorical_dim <= 256
                tans_encoder = TansEncoder()
                indexes = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_size, 1, np.prod(spatial_shape)).reshape_as(samples).numpy()
                
                # prepare for coding
                data = data.astype(np.int32)#.reshape(-1)
                indexes = indexes.astype(np.int32)#.reshape(-1)
                ctables = self._prior_ctables
                offsets = np.zeros(len(self._prior_ctables))
                with self.profiler.start_time_profile("time_tans_encoder"):
                    if self.use_autoregressive_prior and self.ar_method == "finitestate":
                        # data_bytes = tans_encoder.encode_autoregressive_np(
                        #     data, indexes, np.array(self.ar_offsets), ctables, offsets, self.categorical_dim
                        # )
                        data_bytes = self._encoder.encode_autoregressive_np(
                            data, indexes, np.array(self.ar_offsets)
                        )
                    else:
                        data_bytes = tans_encoder.encode_with_indexes_np(
                            data, indexes, ctables, offsets
                        )

        if len(data_bytes) == 0:
            return b''

        # store sample shape in header
        byte_strings = []
        if self.fixed_input_shape is not None:
            assert batch_size == self.fixed_input_shape[0]
            assert spatial_shape == self.fixed_input_shape[1:]
        else:
            byte_head = [struct.pack("B", len(spatial_shape)+1)]
            byte_head.append(struct.pack("<H", batch_size))
            for dim in spatial_shape:
                byte_head.append(struct.pack("<H", dim))
            byte_strings.extend(byte_head)
        byte_strings.append(data_bytes)
        return b''.join(byte_strings)

    def _decode_transformed(self, byte_string : bytes, prior : torch.Tensor = None, **kwargs) -> torch.Tensor:
        # if len(byte_string) == 0:
        #     return torch.zeros(1, self.latent_dim*self.categorical_dim, 8, 8, device=self.device)

        # decode shape from header
        if self.fixed_input_shape is not None:
            byte_ptr = 0
            batch_dim = self.fixed_input_shape[0]
            spatial_shape = self.fixed_input_shape[1:]
            spatial_dim = np.prod(spatial_shape)
        else:
            num_shape_dims = struct.unpack("B", byte_string[:1])[0]
            flat_shape = []
            byte_ptr = 1
            for _ in range(num_shape_dims):
                flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
                byte_ptr += 2
            batch_dim = flat_shape[0]
            spatial_shape = flat_shape[1:]
            spatial_dim = np.prod(spatial_shape)

        if self.use_autoregressive_prior:
            if self.ar_method.startswith("maskconv"):
                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                samples = torch.zeros(batch_dim, self.latent_dim, *spatial_shape, dtype=torch.long, device=self.device)

                assert len(spatial_shape) == 2
                if self.ar_method.startswith("maskconv3d"):
                    c, h, w = (self.latent_dim, *spatial_shape)
                    for c_idx in range(c):
                        for h_idx in range(h):
                            for w_idx in range(w):
                                ar_input = F.one_hot(samples, self.categorical_dim).float().movedim(-1, 1)
                                prior_logits_ar = self.ar_model(ar_input).movedim(1, -1)[:, c_idx, h_idx, w_idx]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device)
                                samples[:, c_idx, h_idx, w_idx] = samples_ar
                else:
                    h, w = spatial_shape
                    for h_idx in range(h):
                        for w_idx in range(w):
                                ar_input = F.one_hot(samples, self.categorical_dim).float().movedim(-1, 2).reshape(batch_dim, self.latent_channels, *spatial_shape)
                                prior_logits_ar = self.ar_model(ar_input).reshape(batch_dim, self.latent_dim, self.categorical_dim, *spatial_shape).movedim(2, -1)[:, :, h_idx, w_idx]
                                prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                                cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                                cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                                indexes_ar = np.arange(len(cdfs_ar), dtype=np.int32)
                                cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                                offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                                samples_ar = rans_decoder.decode_stream_np(
                                    indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                                )
                                samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device).reshape(-1, self.latent_dim)
                                samples[:, :, h_idx, w_idx] = samples_ar

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.latent_dim*self.categorical_dim)\
                    .movedim(-1, 1)

                return samples

            elif self.ar_method.startswith("checkerboard"):
                assert len(spatial_shape) == 2
                checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=self.device)
                checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=self.device)
                checkerboard_index_h_00, checkerboard_index_w_00 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_0)
                checkerboard_index_h_11, checkerboard_index_w_11 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_1)
                checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)

                rans_decoder = RansDecoder()
                rans_decoder.set_stream(byte_string[byte_ptr:])
                indexes_base = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
                    .repeat(batch_dim, 1, spatial_dim // 2).reshape(batch_dim, self.latent_dim, spatial_shape[0] // 2, spatial_shape[1])\
                    .numpy()

                # prepare for coding
                indexes_base = indexes_base.astype(np.int32).reshape(-1)
                cdfs_base = self._prior_cdfs
                cdf_sizes_base = np.array([len(cdf) for cdf in self._prior_cdfs])
                offsets_base = np.zeros(len(self._prior_cdfs))

                samples = torch.zeros(batch_dim, self.latent_dim, *spatial_shape, dtype=torch.long, device=self.device)
                with self.profiler.start_time_profile("time_rans_decoder"):
                    samples_base = rans_decoder.decode_stream_np(
                        indexes_base, cdfs_base, cdf_sizes_base, offsets_base
                    )
                    samples_base = torch.as_tensor(samples_base, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.latent_dim, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_01, checkerboard_index_w_01] = samples_base[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_10, checkerboard_index_w_10] = samples_base[..., (spatial_shape[-1]//2):]
                    ar_input = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
                    ar_input = ar_input.reshape(batch_dim, *spatial_shape, self.latent_dim*self.categorical_dim)\
                        .movedim(-1, 1)
                    
                    prior_logits_reshape = self.ar_model(ar_input)
                    prior_logits_ar = torch.cat([
                        prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0],
                        prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1],
                    ], dim=-1)
                    prior_logits_ar = prior_logits_ar.reshape(batch_dim, self.latent_dim, self.categorical_dim, *prior_logits_ar.shape[-2:]).movedim(2, -1)
                    
                    # prepare for coding (ar)
                    # NOTE: coding may be unstable on GPU!
                    prior_probs_ar = torch.softmax(prior_logits_ar, dim=-1)
                    cdfs_ar = pmf_to_quantized_cdf_batched(prior_probs_ar.reshape(-1, prior_probs_ar.shape[-1]))
                    cdfs_ar = cdfs_ar.detach().cpu().numpy().astype(np.int32)
                    data_length = samples[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_0].numel() + samples[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_1].numel()
                    indexes_ar = np.arange(data_length, dtype=np.int32)
                    cdf_sizes_ar = np.array([len(cdf) for cdf in cdfs_ar])
                    offsets_ar = np.zeros(len(indexes_ar)) # [0] * len(indexes)

                    samples_ar = rans_decoder.decode_stream_np(
                        indexes_ar, cdfs_ar, cdf_sizes_ar, offsets_ar
                    )
                    samples_ar = torch.as_tensor(samples_ar, dtype=torch.long, device=self.device)\
                        .reshape(batch_dim, self.latent_dim, spatial_shape[0] // 2, spatial_shape[1])
                    samples[..., checkerboard_index_h_00, checkerboard_index_w_00] = samples_ar[..., :(spatial_shape[-1]//2)]
                    samples[..., checkerboard_index_h_11, checkerboard_index_w_11] = samples_ar[..., (spatial_shape[-1]//2):]

                # warn about decoding error and fixit!
                if samples.max() >= self.categorical_dim or samples.min() < 0:
                    print("Decode error detected! The decompressed data may be corrupted!")
                    samples.clamp_max_(self.categorical_dim-1).clamp_min_(0)
                samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
                samples = samples.reshape(batch_dim, *spatial_shape, self.latent_dim*self.categorical_dim)\
                    .movedim(-1, 1)

                return samples

            else:
                pass

        # TODO: use iterative autoregressive for overwhelmed states
        if self.use_autoregressive_prior and self.ar_method == "finitestate" and len(self.ar_offsets) > 2:
            raise NotImplementedError("Overwhelmed states!")

        if self.coder_type == "rans":
            # TODO: autoregressive rans
            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                raise NotImplementedError()
            rans_decoder = RansDecoder()
            indexes = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
                .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.latent_dim, *spatial_shape)\
                .numpy()

            # prepare for coding
            indexes = indexes.astype(np.int32).reshape(-1)
            cdfs = self._prior_cdfs
            cdf_sizes = np.array([len(cdf) for cdf in self._prior_cdfs])
            offsets = np.zeros(len(self._prior_cdfs))
            with self.profiler.start_time_profile("time_rans_decoder"):
                samples = rans_decoder.decode_with_indexes_np(
                    byte_string[byte_ptr:], indexes, cdfs, cdf_sizes, offsets
                )
        elif self.coder_type == "tans":
            assert TANS_AVAILABLE
            assert self.categorical_dim <= 256
            tans_decoder = TansDecoder()
            indexes = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
                .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.latent_dim, *spatial_shape)\
                .numpy()

            # prepare for coding
            encoded = byte_string[byte_ptr:]
            indexes = indexes.astype(np.int32) # .reshape(-1)
            dtables = self._prior_dtables
            offsets = np.zeros(len(self._prior_dtables))
            with self.profiler.start_time_profile("time_tans_decoder"):
                if self.use_autoregressive_prior and self.ar_method == "finitestate":
                    # samples = tans_decoder.decode_autoregressive_np(
                    #     encoded, indexes, np.array(self.ar_offsets), dtables, offsets, self.categorical_dim
                    # )
                    samples = self._decoder.decode_autoregressive_np(
                        encoded, indexes, np.array(self.ar_offsets)
                    )
                else:
                    samples = tans_decoder.decode_with_indexes_np(
                        encoded, indexes, dtables, offsets
                    )

        samples = torch.as_tensor(samples, dtype=torch.long, device=self.device)\
            .reshape(batch_dim, self.latent_dim, *spatial_shape)
        # assert (samples == self._samples_cache).all()

        # cat_reduce transform
        if self.cat_reduce:
            samples = self._reduce_mask[samples]
            # if self.cat_reduce_method == "softminus":
            #     raise NotImplementedError()
            # elif self.cat_reduce_method == "sigmoid":
            #     if not self.cat_reduce_channel_same:
            #         # TODO: different transformation for different channels
            #         raise NotImplementedError()
            #     else:
            #         reduce_mask = (self.cat_reduce_logprob[0] > 0).nonzero(as_tuple=False).squeeze(-1)
            #         # samples_full = torch.zeros(*samples.shape, self.categorical_dim).type_as(self.cat_reduce_logprob)
            #         samples = reduce_mask[samples]

        # merge categorical dim back to latent dim
        samples = F.one_hot(samples.movedim(1, -1), self.categorical_dim).float()
        samples = samples.reshape(batch_dim, *spatial_shape, self.latent_dim*self.categorical_dim)\
            .movedim(-1, 1)

        return samples

    def update_state(self, *args, **kwargs) -> None:
        with torch.no_grad():
            if self.prior_trainable:
                if self.use_autoregressive_prior and self.ar_method == "finitestate" and self.ar_fs_method == "table":
                    prior_logits = self._normalize_prior_logits(self.prior_logprob.transpose(0,1)).transpose(0,1)
                else:
                    prior_logits = self._normalize_prior_logits(self.prior_logprob)#.unsqueeze(-1)
            else:
                prior_logits = (torch.ones(self.latent_dim, self.categorical_dim) / self.categorical_dim).log()
            
            categorical_dim = self.categorical_dim # cat reduce moved after fsar
            if self.use_autoregressive_prior and self.ar_method == "finitestate":
                # TODO: this is a hard limit! may could be improved!
                if len(self.ar_offsets) > 2:
                    pass
                else:
                    if self.ar_fs_method == "table":
                        lookup_table_shape = [self.latent_dim] + [categorical_dim] * len(self.ar_offsets) + [categorical_dim]
                        if self.ar_prior_decomp_dim is None:
                            prior_logits = prior_logits.reshape(*lookup_table_shape)
                        else:
                            # normalize logits to 0 mean
                            prior_logits = prior_logits - prior_logits.mean(-1, keepdim=True)
                            # prior_logits = torch.log_softmax(prior_logits, dim=-1)#.unsqueeze(0)
                            prior_logits_ar = prior_logits.reshape(self.latent_dim, self.ar_window_size, self.ar_prior_decomp_dim, categorical_dim, categorical_dim)
                            cur_ar_idx = torch.zeros(len(self.ar_offsets), dtype=torch.long) # [0] * len(self.ar_offsets)
                            ar_dim_idx = torch.arange(len(self.ar_offsets))
                            # ar_idx_all = [cur_ar_idx]
                            prior_logits_ar_all = [prior_logits_ar[..., ar_dim_idx, :, cur_ar_idx, :]]
                            while True:
                                all_reset = True
                                for ar_idx in range(len(self.ar_offsets)):
                                    cur_ar_idx[ar_idx] += 1
                                    if cur_ar_idx[ar_idx] < categorical_dim:
                                        all_reset = False
                                        break
                                    else:
                                        cur_ar_idx[ar_idx] = 0
                                if all_reset: break
                                # ar_idx_all.append(copy.deepcopy(cur_ar_idx))
                                prior_logits_ar_all.append(prior_logits_ar[..., ar_dim_idx, :, cur_ar_idx, :])
                            # ar_idx_all = torch.as_tensor(ar_idx_all, dtype=torch.long).unsqueeze(-1).unsqueeze(-1)
                            # prior_logits_ar = prior_logits_ar[ar_idx_all]
                            prior_logits_ar_all = torch.stack(prior_logits_ar_all).transpose(1, 2)
                            prior_logits = self._merge_prior_logits_ar(prior_logits_ar_all)
                            prior_logits = self._normalize_prior_logits(prior_logits)
                            prior_logits = prior_logits.reshape(*lookup_table_shape)
                    elif self.ar_fs_method == "MLP3":
                        lookup_table_shape = [self.latent_dim] + [categorical_dim+1] * len(self.ar_offsets) + [categorical_dim]
                        ar_idx_all = list(itertools.product(range(self.categorical_dim+1), repeat=self.ar_window_size))
                        ar_idx_all = torch.tensor(ar_idx_all, device=self.device).reshape(-1)
                        ar_input_all = F.one_hot(ar_idx_all).type_as(prior_logits).reshape(-1, self.ar_window_size*(self.categorical_dim+1))
                        if self.ar_mlp_per_channel:
                            ar_logits_reshape = torch.stack([mlp(ar_input_all) for mlp in self.fsar_mlps_per_channel], dim=0)
                        else:
                            ar_logits_reshape = self.fsar_mlp(ar_input_all)
                        prior_logits = prior_logits.unsqueeze(-2) + ar_logits_reshape
                        prior_logits = self._normalize_prior_logits(prior_logits)
                        prior_logits = prior_logits.reshape(*lookup_table_shape)

            prior_pmfs = None

            if self.cat_reduce:
                if self.cat_reduce_method == "softminus":
                    # raise NotImplementedError()
                    pass
                elif self.cat_reduce_method == "sigmoid":
                    if not self.cat_reduce_channel_same:
                        # TODO: different transformation for different channels
                        # raise NotImplementedError()
                        pass
                    else:
                        reduce_mask = (self.cat_reduce_logprob[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                        categorical_dim = reduce_mask.shape[0]
                        prior_logits_reduced = prior_logits
                        if self.use_autoregressive_prior and self.ar_method == "finitestate":
                            # if self.ar_prior_decomp_dim is None:
                            #     prior_logits = prior_logits.reshape(self.latent_dim, self.ar_window_size, self.categorical_dim, self.categorical_dim)
                            # else:
                            #     prior_logits = prior_logits.reshape(self.latent_dim, self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                            # prior_logits = prior_logits.index_select(-2, reduce_mask)
                            for i in range(-1, -len(self.ar_offsets)-1, -1):
                                prior_logits_reduced = prior_logits_reduced.index_select(i, reduce_mask)
                        else:
                            prior_logits = self.cat_reduce_logit_thres * torch.sigmoid(prior_logits).index_select(-1, reduce_mask)
                        prior_logits = torch.log_softmax(prior_logits_reduced, dim=-1)
                        prior_pmfs = prior_logits.exp()
                        # self._reduce_mask = reduce_mask
                        self.register_buffer("_reduce_mask", reduce_mask, persistent=False)
                elif self.cat_reduce_method == "entmax":
                    if not self.cat_reduce_channel_same:
                        # TODO: different transformation for different channels
                        # raise NotImplementedError()
                        pass
                    else:
                        reduce_mask = (self._get_entmax_probs()[0] > 0).nonzero(as_tuple=False).squeeze(-1)
                        categorical_dim = reduce_mask.shape[0]
                        prior_logits_reduced = prior_logits
                        if self.use_autoregressive_prior and self.ar_method == "finitestate":
                            # if self.ar_prior_decomp_dim is None:
                            #     prior_logits = prior_logits.reshape(self.latent_dim, self.ar_window_size, self.categorical_dim, self.categorical_dim)
                            # else:
                            #     prior_logits = prior_logits.reshape(self.latent_dim, self.ar_window_size, self.ar_prior_decomp_dim, self.categorical_dim, self.categorical_dim)
                            # prior_logits = prior_logits.index_select(-2, reduce_mask)
                            for i in range(-1, -len(self.ar_offsets)-1, -1):
                                prior_logits_reduced = prior_logits_reduced.index_select(i, reduce_mask)
                        else:
                            prior_logits = self.cat_reduce_logit_thres * torch.sigmoid(prior_logits).index_select(-1, reduce_mask)
                        prior_logits = torch.log_softmax(prior_logits_reduced, dim=-1)
                        # self._reduce_mask = reduce_mask
                        self.register_buffer("_reduce_mask", reduce_mask, persistent=False)

            if prior_pmfs is None:
                prior_pmfs = prior_logits.exp()
            if self.coder_type == "rans":
                self._prior_cdfs = pmf_to_quantized_cdf_serial(prior_pmfs.reshape(-1, categorical_dim))
                # self._prior_cdfs = np.array([pmf_to_quantized_cdf(pmf.tolist()) for pmf in prior_pmfs.reshape(-1, categorical_dim)])
                # TODO: rans fsar?
            elif self.coder_type == "tans":
                assert TANS_AVAILABLE
                assert categorical_dim <= 256
                prior_cnt = prior_pmfs * (1<<10) # default tablelog is 10
                prior_cnt = prior_cnt.clamp_min(1).detach().cpu().numpy().astype(np.int32)
                self._prior_ctables = create_ctable_using_cnt(prior_cnt, maxSymbolValue=(categorical_dim-1))
                self._prior_dtables = create_dtable_using_cnt(prior_cnt, maxSymbolValue=(categorical_dim-1))
                self._encoder = TansEncoder()
                self._encoder.create_ctable_using_cnt(prior_cnt, maxSymbolValue=(categorical_dim-1))
                self._decoder = TansDecoder()
                self._decoder.create_dtable_using_cnt(prior_cnt, maxSymbolValue=(categorical_dim-1))
            else:
                raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")


class EmbeddingCategoricalDistributionPriorCoder(CategoricalDistributionPriorCoder):
    def __init__(self, latent_dim=8, categorical_dim=128, embedding_dim=32, 
        one_hot_initialization=False, embedding_init_method="uniform",
        fix_embedding=False, 
        embedding_variance=0.0, embedding_variance_trainable=False, embedding_variance_lr_modifier=1.0,
        use_embedding_variance_entropy=False,
        adjust_embedding_sample=False,
        var_scale=1.0, var_scale_anneal=False,
        **kwargs):
        if one_hot_initialization:
            embedding_dim = categorical_dim
        self.embedding_dim = embedding_dim
        self.adjust_embedding_sample = adjust_embedding_sample
        super().__init__(latent_dim=latent_dim, categorical_dim=categorical_dim, **kwargs)

        # TODO: autoregressive embedding
        embedding = torch.zeros(latent_dim, categorical_dim, embedding_dim)
        if one_hot_initialization:
            self.embedding_dim = categorical_dim # force embedding dim equal to categorical_dim
            embedding = torch.eye(categorical_dim).unsqueeze(0).repeat(latent_dim, 1, 1)
        else:
            if embedding_init_method == "normal":
                nn.init.normal_(embedding)
            elif embedding_init_method == "position":
                position_cos = torch.cos(
                    torch.arange(self.categorical_dim * self.embedding_dim) / self.categorical_dim * 2 * np.pi)
                embedding[:] = position_cos.reshape(categorical_dim, embedding_dim).unsqueeze(-1)
            else:
                nn.init.uniform_(embedding, -1, 1)
        
        if fix_embedding:
            self.register_buffer("embedding", embedding)
        else:
            self.embedding = nn.Parameter(embedding)

        if adjust_embedding_sample:
            self.embedding_adjustor = nn.Sequential(
                nn.Linear(categorical_dim*(embedding_dim+1), 2*categorical_dim*embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(2*categorical_dim*embedding_dim, categorical_dim*embedding_dim),
            )

        self.embedding_variance_trainable = embedding_variance_trainable
        self.use_embedding_variance_entropy = use_embedding_variance_entropy
        if embedding_variance > 0:
            embedding_variance = torch.ones_like(self.embedding) * embedding_variance
            if embedding_variance_trainable:
                self.embedding_variance = nn.Parameter(embedding_variance)
                self.embedding_variance.lr_modifier = embedding_variance_lr_modifier
            else:
                self.register_buffer("embedding_variance", embedding_variance)
        else:
            self.embedding_variance = None

        self.var_scale_anneal = var_scale_anneal
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.var_scale = var_scale


    @property
    def latent_channels_out(self):
        return self.latent_dim * self.embedding_dim

    # def sample_from_posterior(self, posterior_dist: distributions.Distribution):
    #     output = super().sample_from_posterior(posterior_dist)
    
    def postprocess_samples(self, samples):
        output = samples
        output = output.view(-1, self.latent_dim, self.categorical_dim)
        samples = output.permute(1, 0, 2).reshape(self.latent_dim, -1, self.categorical_dim)
        if self.adjust_embedding_sample:
            samples_cat_n = samples.mean(dim=1)
            embedding = self.embedding.reshape(self.latent_dim, self.categorical_dim*self.embedding_dim)
            embedding = embedding + self.embedding_adjustor(torch.cat([embedding, samples_cat_n], dim=1))
            embedding = embedding.reshape(self.latent_dim, self.categorical_dim, self.embedding_dim)
        else:
            embedding = self.embedding

        if self.training and self.embedding_variance is not None:
            # embedding = torch.normal(embedding, self.embedding_variance)
            embedding = embedding + self.embedding_variance * self.var_scale * torch.normal(torch.zeros_like(embedding))
            if self.embedding_variance_trainable:
                if self.use_embedding_variance_entropy:
                    embedding_variance_entropy = self.entropy_temp * torch.log(self.embedding_variance)
                    self.update_cache("loss_dict",
                        embedding_variance_entropy=embedding_variance_entropy.mean(),
                    )
                self.update_cache("moniter_dict",
                    embedding_variance_mean = self.embedding_variance.mean(),
                )
        embedding_samples = torch.bmm(samples, embedding)
        return embedding_samples.permute(1, 0, 2).reshape(-1, self.latent_channels_out)


class EmbeddingStandardNormalPriorCategoricalDistributionPriorCoder(EmbeddingCategoricalDistributionPriorCoder):
    def __init__(self, normalize_embedding_prob=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize_embedding_prob = normalize_embedding_prob

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, **kwargs):
        cat_entropy = posterior_dist.probs * posterior_dist.logits

        if self.normalize_embedding_prob:
            embedding_log_prob = -0.5 * (self.embedding.pow(2) + math.log(2*math.pi)).sum(-1)
            embedding_log_prob = torch.log_softmax(embedding_log_prob, dim=-1)
            prior_entropy = posterior_dist.probs * embedding_log_prob.unsqueeze(0)
        else:
            prior_entropy = posterior_dist.probs * -0.5 * self.embedding.pow(2).sum(-1).unsqueeze(0)
        
        if self.normalize_embedding_prob:
            # use normalized prob as prior entropy
            self.update_cache("metric_dict",
                prior_entropy=-prior_entropy.sum() / input_shape[0],
            )
        else:
            # otherwise use default CatVAE ones
            self.update_cache("metric_dict",
                prior_entropy=-(posterior_dist.probs * prior_dist.logits.unsqueeze(0)).sum() / input_shape[0],
            )
        return cat_entropy - prior_entropy


class EmbeddingGaussianPriorCategoricalDistributionPriorCoder(EmbeddingCategoricalDistributionPriorCoder):
    def __init__(self, 
        gaussian_mixture_dim=None, 
        global_latent_normalizer=1./60000, 
        normalize_embedding_prob=True,
        **kwargs):
        """_summary_

        Args:
            gaussian_mixture_dim (int, optional): GMM components. Defaults to None, same as categorical_dim.
            global_latent_normalizer (float, optional): should usually be (1 / size of the dataset). Defaults to (1./60000) (CIFAR10 size).
        """
        super().__init__(**kwargs)

        if gaussian_mixture_dim is None:
            gaussian_mixture_dim = self.categorical_dim
        self.gaussian_mixture_dim = gaussian_mixture_dim
        self.global_latent_normalizer = global_latent_normalizer
        self.normalize_embedding_prob = normalize_embedding_prob
        
        self.embedding_mean = nn.Parameter(torch.zeros(self.latent_dim, self.gaussian_mixture_dim, self.embedding_dim))
        nn.init.uniform_(self.embedding_mean, -1, 1)
        self.embedding_logvar = nn.Parameter(torch.zeros(self.latent_dim, self.gaussian_mixture_dim, self.embedding_dim))
        nn.init.uniform_(self.embedding_logvar, -1, 1)
        if self.gaussian_mixture_dim != self.categorical_dim:
            self.prior_logprob.data = torch.zeros(self.latent_dim, self.gaussian_mixture_dim)

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, **kwargs):
        batch_size = posterior_dist.logits.shape[0]

        kl_gaussian = -0.5 * (1 + self.embedding_logvar - self.embedding_mean ** 2 - self.embedding_logvar.exp())
        kl_gaussian = kl_gaussian.sum() * batch_size * self.global_latent_normalizer

        means_repeat = self.embedding_mean.unsqueeze(1)
        scales_repeat = torch.exp(0.5 * self.embedding_logvar).unsqueeze(1)
        gaussian_dist = distributions.LowRankMultivariateNormal(means_repeat, torch.zeros_like(scales_repeat).unsqueeze(-1), scales_repeat)
        
        cat_entropy = posterior_dist.probs * posterior_dist.logits
        embedding_log_probs_all = gaussian_dist.log_prob(self.embedding.unsqueeze(2))
        # embedding_logits_norm_all = torch.log_softmax(embedding_log_probs_all, dim=-1)
        embedding_logits_all = torch.logsumexp(embedding_log_probs_all + prior_dist.logits.unsqueeze(1), dim=2)
        embedding_logits_norm = torch.log_softmax(embedding_logits_all, dim=1)
        if self.normalize_embedding_prob:
            embedding_logits = embedding_logits_norm
        else:
            embedding_logits = embedding_logits_all # (embedding_log_probs_all.exp() * prior_dist.logits.unsqueeze(1)).sum(-1)
        cat_cross_entropy = posterior_dist.probs * embedding_logits.unsqueeze(0)
        if self.normalize_embedding_prob:
            prior_entropy = cat_cross_entropy.sum()
        else:
            # TODO:
            prior_entropy = posterior_dist.probs * embedding_logits_norm.unsqueeze(0)
        # embedding_cat_prob_norm = torch.softmax(
        #     (posterior_dist.probs.unsqueeze(-1) * embedding_probs_all.unsqueeze(0)).sum(dim=2),
        #     dim=-1,
        # )
        # cat_cross_entropy = embedding_cat_prob_norm * prior_dist.logits.unsqueeze(0)
        kl_cat = (cat_entropy.sum() - cat_cross_entropy.sum())

        self.update_cache("metric_dict",
            kl_gaussian=kl_gaussian / input_shape[0],
            kl_cat=kl_cat / input_shape[0],
            prior_entropy=prior_entropy / input_shape[0],
        )

        return kl_gaussian + kl_cat


class StickBreakingPriorCategoricalDistributionPriorCoder(CategoricalDistributionPriorCoder):
    def __init__(self, prior_alpha0=5., **kwargs):
        super().__init__(**kwargs)
        self.prior_logprob.data[:] = np.log(np.exp(prior_alpha0) - 1) # inverse softplus
        self._resample_prior()

    # @property
    # def prior_beta(self):
    #     return F.softplus(self.prior_logprob)

    def _resample_prior(self):
        # TODO: use Beta or Kumaraswamy?
        dist = distributions.Beta(1., F.softplus(self.prior_logprob))
        samples = dist.rsample()
        self._prior_sb = torch.cumprod(1 - samples, dim=-1) * samples / (1 - samples)

    def prior_distribution(self, prior=None, **kwargs):
        # TODO: prior
        return distributions.Categorical(probs=self._prior_sb)

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        latent = latent.view(-1, self.latent_dim, self.categorical_dim) \
            + self._prior_sb.unsqueeze(0)
        return super().posterior_distribution(latent)

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        # resample prior before every iteration
        self._resample_prior()
        return super()._forward_flat(input, input_shape, prior=prior, **kwargs)


class StickBreakingGEMPriorCoder(DistributionPriorCoder):
    def __init__(self, in_channels=256, 
        latent_dim=8, categorical_dim=128,
        prior_trainable=False, prior_alpha0=5., 
        use_kumaraswamy_posterior=False, eps=1e-8,
        gs_temp=0.5, gs_temp_anneal=False,
        **kwargs):
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        latent_channels = latent_dim * self.num_posterior_params
        super().__init__(in_channels, latent_channels=latent_channels, **kwargs)

        self.prior_trainable = prior_trainable
        self.prior_alpha0 = prior_alpha0
        self.use_kumaraswamy_posterior = use_kumaraswamy_posterior
        self.eps = eps
        a_val = np.log(np.exp(prior_alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        if prior_trainable:
            self.prior_a = nn.Parameter(torch.zeros(latent_dim, categorical_dim) + a_val)
            self.prior_b = nn.Parameter(torch.zeros(latent_dim, categorical_dim) + b_val)
        else:
            self.register_buffer("prior_a", torch.zeros(latent_dim, categorical_dim) + a_val, persistent=False)
            self.register_buffer("prior_b", torch.zeros(latent_dim, categorical_dim) + b_val, persistent=False)

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

    @property
    def latent_channels_in(self):
        return self.latent_dim * self.num_posterior_params

    @property
    def latent_channels_out(self):
        return self.latent_dim * self.categorical_dim

    @property
    def num_posterior_params(self):
        return self.categorical_dim * 2

    def prior_distribution(self, prior=None, **kwargs):
        # TODO: prior
        return distributions.Beta(F.softplus(self.prior_a), F.softplus(self.prior_b))

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        latent = latent.view(-1, self.latent_dim, self.categorical_dim * 2)
        post_a, post_b = latent.chunk(2, dim=-1)
        if self.use_kumaraswamy_posterior:
            return Kumaraswamy(F.softplus(post_a) + 0.01, F.softplus(post_b) + 0.01)
        else:
            return distributions.Beta(F.softplus(post_a) + 0.01, F.softplus(post_b) + 0.01)

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, **kwargs):
        if self.use_kumaraswamy_posterior:
            return posterior_dist.kl_beta(prior_dist) # torch.zeros_like(self.prior_a)
        else:
            return distributions.kl_divergence(posterior_dist, prior_dist)

    def sample_from_posterior(self, posterior_dist: distributions.Distribution):
        output = super().sample_from_posterior(posterior_dist)
        # SBP
        # sb_matrix = 1 - torch.tril(output.unsqueeze(-1).repeat(1, 1, 1, output.shape[-1]))
        # diag_idx = torch.arange(output.shape[-1], dtype=output.dtype).long()
        # sb_matrix[:, :, diag_idx, diag_idx] = output[:, :, diag_idx]
        # sb_cat = torch.prod(sb_matrix, dim=-2)
        sb_cat = torch.cumprod(1 - output + self.eps, dim=-1) * output / (1 - output + self.eps)

        output = torch.softmax((sb_cat + self.eps).log() / self.gs_temp, dim=-1)

        return output.view(output.shape[0], self.latent_channels_out)

# rewritten from https://github.com/rachtsingh/ibp_vae/blob/master/src/models/S_IBP_Concrete.py
# TODO: nan occurs during training
class BetaBernoulliGaussianPriorCoder(DistributionPriorCoder):
    def __init__(self, in_channels=256, 
        latent_dim=8, truncate_dim=128,
        prior_trainable=False, prior_alpha0=5., use_ibp=True,
        dataset_size=60000,
        gs_temp=0.5, gs_temp_anneal=False,
        eps=1e-6,
        **kwargs):
        self.latent_dim = latent_dim
        self.truncate_dim = truncate_dim
        latent_channels = latent_dim
        super().__init__(in_channels, latent_channels=latent_channels, **kwargs)

        self.prior_trainable = prior_trainable
        self.prior_alpha0 = prior_alpha0
        self.dataset_size = dataset_size

        a_val = np.log(np.exp(prior_alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        if prior_trainable:
            self.prior_a = nn.Parameter(torch.zeros(latent_dim, truncate_dim) + a_val)
            self.prior_b = nn.Parameter(torch.zeros(latent_dim, truncate_dim) + b_val)
        else:
            self.register_buffer("prior_a", torch.zeros(latent_dim, truncate_dim) + a_val, persistent=False)
            self.register_buffer("prior_b", torch.zeros(latent_dim, truncate_dim) + b_val, persistent=False)

        self.use_ibp = use_ibp

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.register_buffer("gs_temp", torch.tensor(gs_temp), persistent=False)

        self.eps = eps

        # cache
        self._prior_logits = None
        self._bernoulli_samples = None

    @property
    def num_posterior_params(self):
        return self.truncate_dim * 3

    @property
    def latent_channels_out(self):
        return self.latent_dim * self.truncate_dim

    def prior_distribution(self, prior=None, **kwargs):
        # TODO: prior
        # NOTE: use kumaraswamy?
        # return distributions.Beta(F.softplus(self.prior_a) + 0.01, F.softplus(self.prior_b) + 0.01)
        return Kumaraswamy(F.softplus(self.prior_b) + 0.01, F.softplus(self.prior_a) + 0.01)

    def posterior_distribution(self, latent) -> distributions.Distribution:
        latent = latent.view(-1, self.latent_dim, self.num_posterior_params)
        logit_x, mu, logvar = latent.chunk(3, dim=-1)
        # prior_prob = self._prior_samples
        logit_prior = self._prior_logits
        logit_post = logit_x + logit_prior
        dist_discrete = distributions.RelaxedBernoulli(self.gs_temp, logits=logit_post)
        dist_cont = distributions.Normal(mu, torch.exp(0.5 * logvar))
        return [dist_discrete, dist_cont]

    def kl_divergence(self, prior_dist : distributions.Distribution, posterior_dist : distributions.Distribution, input_shape : torch.Size = None, posterior_samples=None, **kwargs):
        dist_discrete, dist_cont = posterior_dist
        kl_gaussian = distributions.kl_divergence(
            dist_cont, 
            distributions.Normal(0, 1),
        ).sum()

        # kl_beta = distributions.kl_divergence(
        #     prior_dist, 
        #     distributions.Beta(self.prior_alpha0, 1.)
        #     # distributions.Beta(torch.ones_like(self.prior_a), torch.ones_like(self.prior_b) * self.prior_alpha0)
        # ).sum()
        kl_beta = prior_dist.kl_beta(distributions.Beta(1., self.prior_alpha0)).sum()
        # global latent should div dataset_size
        kl_beta = kl_beta * (input_shape[0] / self.dataset_size)
        
        # relaxed_probs = F.sigmoid(dist_discrete.logits / self.gs_temp)
        # relaxed_logits = dist_discrete.logits #/ self.gs_temp
        logit_prior = self._prior_logits
        # using rsample kl may be more stable
        # kl_bernoulli = distributions.kl_divergence(
        #     distributions.Bernoulli(logits=relaxed_logits),
        #     distributions.Bernoulli(logits=logit_prior),
        # ).sum()
        # kl_bernoulli = super().kl_divergence(
        #     distributions.RelaxedBernoulli(self.gs_temp, logits=logit_prior),
        #     dist_discrete,
        #     # posterior_samples=self._bernoulli_samples,
        # ).sum()
        pi_prior = torch.sigmoid(logit_prior)
        pi_posterior = dist_discrete.probs
        kl_1 = self._bernoulli_samples * (pi_posterior + self.eps).log() + (1 - self._bernoulli_samples) * (1 - pi_posterior + self.eps).log()
        kl_2 = self._bernoulli_samples * (pi_prior + self.eps).log() + (1 - self._bernoulli_samples) * (1 - pi_prior + self.eps).log()
        kl_bernoulli = (kl_1 - kl_2).sum()

        self.update_cache("metric_dict",
            kl_gaussian=kl_gaussian / input_shape[0],
            kl_beta=kl_beta / input_shape[0],
            kl_bernoulli=kl_bernoulli / input_shape[0],
            prior_entropy=(kl_gaussian+kl_bernoulli) / input_shape[0], # kl_beta is global latent
        )
        return kl_gaussian + kl_beta + kl_bernoulli

    def _resample_prior(self, batch_shape):
        samples = self.prior_distribution().rsample([batch_shape])
        log_samples = (samples + self.eps).log()
        if self.use_ibp:
            prior_log_samples = torch.cumsum(log_samples, dim=-1)
        else:
            # stick breaking process
            log1p_samples = (-samples).log1p()
            prior_log_samples = torch.cumsum(log1p_samples, dim=-1) + log_samples - log1p_samples

        self._prior_logits = (prior_log_samples.exp() + self.eps).log() - (-prior_log_samples.exp() + self.eps).log1p()

    def sample_from_posterior(self, posterior_dist: distributions.Distribution):
        dist_discrete, dist_cont = posterior_dist
        z_discrete = dist_discrete.rsample()
        self._bernoulli_samples = z_discrete # cache samples for calculating kl_divergence
        # zero-temperature rounding
        if not self.training:
            z_discrete = torch.round(z_discrete)
        z_continuous = dist_cont.rsample()
        return (z_discrete * z_continuous).view(-1, self.latent_channels_out)

    def _forward_flat(self, input : torch.Tensor, input_shape : torch.Size, prior : torch.Tensor = None, **kwargs):
        # resample prior before every iteration
        self._resample_prior(input.shape[0])
        return super()._forward_flat(input, input_shape, prior=prior, **kwargs)


class GaussianC2DDistributionPriorCoder(ContinuousToDiscreteDistributionPriorCoder):
    @property
    def num_continuous_posterior_params(self):
        return 2

    def continuous_prior_distribution(self):
        return distributions.Normal(0, 1)

    def continuous_posterior_distribution(self, latent) -> distributions.Distribution:
        mean, logvar = latent.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return distributions.Normal(mean, std)

    def continuous_loss(self, prior_dist: distributions.Distribution, posterior_dist: distributions.Distribution, input_shape: torch.Size, **kwargs):
        return distributions.kl_divergence(posterior_dist, prior_dist)


class GaussianC2VQDistributionPriorCoder(ContinuousToVQDistributionPriorCoder):
    @property
    def num_posterior_params(self):
        return 2

    def prior_distribution(self, prior=None, **kwargs):
        if prior is not None:
            return self.posterior_distribution(prior)
        return distributions.Normal(0, 1)

    def posterior_distribution(self, latent, **kwargs) -> distributions.Distribution:
        mean, logvar = latent.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return distributions.Normal(mean, std)

    def continuous_loss(self, prior_dist: distributions.Distribution, posterior_dist: distributions.Distribution, input_shape: torch.Size, **kwargs):
        return distributions.kl_divergence(posterior_dist, prior_dist)

# class VQPriorCoder(NNPriorCoder):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
#         super().__init__()
        
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
        
#         self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
#         self._commitment_cost = commitment_cost

#     def forward(self, inputs) -> torch.Tensor:
#         # convert inputs from BCHW -> BHWC
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape
        
#         # Flatten input
#         flat_input = inputs.view(-1, self._embedding_dim)
        
#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
#                     + torch.sum(self._embedding.weight**2, dim=1)
#                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
        
#         # Quantize and unflatten
#         quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         # loss = q_latent_loss + self._commitment_cost * e_latent_loss
#         loss = dict(
#             q_latent_loss = q_latent_loss,
#             e_latent_loss = self._commitment_cost * e_latent_loss,
#         )
#         self.update_cache("loss_dict", 
#             **loss
#         )

#         quantized = inputs + (quantized - inputs).detach()
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         self.update_cache("metric_dict", 
#             perplexity=perplexity.sum()
#         )

#         # convert quantized from BHWC -> BCHW
#         return quantized.permute(0, 3, 1, 2).contiguous()

# class MultiChannelVQPriorCoder(NNPriorCoder):
#     def __init__(self, *args, latent_dim=8, num_embeddings=128, embedding_dim=32, **kwargs):
#         super().__init__()
#         self.coder = VQEmbeddingGSSoft(*args, latent_dim=latent_dim, num_embeddings=num_embeddings, embedding_dim=embedding_dim, **kwargs)

#     def forward(self, x):
#         loss, quantized = self.coder(x)
#         self.update_cache("loss_dict", **loss)
#         return quantized


# gumbel-softmax training vq from https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
class MultiChannelVQPriorCoder(NNPriorCoder):
    def __init__(self, latent_dim=8, num_embeddings=128, embedding_dim=32,
                 channels_share_codebook=False,
                 # smoothing
                 input_variance=0.0, input_variance_trainable=False,
                 # embedding
                 embedding_variance=0.0, embedding_variance_per_dimension=False,
                 embedding_variance_trainable=True, embedding_variance_lr_modifier=1.0,
                 # misc
                 dist_type=None, # RelaxedOneHotCategorical, AsymptoticRelaxedOneHotCategorical, DoubleRelaxedOneHotCategorical
                 force_use_straight_through=False, st_weight=1.0,
                 # coding
                 coder_type="rans", # current support "rans", "tans"
                 coder_freq_precision=16,
                 fixed_input_shape=None,
                 # vamp
                 use_vamp_prior=False,
                 # code update
                 use_ema_update=False, ema_decay=0.999, ema_epsilon=1e-5, ema_reduce_ddp=True,
                 ema_adjust_sample=False,
                 embedding_lr_modifier=1.0,
                 # code prior
                 use_code_freq=False, code_freq_manual_update=False, update_code_freq_ema_decay=0.9,
                 use_code_variance=False,
                 # autoregressive prior
                 use_autoregressive_prior=False, 
                 ar_window_size=1, ar_offsets=None,
                 ar_method="finitestate", ar_mlp_per_channel=True,
                 ar_input_quantized=False,
                 ar_input_st_logits=False,
                #  ar_fs_method="table",
                 # autoregressive input (posterior)
                 use_autoregressive_posterior=False, autoregressive_posterior_method="maskconv3x3",
                 # loss
                 kl_cost=1.0, distance_detach_codebook=False,
                 use_st_gumbel=False, 
                 commitment_cost=0.25, commitment_over_exp=False, 
                 vq_cost=1.0, use_vq_loss_with_dist=False,
                 # testing
                 test_sampling=False, 
                 # init
                 initialization_mean=0.0, initialization_scale=None,
                 # monte-carlo sampling
                 train_mc_sampling=False, mc_loss_func=None, mc_sampling_size=64, mc_cost=1.0,
                 # annealing
                 relax_temp=1.0, relax_temp_anneal=False, gs_temp=0.5, gs_temp_anneal=False, 
                 entropy_temp=1.0, entropy_temp_min=1.0, entropy_temp_threshold=0.0, entropy_temp_anneal=False, 
                 use_st_below_entropy_threshold=False, use_vq_loss_below_entropy_threshold=False, use_commit_loss_below_entropy_threshold=False,
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.categorical_dim = num_embeddings # alias
        self.embedding_dim = embedding_dim
        self.channels_share_codebook = channels_share_codebook

        self.coder_type = coder_type
        self.coder_freq_precision = coder_freq_precision

        self.input_variance = input_variance
        self.input_variance_trainable = input_variance_trainable
        if input_variance_trainable and input_variance > 0:
            self.input_variance = nn.Parameter(torch.tensor([np.log(input_variance)]))

        self.use_vamp_prior = use_vamp_prior
        
        self.use_ema_update = use_ema_update
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.ema_reduce_ddp = ema_reduce_ddp
        self.ema_adjust_sample = ema_adjust_sample
        
        self.embedding_lr_modifier = embedding_lr_modifier
        embedding = torch.zeros(1 if self.channels_share_codebook else latent_dim, num_embeddings, embedding_dim)
        if initialization_scale is None:
            initialization_scale = 1/num_embeddings
        embedding.uniform_(initialization_mean-initialization_scale, initialization_mean+initialization_scale)
        if use_vamp_prior:
            # embedding should be set by set_vamp_posterior
            self.register_buffer("embedding", embedding)
        else:
            if use_ema_update:
                self.register_buffer("embedding", embedding)
                self.register_buffer("ema_count", torch.zeros(*self.embedding.shape[:-1]))
                self.register_buffer("ema_weight", self.embedding.clone())
                # if ema_adjust_sample:
                #     self.register_buffer("ema_weight_posterior", self.embedding.clone())
            else:
                self.embedding = nn.Parameter(embedding)
                self.embedding.lr_modifier = embedding_lr_modifier

        self.use_embedding_variance = (embedding_variance > 0)
        self.embedding_variance_trainable = embedding_variance_trainable
        self.embedding_variance_per_dimension = embedding_variance_per_dimension
        if self.use_embedding_variance:
            if self.embedding_variance_per_dimension:
                embedding_variance = torch.ones_like(self.embedding) * np.log(embedding_variance) # exponential reparameterization
            else:
                embedding_variance = torch.ones(1) * np.log(embedding_variance) # exponential reparameterization
            if embedding_variance_trainable:
                self.embedding_variance = nn.Parameter(embedding_variance)
                self.embedding_variance.lr_modifier = embedding_variance_lr_modifier
            else:
                self.register_buffer("embedding_variance", embedding_variance)

        self.use_code_freq = use_code_freq
        self.code_freq_manual_update = code_freq_manual_update
        self.update_code_freq_ema_decay = update_code_freq_ema_decay

        self.use_code_variance = use_code_variance

        self.use_autoregressive_prior = use_autoregressive_prior
        self.ar_window_size = ar_window_size
        self.ar_offsets = ar_offsets
        self.ar_method = ar_method
        self.ar_mlp_per_channel = ar_mlp_per_channel
        self.ar_input_quantized = ar_input_quantized
        self.ar_input_st_logits = ar_input_st_logits
        # self.ar_fs_method = ar_fs_method
        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_dim - 1
        # custom ar offset setting
        if self.ar_offsets is None:
            self.ar_offsets = [(-offset,) for offset in range(1, self.ar_window_size+1)]
        else:
            self.ar_window_size = len(ar_offsets)

        # full ar
        if self.ar_window_size is None:
            self.ar_window_size = self.latent_dim - 1
        if use_code_freq:
            prior_logprob = torch.zeros(*self.embedding.shape[:-1])

            # self.embedding_freq = nn.Parameter(torch.ones(latent_dim, num_embeddings) / num_embeddings)
            # if code_freq_manual_update:
            #     self.embedding_freq.requires_grad = False
            self.embedding_logprob = nn.Parameter(prior_logprob)
            if code_freq_manual_update:
                self.embedding_logprob.requires_grad = False
        else:
            self.register_buffer("embedding_logprob", 
                                 torch.zeros(*self.embedding.shape[:-1]) - math.log(self.num_embeddings), 
                                 persistent=False)
            
        if self.use_code_variance:
            embedding_logvar = torch.zeros(*self.embedding.shape[:-1])
            self.embedding_logvar = nn.Parameter(embedding_logvar)
        
        self.coder_type = coder_type
        # TODO: temp fix for no rans fsar impl! Remove this after fsar-rans is done!
        if self.use_autoregressive_prior and self.ar_method == "finitestate":
            if self.coder_type == "rans":
                print("Warning! rans fsar is not implemented! switching to tans!")
                self.coder_type = "tans"
        self.fixed_input_shape = fixed_input_shape

        if use_autoregressive_prior:
            if self.ar_input_quantized:
                ar_input_channels = self.embedding_dim
            else:
                ar_input_channels = self.categorical_dim + 1
            if self.ar_method == "finitestate":
                if self.ar_mlp_per_channel:
                    self.fsar_mlps_per_channel = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(ar_input_channels * self.ar_window_size, 2 * self.ar_window_size * ar_input_channels),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.ar_window_size * ar_input_channels, 2 * self.categorical_dim),
                                nn.LeakyReLU(),
                                nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                            )
                            for _ in range(self.latent_dim)
                        ]
                    )
                else:
                    self.fsar_mlp = nn.Sequential(
                        nn.Linear((self.categorical_dim + 1) * self.ar_window_size, 2 * self.ar_window_size * (self.categorical_dim + 1)),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.ar_window_size * (self.categorical_dim + 1), 2 * self.categorical_dim),
                        nn.LeakyReLU(),
                        nn.Linear(2 * self.categorical_dim, self.categorical_dim),
                    )

        # model based ar
        if self.use_autoregressive_prior:
            ar_model = None
            if self.ar_input_quantized:
                ar_input_channels = self.embedding_dim
            else:
                ar_input_channels = self.categorical_dim
            if self.ar_method == "maskconv3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 3, padding=1)
            elif self.ar_method == "maskconv5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 5, padding=2)
            elif self.ar_method == "maskconv3d3x3x3":
                ar_model = MaskedConv3d(ar_input_channels, self.categorical_dim, 3, padding=1)
            elif self.ar_method == "maskconv3d5x5x5":
                ar_model = MaskedConv3d(ar_input_channels, self.categorical_dim, 5, padding=2)
            elif self.ar_method == "checkerboard3x3":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 3, padding=1, mask_type="Checkerboard")
            elif self.ar_method == "checkerboard5x5":
                ar_model = MaskedConv2d(ar_input_channels * self.latent_dim, self.latent_dim * self.categorical_dim, 5, padding=2, mask_type="Checkerboard")

            if ar_model is not None:
                self.ar_model = nn.Sequential(
                    ar_model,
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 6 // 3, ar_input_channels * self.latent_dim * 5 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 5 // 3, ar_input_channels * self.latent_dim * 4 // 3, 1),
                    # nn.LeakyReLU(inplace=True),
                    # nn.Conv2d(ar_input_channels * self.latent_dim * 4 // 3, ar_input_channels * self.latent_dim * 3 // 3, 1),
                )

        self.use_autoregressive_posterior = use_autoregressive_posterior
        self.autoregressive_posterior_method = autoregressive_posterior_method
        if autoregressive_posterior_method == "maskconv3x3":
            self.input_ar_model = MaskedConv2d(self.embedding_dim * self.latent_dim, self.embedding_dim * self.latent_dim, 3, padding=1)
        
        self.dist_type = dist_type
        self.use_straight_through = (dist_type is None) or force_use_straight_through
        self.st_weight = st_weight
                
        self.kl_cost = kl_cost
        self.distance_detach_codebook = distance_detach_codebook
        self.use_st_gumbel = use_st_gumbel
        self.commitment_cost = commitment_cost
        self.commitment_over_exp = commitment_over_exp
        self.vq_cost = vq_cost
        self.use_vq_loss_with_dist = use_vq_loss_with_dist

        self.test_sampling = test_sampling

        self.train_mc_sampling = train_mc_sampling
        self.mc_loss_func = mc_loss_func
        self.mc_sampling_size = mc_sampling_size
        self.mc_cost = mc_cost

        self.relax_temp_anneal = relax_temp_anneal
        if relax_temp_anneal:
            self.relax_temp = nn.Parameter(torch.tensor(relax_temp), requires_grad=False)
        else:
            self.relax_temp = relax_temp

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        self.entropy_temp_anneal = entropy_temp_anneal
        if entropy_temp_anneal:
            self.entropy_temp = nn.Parameter(torch.tensor(entropy_temp), requires_grad=False)
            self.register_buffer("entropy_temp_min", torch.tensor(entropy_temp_min, requires_grad=False), persistent=False)
        else:
            self.entropy_temp = entropy_temp
            self.entropy_temp_min = entropy_temp_min
        self.entropy_temp_threshold = entropy_temp_threshold
        self.use_st_below_entropy_threshold = use_st_below_entropy_threshold
        self.use_vq_loss_below_entropy_threshold = use_vq_loss_below_entropy_threshold
        self.use_commit_loss_below_entropy_threshold = use_commit_loss_below_entropy_threshold

        # model state
        self.state_gs_perturb = True

        # initalize members for coding
        self.update_state()

    def _pairwise_distance(self, x1 : torch.Tensor, x2 : torch.Tensor, scale2 : torch.Tensor = None) -> torch.Tensor:
        """_summary_

        Args:
            x1 (torch.Tensor): Batch * Elements1 * Vector
            x2 (torch.Tensor): Batch * Elements2 * Vector
            scale2 (torch.Tensor): Batch * Elements2

        Returns:
            torch.Tensor): Batch * Elements1 * Elements2
        """        
        # dists = torch.baddbmm(torch.sum(x2 ** 2, dim=2).unsqueeze(1) +
        #                           torch.sum(x1 ** 2, dim=2, keepdim=True),
        #                           x1, x2.transpose(1, 2),
        #                           alpha=-2.0, beta=1.0) / x1.shape[-1]
        dists = (torch.sum(x1**2, dim=-1, keepdim=True) \
                + torch.sum(x2**2, dim=-1).unsqueeze(-2) \
                - 2 * torch.matmul(x1, x2.transpose(-2, -1))) / x1.shape[-1]
        if scale2 is not None:
            dists = dists / scale2.unsqueeze(1)
        return dists
    
    def _distance_loss(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        return self._pairwise_distance(x1, x2).mean()

    def _logits_from_distances(self, distances):
        # NOTE: the original code use the l2-sum distance!
        return -distances * self.embedding_dim

    def _sample_from_param(self, param) -> torch.Tensor:
        return param
        
    def _sample_from_embedding(self, samples, embedding=None) -> torch.Tensor:
        if embedding is None:
            embedding = self.embedding
        return torch.matmul(samples, embedding)

    def _calculate_kl_from_dist(self, dist : distributions.Distribution, prior_logits=None):
        # KL: N, B, spatial_dim, M
        entropy_temp = max(self.entropy_temp, self.entropy_temp_min)
        # KL = dist.probs * (dist.logits * entropy_temp - prior_logits)
        # KL[(dist.probs == 0).expand_as(KL)] = 0
        # KL = KL.mean(dim=1).sum() # mean on batch dim
        posterior_entropy = dist.probs * dist.logits
        posterior_entropy[dist.probs == 0] = 0 # prevent nan
        prior_entropy = dist.probs.reshape(dist.probs.shape[0], -1, self.num_embeddings) * prior_logits

        KL = posterior_entropy * entropy_temp - prior_entropy.reshape_as(posterior_entropy)
        KL = KL.mean(dim=1).sum()
        return KL

    def _calculate_ar_prior_logits(self, samples=None, input_shape=None):
        if self.use_autoregressive_prior:
            assert samples is not None # N * flat_dim * num_embeddings
            assert input_shape is not None # 
            batch_size = input_shape[0]
            spatial_shape = input_shape[2:]
            flat_dim = batch_size * np.prod(spatial_shape)
            prior_logits = self.embedding_logprob.unsqueeze(0)
            samples = samples.transpose(0, 1)
            if self.ar_method == "finitestate":
                autoregressive_samples = []
                if self.ar_input_quantized:
                    ar_samples_reshape = samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.embedding_dim).movedim(-2, 1)
                    for ar_offset in self.ar_offsets:
                        default_samples = torch.zeros_like(ar_samples_reshape)# [..., :1]
                        ar_samples = ar_samples_reshape
                        # take ar samples
                        # ar_samples = torch.cat(
                        #     [
                        #         default_sample,
                        #         ar_samples_reshape,
                        #     ], dim=-1
                        # )
                        # leave 0 as unknown sample, let total categories categorical_dim+1
                        # default_samples = torch.cat(
                        #     [
                        #         default_sample + 1,
                        #         torch.zeros_like(ar_samples_reshape),
                        #     ], dim=-1
                        # )
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, ar_samples.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        autoregressive_samples.append(ar_samples)
                else:
                    ar_samples_reshape = samples.reshape(batch_size, *spatial_shape, self.latent_dim, self.categorical_dim).movedim(-2, 1)
                    for ar_offset in self.ar_offsets:
                        default_sample = torch.zeros_like(ar_samples_reshape)[..., :1]
                        ar_samples = ar_samples_reshape
                        # take ar samples
                        ar_samples = torch.cat(
                            [
                                default_sample,
                                ar_samples_reshape,
                            ], dim=-1
                        )
                        # leave 0 as unknown sample, let total categories categorical_dim+1
                        default_samples = torch.cat(
                            [
                                default_sample + 1,
                                torch.zeros_like(ar_samples_reshape),
                            ], dim=-1
                        )
                        for data_dim, data_offset in enumerate(ar_offset):
                            if data_offset >= 0: continue
                            batched_data_dim = data_dim + 1
                            assert batched_data_dim != ar_samples.ndim - 1 # ar could not include categorical_dim
                            ar_samples = torch.cat((
                                default_samples.narrow(batched_data_dim, 0, -data_offset),
                                ar_samples.narrow(batched_data_dim, 0, ar_samples.shape[batched_data_dim]+data_offset)
                            ), dim=batched_data_dim)
                        autoregressive_samples.append(ar_samples)
                # [batch_size, self.latent_dim, *spatial_shape, self.ar_window_size*(self.categorical_dim+1)]
                autoregressive_samples = torch.cat(autoregressive_samples, dim=-1)
                if self.ar_mlp_per_channel:
                    autoregressive_samples_per_channel = autoregressive_samples.movedim(1, -2)\
                        .reshape(flat_dim, self.latent_dim, -1)
                    ar_logits_reshape = torch.stack([mlp(sample_channel.squeeze(1)) for mlp, sample_channel in zip(self.fsar_mlps_per_channel, autoregressive_samples_per_channel.split(1, dim=1))], dim=1)
                    prior_logits = ar_logits_reshape + prior_logits
                else:
                    autoregressive_samples_flat = autoregressive_samples.movedim(1, -2).reshape(flat_dim * self.latent_dim, -1)
                    ar_logits_reshape = self.fsar_mlp(autoregressive_samples_flat)
                    # merge ar logits and prior logits
                    prior_logits = ar_logits_reshape.reshape_as(samples) + prior_logits
            # TODO: ar models for vqvae
            else:
                assert len(spatial_shape) == 2
                ar_samples_reshape = samples.reshape(batch_size, *spatial_shape, -1).movedim(-1, 1)
                if self.ar_method.startswith("maskconv"):
                    if self.ar_method.startswith("maskconv3d"):
                        ar_samples_reshape = ar_samples_reshape.reshape(batch_size, self.latent_dim, -1, *spatial_shape)\
                            .permute(0, 2, 1, 3, 4)
                    prior_logits_reshape = self.ar_model(ar_samples_reshape)
                    if self.ar_method.startswith("maskconv3d"):
                        prior_logits_reshape = prior_logits_reshape.permute(0, 2, 1, 3, 4)\
                            .reshape(batch_size, self.latent_dim*self.categorical_dim, *spatial_shape)
                elif self.ar_method.startswith("checkerboard"):
                    prior_logits_reshape = self.ar_model(ar_samples_reshape)
                    checkerboard_mask_h_0 = torch.arange(0, spatial_shape[-2], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_mask_h_1 = torch.arange(1, spatial_shape[-2], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_mask_w_0 = torch.arange(0, spatial_shape[-1], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_mask_w_1 = torch.arange(1, spatial_shape[-1], 2, dtype=torch.long, device=ar_samples_reshape.device)
                    checkerboard_index_h_01, checkerboard_index_w_01 = torch.meshgrid(checkerboard_mask_h_0, checkerboard_mask_w_1)
                    checkerboard_index_h_10, checkerboard_index_w_10 = torch.meshgrid(checkerboard_mask_h_1, checkerboard_mask_w_0)
                    # multi-indexed tensor cannot be used as mutable left value
                    # prior_logits_reshape[..., checkerboard_mask_h_0, :][..., checkerboard_mask_w_1] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    # prior_logits_reshape[..., checkerboard_mask_h_1, :][..., checkerboard_mask_w_0] = prior_dist.logits.reshape(1, self.latent_channels, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_01, checkerboard_index_w_01] = prior_logits.reshape(1, self.latent_dim*self.categorical_dim, 1, 1)
                    prior_logits_reshape[..., checkerboard_index_h_10, checkerboard_index_w_10] = prior_logits.reshape(1, self.latent_dim*self.categorical_dim, 1, 1)
                else:
                    raise NotImplementedError(f"Unknown self.ar_method {self.ar_method}")
                prior_logits = prior_logits_reshape.movedim(1, -1).reshape(samples.shape[0], self.latent_dim, self.categorical_dim)
            # normalize logits
            prior_logits = torch.log_softmax(prior_logits, dim=-1).transpose(0, 1) ## N*flat_dim*M
        else:
            # prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1).unsqueeze(1) # N*1*M
            if self.use_code_freq:
                # NOTE: maybe it's better to use log freq as parameter?
                # prior_logits = torch.log(self.embedding_freq / self.embedding_freq.sum(-1, keepdim=True))
                prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1).unsqueeze(1) # N*1*M
            else:
                prior_logits = -math.log(self.num_embeddings)

        return prior_logits

    def _manual_update_code_freq(self, samples : torch.Tensor) -> None:
        with torch.no_grad():
            # NOTE: should soft samples be allowed here?
            total_count = samples.sum(dim=1)
            # sum over all gpus
            if self.ema_reduce_ddp and distributed.is_initialized():
                distributed.all_reduce(total_count)

            # normalize to probability.
            normalized_freq = total_count / total_count.sum(-1, keepdim=True)

            # ema update
            # ema = (1 - self.update_code_freq_ema_decay) * normalized_freq + self.update_code_freq_ema_decay * self.embedding_freq
            # self.embedding_freq.copy_(ema)

            ema = (1 - self.update_code_freq_ema_decay) * normalized_freq + \
                self.update_code_freq_ema_decay * torch.softmax(self.embedding_logprob, dim=-1)
            self.embedding_logprob.copy_(torch.log(ema))

    def set_custom_state(self, state: str = None):
        if state == "perturbed":
            self.state_gs_perturb = True
        else:
            self.state_gs_perturb = False
        return super().set_custom_state(state)

    def forward(self, x : torch.Tensor, calculate_sample_kl=False, **kwargs):
        x_shape = x.shape
        spatial_shape = x.shape[2:]
        # B, C, H, W = x.size()
        B, C = x.shape[:2]
        spatial_dim = np.prod(spatial_shape)
        N, M, D = self.latent_dim, self.num_embeddings, self.embedding_dim # self.embedding.size()
        x_embedding_dim = C // N
        assert C == N * x_embedding_dim

        if self.training and self.input_variance > 0:
            x = x + torch.normal(torch.zeros_like(x)) * self.input_variance.exp()
            self.update_cache("moniter_dict", 
                input_variance_mean=torch.mean(self.input_variance.exp()),
            )

        if self.use_autoregressive_posterior:
            x = self.input_ar_model(x)

        # x = x.view(B, N, x_embedding_dim H, W).permute(1, 0, 3, 4, 2)
        x = x.view(B, N, x_embedding_dim, spatial_dim).permute(1, 0, 3, 2) # N*B*spatial_dim*x_embedding_dim
        x_sample = self._sample_from_param(x)
        flat_dim = B * spatial_dim
        x_flat = x.reshape(N, flat_dim, x_embedding_dim)

        embedding = self.embedding
        if self.training and self.use_embedding_variance:
            embedding = embedding + torch.normal(torch.zeros_like(embedding)) * self.embedding_variance.exp()
            self.update_cache("moniter_dict", 
                embedding_variance_mean=torch.mean(self.embedding_variance.exp()),
            )

        # detach x_flat for straight through optimization
        if self.dist_type is None:
            x_flat = x_flat.detach()

        distances = self._pairwise_distance(x_flat, embedding.detach() if self.distance_detach_codebook else embedding)
        if self.use_code_variance:
            distances = distances / self.embedding_logvar.exp().unsqueeze(1)
        # distances = distances.view(N, B, H, W, M)
        distances = distances.view(N, B, spatial_dim, M)

        dist : distributions.Distribution = None # for lint
        logits = self._logits_from_distances(distances)
        eps = 1e-6
        if self.dist_type is None:
            dist = None
        elif self.dist_type == "CategoricalRSample":
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
            dist = CategoricalRSample(logits=logits)
        elif self.dist_type == "RelaxedOneHotCategorical":
            logits_norm = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
            logits = logits_norm / self.relax_temp
            dist = RelaxedOneHotCategorical(self.gs_temp if self.state_gs_perturb else eps,
                logits=logits,
            )
        elif self.dist_type == "AsymptoticRelaxedOneHotCategorical":
            dist = AsymptoticRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
                logits=logits,
            )
        elif self.dist_type == "DoubleRelaxedOneHotCategorical":
            dist = DoubleRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
                logits=logits,
            )
        else:
            raise ValueError(f"Unknown dist_type {self.dist_type} !")
            
        
        # do sampling from dist
        if dist is not None and (self.training or self.test_sampling):
            # if not dist.has_rsample:
            #     if self.training:
            #         raise ValueError(f"distribution {self.dist_type} cannot be used for training!")
            #     samples = dist.sample().view(N, -1, M)
            # else:
            if self.train_mc_sampling:
                samples = dist.sample_n(self.mc_sampling_size)
                quantized = self._sample_from_embedding(samples, embedding)
                loss_mc = self.mc_loss_func(quantized) / self.mc_sampling_size
                self.update_cache("loss_dict", loss_mc=loss_mc * self.mc_cost)
            else:
                samples = dist.rsample().view(N, flat_dim, M)
                if self.use_st_gumbel:
                    _, ind = samples.max(dim=-1)
                    samples_hard = torch.zeros_like(samples).view(N, flat_dim, M)
                    samples_hard.scatter_(-1, ind.view(N, flat_dim, 1), 1)
                    samples_hard = samples_hard.view(N, flat_dim, M)
                    samples = samples_hard - samples.detach() + samples
                if not self.training:
                    _, ind = samples.max(dim=-1)
                    self.update_cache("hist_dict",
                        code_hist=ind.view(N, -1).float().cpu().detach_()
                    )
                    samples = torch.zeros_like(samples).view(N, flat_dim, M)
                    samples.scatter_(-1, ind.view(N, flat_dim, 1), 1)
        else:
            samples = torch.argmin(distances, dim=-1)
            if not self.training:
                self.update_cache("hist_dict",
                    code_hist=samples.view(N, -1).float().cpu().detach_()
                )
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, flat_dim, M)

        quantized = self._sample_from_embedding(samples, embedding)
        quantized = quantized.view_as(x_sample)
        
        if self.ar_input_quantized:
            samples_quantized = quantized
            if self.use_straight_through:
                samples_quantized = x_sample + (samples_quantized - x_sample).detach() # straight through
            samples_quantized = samples_quantized.reshape(N, flat_dim, D)
            prior_logits = self._calculate_ar_prior_logits(samples_quantized, x_shape)
        else:
            if self.ar_input_st_logits and self.use_straight_through:
                ar_st_logits = torch.softmax(logits.view(N, flat_dim, M), dim=-1)
                samples = ar_st_logits + (samples - ar_st_logits).detach()
            prior_logits = self._calculate_ar_prior_logits(samples, x_shape)

        if calculate_sample_kl:
            if self.use_code_freq or self.use_autoregressive_prior:
                sample_kl = (samples * prior_logits).sum(-1)
            else:
                sample_kl = torch.ones(N, flat_dim).type_as(samples) * -math.log(self.num_embeddings)
            sample_kl = sample_kl.view(N, B, spatial_dim).sum((0, 2))
            self.update_cache("common", sample_kl=sample_kl)

        if self.training:
            # manual code freq update
            if self.use_code_freq and self.code_freq_manual_update:
                self._manual_update_code_freq(samples)

            # vq loss / kl loss
            if self.use_ema_update:
                    # TODO: fix ema for shared codebook?
                    with torch.no_grad():
                        total_count = samples.sum(dim=1)
                        dw = torch.bmm(samples.transpose(1, 2), x_flat)
                        if self.ema_reduce_ddp and distributed.is_initialized():
                            distributed.all_reduce(total_count)
                            distributed.all_reduce(dw)
                        self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * total_count
                        n = torch.sum(self.ema_count, dim=-1, keepdim=True)
                        self.ema_count = (self.ema_count + self.ema_epsilon) / (n + M * self.ema_epsilon) * n
                        self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * dw
                        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
            else:
                if dist is None \
                    or self.use_vq_loss_below_entropy_threshold and self.entropy_temp < self.entropy_temp_threshold\
                    or self.use_vq_loss_with_dist:
                    # loss_vq = F.mse_loss(x.detach(), quantized)
                    # distances_x_detached = self._pairwise_distance(embedding, x_flat.detach())
                    # loss_vq = torch.bmm(samples, distances_x_detached).mean() * self.embedding_dim
                    # loss_vq = self._pairwise_distance(quantized.reshape(N*flat_dim, 1, x_embedding_dim), x.detach().reshape(N*flat_dim, 1, x_embedding_dim)).mean()
                    # loss_vq = self._pairwise_distance(x.detach().reshape(N*flat_dim, 1, x_embedding_dim), quantized.reshape(N*flat_dim, 1, x_embedding_dim)).mean()
                    loss_vq = self._distance_loss(x_sample.reshape(N*flat_dim, 1, x_sample.shape[-1]).detach(), 
                        quantized.reshape(N*flat_dim, 1, x_sample.shape[-1]))
                    self.update_cache("loss_dict", loss_vq=loss_vq * self.vq_cost)
                
            # update prior logits
            if dist is None:
                # code freq should be updated with loss_rate
                if self.use_code_freq or self.use_autoregressive_prior:
                    loss_rate = -(samples * prior_logits).sum() / B
                    self.update_cache("loss_dict", loss_rate=loss_rate)
            else:
                KL = self._calculate_kl_from_dist(dist, prior_logits=prior_logits)
                self.update_cache("loss_dict", loss_rate=KL * self.kl_cost)

            # commitment loss
            commitment_cost = self.commitment_cost
            if self.use_commit_loss_below_entropy_threshold:
                if self.entropy_temp < self.entropy_temp_threshold:
                    commitment_cost = 0.25

            if commitment_cost != 0:
                if self.commitment_over_exp and dist is not None:
                    loss_commitment = (dist.probs * distances).mean()
                else:
                    # loss_commitment = F.mse_loss(x, quantized.detach())
                    # distances_embedding_detached = self._pairwise_distance(x_flat, embedding.detach())
                    # loss_commitment = self._pairwise_distance(x.reshape(N*flat_dim, 1, D), quantized.detach().reshape(N*flat_dim, 1, D)).mean()
                    loss_commitment = self._distance_loss(x_sample.reshape(N*flat_dim, 1, x_sample.shape[-1]), 
                        quantized.reshape(N*flat_dim, 1, x_sample.shape[-1]).detach())

                # loss = self.commitment_cost * e_latent_loss
                self.update_cache("loss_dict", loss_commitment = commitment_cost * loss_commitment)

        # TODO: add kl entropy metric?
        if self.use_code_freq or self.use_autoregressive_prior:
            # normalized_freq = self.embedding_freq / self.embedding_freq.sum(-1, keepdim=True)
            # prior_entropy = torch.bmm(samples, -torch.log(normalized_freq).unsqueeze(-1)).sum() / B
            prior_entropy = -(samples * prior_logits).sum() / B
        else:
            prior_entropy = math.log(self.num_embeddings) * (samples.numel() / M) / B
        self.update_cache("metric_dict", 
            prior_entropy=prior_entropy,
        )

        # kl
        # if dist is not None:
        #     KL = dist.probs * (dist.logits + math.log(M))
        #     KL[(dist.probs == 0).expand_as(KL)] = 0
        #     KL = KL.mean(dim=1).sum() # mean on batch dim

        # perplexity
        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )
        
        # centriod distance variance
        embedding_centriod = self.embedding.mean(-2, keepdim=True)
        embedding_centriod_distances = self._pairwise_distance(embedding_centriod, self.embedding)
        self.update_cache("metric_dict", 
            centriod_distance_variance=embedding_centriod_distances.var(dim=-1).sum()
        )

        # annealing
        if self.gs_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    gs_temp=self.gs_temp
                )
        if self.relax_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    relax_temp=self.relax_temp
                )
        if self.entropy_temp_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    entropy_temp=self.entropy_temp
                )

        # ema adjust sample
        if self.use_ema_update and self.ema_adjust_sample:
            with torch.no_grad():
                total_count = samples.sum(dim=1)
                dw = torch.bmm(samples.transpose(1, 2), torch.bmm(samples, embedding))
                if self.ema_reduce_ddp and distributed.is_initialized():
                    distributed.all_reduce(total_count)
                    distributed.all_reduce(dw)
                ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * total_count
                n = torch.sum(ema_count, dim=-1, keepdim=True)
                ema_count = (ema_count + self.ema_epsilon) / (n + M * self.ema_epsilon) * n
                ema_weight = self.ema_decay * embedding + (1 - self.ema_decay) * dw
                ema_weight = ema_weight / ema_count.unsqueeze(-1)
                quantized = torch.bmm(samples, ema_weight).view_as(x_sample)
        # output
        # quantized = quantized.permute(1, 0, 4, 2, 3).reshape(*x_shape)
        if self.use_straight_through or \
            self.use_st_below_entropy_threshold and self.entropy_temp < self.entropy_temp_threshold:
            quantized = quantized * (1 - self.st_weight) + x_sample * self.st_weight + (quantized * self.st_weight - x_sample * self.st_weight).detach()
        quantized = quantized.permute(1, 0, 3, 2).reshape(B, -1, *spatial_shape) #.reshape(*x_shape)
        
        return quantized

    def set_vamp_posterior(self, posterior):
        if not self.use_vamp_prior:
            raise RuntimeError("Should specify use_vamp_prior=True!")

        # check shape
        spatial_shape = posterior.shape[2:]
        # B, C, H, W = x.size()
        B, C = posterior.shape[:2]
        spatial_dim = np.prod(spatial_shape)
        N, M, D = self.embedding.size()
        assert C == N * D
        assert M == B * spatial_dim

        posterior = posterior.view(B, N, D, spatial_dim).permute(1, 0, 3, 2) # N*B*spatial_dim*D
        posterior = posterior.reshape(N, M, D).contiguous()
        self.embedding = posterior

    def encode(self, input, *args, **kwargs) -> bytes:
        spatial_shape = input.shape[2:]
        # B, C, H, W = x.size()
        B, C = input.shape[:2]
        spatial_dim = np.prod(spatial_shape)
        N, M, D = self.embedding.size()
        x_embedding_dim = C // N
        assert C == N * x_embedding_dim

        # x = x.view(B, N, x_embedding_dim H, W).permute(1, 0, 3, 4, 2)
        x = input.view(B, N, x_embedding_dim, spatial_dim).permute(1, 0, 3, 2) # N*B*spatial_dim*x_embedding_dim
        x_sample = self._sample_from_param(x)
        flat_dim = B * spatial_dim
        x_flat = x.reshape(N, flat_dim, x_embedding_dim)

        # detach x_flat for straight through optimization
        if self.dist_type is None:
            x_flat = x_flat.detach()

        distances = self._pairwise_distance(x_flat, self.embedding)
        # distances = distances.view(N, B, H, W, M)
        distances = distances.view(N, B, spatial_dim, M)

        # dist : distributions.Distribution = None # for lint
        # logits = self._logits_from_distances(distances)
        # eps = 1e-6
        # if self.dist_type is None:
        #     dist = None
        # elif self.dist_type == "CategoricalRSample":
        #     logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
        #     dist = CategoricalRSample(logits=logits)
        # elif self.dist_type == "RelaxedOneHotCategorical":
        #     logits_norm = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 
        #     logits = logits_norm / self.relax_temp
        #     dist = RelaxedOneHotCategorical(self.gs_temp if self.state_gs_perturb else eps,
        #         logits=logits,
        #     )
        # elif self.dist_type == "AsymptoticRelaxedOneHotCategorical":
        #     dist = AsymptoticRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
        #         logits=logits,
        #     )
        # elif self.dist_type == "DoubleRelaxedOneHotCategorical":
        #     dist = DoubleRelaxedOneHotCategorical(self.gs_temp, self.relax_temp if self.state_gs_perturb else eps,
        #         logits=logits,
        #     )
        # else:
        #     raise ValueError(f"Unknown dist_type {self.dist_type} !")

        # do sampling from dist
        # if dist is not None and self.test_sampling:
        #     samples = dist.rsample().view(N, flat_dim, M)
        #     _, samples = samples.max(dim=-1)
        # else:
        
        samples = torch.argmin(distances, dim=-1)
        samples = samples.movedim(1, 0).reshape(B, N, *spatial_shape)
        data = samples.detach().cpu().contiguous().numpy()
        indexes = torch.arange(N).unsqueeze(0).unsqueeze(-1)\
            .repeat(B, 1, spatial_dim).reshape(B, N, *spatial_shape)\
            .contiguous().numpy().astype(np.int32)

        # if self.coder_type == "rans":
        #     rans_encoder = RansEncoder()   
        #     indexes = torch.arange(self.latent_dim).unsqueeze(-1).unsqueeze(-1).expand_as(samples).numpy()
            
        #     # prepare for coding
        #     data = data.astype(np.int32).reshape(-1)
        #     indexes = indexes.astype(np.int32).reshape(-1)
        #     cdfs = np.array(self._prior_cdfs)
        #     cdf_sizes = np.array([len(cdf) for cdf in self._prior_cdfs])
        #     offsets = np.zeros(len(self._prior_cdfs))
        #     with self.profiler.start_time_profile("time_rans_encoder"):
        #         # rans_encoder.encode_with_indexes(
        #         #     data.tolist(), indexes.tolist(),
        #         #     self._prior_cdfs,
        #         #     [len(cdf) for cdf in self._prior_cdfs],
        #         #     [0]*len(self._prior_cdfs),
        #         # )
        #         data_bytes = rans_encoder.encode_with_indexes_np(
        #             data, indexes, cdfs, cdf_sizes, offsets
        #         )
        # elif self.coder_type == "tans":
        #     assert TANS_AVAILABLE
        #     assert self.num_embeddings <= 256
        #     # data_bytes_allchannel = []
        #     # data_channel_strings = [c.astype(np.uint8).tobytes() for c in data]
        #     # with self.profiler.start_time_profile("time_tans_encoder"):
        #     #     for c_idx in range(self.latent_dim):
        #     #         channel_bytes = fse_compress_using_ctable(data_channel_strings[c_idx], self._prior_ctables[c_idx])
        #     #         data_bytes_allchannel.append(struct.pack("<L", len(channel_bytes)))
        #     #         data_bytes_allchannel.append(channel_bytes)
        #     # data_bytes = b''.join(data_bytes_allchannel)
        #     tans_encoder = TansEncoder()   
        #     indexes = torch.arange(self.latent_dim).unsqueeze(-1).unsqueeze(-1).expand_as(samples).numpy()
            
        #     # prepare for coding
        #     data = data.astype(np.int32)#.reshape(-1)
        #     indexes = indexes.astype(np.int32)#.reshape(-1)
        #     ctables = np.array(self._prior_ctables)
        #     offsets = np.zeros(len(self._prior_ctables))
        #     with self.profiler.start_time_profile("time_tans_encoder"):
        #         # data_bytes = tans_encoder.encode_with_indexes(
        #         #     data.astype(np.int32).reshape(-1).tolist(), indexes.astype(np.int32).reshape(-1).tolist(),
        #         #     self._prior_ctables,
        #         #     [0]*len(self._prior_ctables),
        #         # )
        #         data_bytes = tans_encoder.encode_with_indexes_np(
        #             data, indexes, ctables, offsets
        #         )

        data_bytes = self._encoder.encode_with_indexes(data, indexes)

        # store sample shape in header
        byte_head = [struct.pack("B", len(spatial_shape)+1)]
        byte_head.append(struct.pack("<H", B))
        for dim in spatial_shape:
            byte_head.append(struct.pack("<H", dim))
        byte_head.append(data_bytes)
        return b''.join(byte_head)

    def decode(self, byte_string, *args, **kwargs) -> Any:
        # decode shape from header
        num_shape_dims = struct.unpack("B", byte_string[:1])[0]
        flat_shape = []
        byte_ptr = 1
        for _ in range(num_shape_dims):
            flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
            byte_ptr += 2
        flat_dim = np.prod(flat_shape)
        batch_dim = flat_shape[0]
        spatial_shape = flat_shape[1:]
        spatial_dim = np.prod(spatial_shape)

        indexes = torch.arange(self.latent_dim).unsqueeze(0).unsqueeze(-1)\
            .repeat(batch_dim, 1, spatial_dim).reshape(batch_dim, self.latent_dim, *spatial_shape)\
            .contiguous().numpy().astype(np.int32)

        # if self.coder_type == "rans":
        #     rans_decoder = RansDecoder()
        #     indexes = torch.arange(self.latent_dim).unsqueeze(-1).repeat(1, flat_dim).numpy()

        #     # prepare for coding
        #     indexes = indexes.astype(np.int32).reshape(-1)
        #     cdfs = np.array(self._prior_cdfs)
        #     cdf_sizes = np.array([len(cdf) for cdf in self._prior_cdfs])
        #     offsets = np.zeros(len(self._prior_cdfs))
        #     with self.profiler.start_time_profile("time_rans_decoder"):
        #         # samples = rans_decoder.decode_with_indexes(
        #         #     byte_string[byte_ptr:], indexes.tolist(),
        #         #     self._prior_cdfs,
        #         #     [len(cdf) for cdf in self._prior_cdfs],
        #         #     [0]*len(self._prior_cdfs),
        #         # )
        #         samples = rans_decoder.decode_with_indexes_np(
        #             byte_string[byte_ptr:], indexes, cdfs, cdf_sizes, offsets
        #         )
        # elif self.coder_type == "tans":
        #     assert TANS_AVAILABLE
        #     assert self.num_embeddings <= 256
        #     # samples_allchannel = []
        #     # with self.profiler.start_time_profile("time_tans_decoder"):
        #     #     for c_idx in range(self.latent_dim):
        #     #         channel_string_length = struct.unpack("<L", byte_string[byte_ptr:(byte_ptr+4)])[0]
        #     #         byte_ptr += 4
        #     #         channel_string = fse_decompress_using_dtable(byte_string[byte_ptr:(byte_ptr+channel_string_length)], self._prior_dtables[c_idx], flat_dim)
        #     #         byte_ptr += channel_string_length
        #     #         samples_allchannel.append(np.frombuffer(channel_string, dtype=np.uint8))
        #     # samples = np.stack(samples_allchannel)
        #     tans_decoder = TansDecoder()
        #     indexes = torch.arange(self.latent_dim).unsqueeze(-1).repeat(1, flat_dim).numpy()

        #     # prepare for coding
        #     encoded = byte_string[byte_ptr:]
        #     indexes = indexes.astype(np.int32) # .reshape(-1)
        #     dtables = np.array(self._prior_dtables)
        #     offsets = np.zeros(len(self._prior_dtables))
        #     with self.profiler.start_time_profile("time_tans_decoder"):
        #         # samples = tans_decoder.decode_with_indexes(
        #         #     byte_string[byte_ptr:], indexes.astype(np.int32).reshape(-1).tolist(),
        #         #     self._prior_dtables,
        #         #     [0]*len(self._prior_dtables),
        #         # )
        #         samples = tans_decoder.decode_with_indexes_np(
        #             encoded, indexes, dtables, offsets
        #         )

        samples = self._decoder.decode_with_indexes(byte_string[byte_ptr:], indexes)

        samples = torch.as_tensor(samples).to(dtype=torch.long, device=self.device)
        samples = F.one_hot(samples, self.num_embeddings).float()
        samples = samples.movedim(1, 0).reshape(self.latent_dim, flat_dim, self.num_embeddings)

        quantized = self._sample_from_embedding(samples)
        quantized = quantized.view(self.latent_dim, batch_dim, spatial_dim, self.embedding_dim)
        quantized = quantized.permute(1, 0, 3, 2).reshape(batch_dim, self.latent_dim * self.embedding_dim, *spatial_shape)
        return quantized

    def update_state(self, *args, **kwargs) -> None:
        with torch.no_grad():
            if self.use_code_freq:
                prior_pmfs = torch.softmax(self.embedding_logprob, dim=-1)#.unsqueeze(-1)
            else:
                prior_pmfs = torch.ones(self.latent_dim, self.num_embeddings) / self.num_embeddings
            # if self.coder_type == "rans":
            #     self._prior_cdfs = [pmf_to_quantized_cdf(pmf.tolist()) for pmf in prior_pmfs.reshape(-1, self.num_embeddings)]
            # elif self.coder_type == "tans":
            #     assert TANS_AVAILABLE
            #     assert self.num_embeddings <= 256
            #     prior_cnt = (prior_pmfs * (1<<10)).clamp_min(1).reshape(-1, self.num_embeddings).detach().cpu().numpy().astype(np.int32)
            #     self._prior_ctables = [create_ctable_using_cnt(cnt, maxSymbolValue=(self.num_embeddings-1)).tolist() for cnt in prior_cnt]
            #     self._prior_dtables = [create_dtable_using_cnt(cnt, maxSymbolValue=(self.num_embeddings-1)).tolist() for cnt in prior_cnt]
            # else:
            #     raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")
            
            # TODO: autoregressive vq coding
            # categorical_dim = self.categorical_dim
            # if self.use_autoregressive_prior and self.ar_method == "finitestate":
            #     # TODO: this is a hard limit! may could be improved!
            #     if len(self.ar_offsets) > 2:
            #         pass
            #     else:
            #         lookup_table_shape = [self.latent_channels] + [categorical_dim+1] * len(self.ar_offsets) + [categorical_dim]
            #         ar_idx_all = list(itertools.product(range(self.categorical_dim+1), repeat=self.ar_window_size))
            #         ar_idx_all = torch.tensor(ar_idx_all, device=self.device).reshape(-1, 1).repeat(1, self.latent_channels)
            #         ar_input_all = self._finite_state_to_samples(ar_idx_all, add_default_samples=True).type_as(prior_logits)\
            #             .reshape(-1, self.ar_window_size, self.latent_channels, self.num_sample_params).movedim(1, -2)\
            #             .reshape(-1, self.latent_channels, self.ar_window_size*self.num_sample_params).movedim(1, 0)
            #         if self.ar_mlp_per_channel:
            #             ar_logits_reshape = torch.stack([mlp(ar_input) for (mlp, ar_input) in zip(self.fsar_mlps_per_channel, ar_input_all)], dim=0)
            #         else:
            #             ar_logits_reshape = self.fsar_mlp(ar_input_all)
            #         prior_logits = prior_logits.unsqueeze(-2) + ar_logits_reshape
            #         prior_logits = self._normalize_prior_logits(prior_logits)
            #         prior_logits = prior_logits.reshape(*lookup_table_shape)

            # prior_pmfs = prior_logits.exp()

            # TODO: customize freq precision
            if self.coder_type == "rans" or self.coder_type == "rans64":
                self._encoder = Rans64Encoder(freq_precision=self.coder_freq_precision)
                self._decoder = Rans64Decoder(freq_precision=self.coder_freq_precision)
            elif self.coder_type == "tans":
                self._encoder = TansEncoder(table_log=self.coder_freq_precision, max_symbol_value=self.categorical_dim-1)
                self._decoder = TansDecoder(table_log=self.coder_freq_precision, max_symbol_value=self.categorical_dim-1)
            else:
                raise NotImplementedError(f"Unknown coder_type {self.coder_type}!")

            prior_cnt = (prior_pmfs * (1<<self.coder_freq_precision)).clamp_min(1).reshape(-1, self.categorical_dim)
            prior_cnt = prior_cnt.detach().cpu().numpy().astype(np.int32)
            num_symbols = np.zeros(len(prior_cnt), dtype=np.int32) + self.categorical_dim
            offsets = np.zeros(len(prior_cnt), dtype=np.int32)

            self._encoder.init_params(prior_cnt, num_symbols, offsets)
            self._decoder.init_params(prior_cnt, num_symbols, offsets)

            # if self.use_autoregressive_prior and self.ar_method == "finitestate":
            #     ar_indexes = np.arange(len(prior_cnt), dtype=np.int32).reshape(1, *prior_pmfs.shape[:-1])

            #     self._encoder.init_ar_params(ar_indexes, [self.ar_offsets])
            #     self._decoder.init_ar_params(ar_indexes, [self.ar_offsets])




# class VQGaussianEmbeddingPriorCoder(MultiChannelVQPriorCoder):
#     def __init__(self, latent_dim=8, num_embeddings=128, embedding_dim=32, **kwargs):
#         super().__init__(latent_dim, num_embeddings, embedding_dim, **kwargs)
#         self.embedding_logvar = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
#         # nn.init.uniform_(self.embedding, -1, 1)
#         nn.init.constant_(self.embedding_logvar, 0)
#         self.embedding_logvar.lr_modifier = self.embedding_lr_modifier

#     def _sample_from_embedding(self, samples) -> torch.Tensor:
#         means = torch.bmm(samples, self.embedding)
#         logvars = torch.bmm(samples, self.embedding_logvar)
#         dist = distributions.Normal(means, torch.exp(0.5 * logvars))
#         return dist.rsample()


class GaussianVQPriorCoder(MultiChannelVQPriorCoder):
    def __init__(self, latent_dim=128, num_embeddings=128, embedding_dim=2, 
        in_channels=256, # latent_channels=None, 
        distance_method="l2", distance_loss_method=None, 
        use_pyramid_init=False, pyramid_init_invert_logprob=True, # freeze_logvar=False,
        gaussian_kl_cost=1.0, gaussian_kl_from_encoder=False,
        rsample_params=False, rsample_params_method="rsample",
        **kwargs,
        ):
        self.in_channels = in_channels
        self.latent_channels = latent_dim * embedding_dim // 2 # (in_channels // 2) if latent_channels is None else latent_channels
        self.distance_method = distance_method
        self.distance_loss_method = self.distance_method if distance_loss_method is None else distance_loss_method
        self.gaussian_kl_cost = gaussian_kl_cost
        self.gaussian_kl_from_encoder = gaussian_kl_from_encoder
        self.rsample_params = rsample_params
        self.rsample_params_method = rsample_params_method
        super().__init__(latent_dim, num_embeddings, embedding_dim, 
            **kwargs
        )
        # we need extra logvar params in the embedding
        # if self.rsample_params:
        #     nn.init.uniform_(self.embedding[:, :, (embedding_dim//2):], math.log(1/self.num_embeddings), math.log(1/self.num_embeddings * 6))
        #     embedding_logvar = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        #     embedding_new = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim*2))
        #     embedding_new.data[:, :, :embedding_dim] = self.embedding.data
        #     embedding_new.data[:, :, embedding_dim:] = embedding_logvar.data
        #     self.embedding = embedding_new
        #     self.embedding.lr_modifier = self.embedding_lr_modifier
            
        # init scale
        # self.embedding.data[:,:,(self.embedding_dim//2):] = torch.ones(1, self.num_embeddings, 1) * -10
        if use_pyramid_init:
            # kl_level_base = 2
            # num_kl_levels = math.floor(math.log(num_embeddings * (kl_level_base - 1) + 1) / math.log(kl_level_base))
            # cur_idx = 0
            # for i in range(num_kl_levels):
            #     cur_num_embeddings = kl_level_base ** i
            #     cur_kl = math.log(cur_num_embeddings)
            #     # kl = -0.5 * (1 + logvar - mean ** 2 - logvar.exp())
            #     with torch.no_grad():
            #         trial_step = 0.1
            #         max_trial = math.exp(1 + cur_kl * 2)
            #         trial_num_steps = math.floor(max_trial / trial_step)
            #         logvar_trials = torch.linspace(0, math.exp(1 + cur_kl * 2), steps=trial_num_steps)
            #         max_logvar_eq_trials = logvar_trials + 1 - logvar_trials.exp() + cur_kl * 2
            #         max_logvar_eq_trials[max_logvar_eq_trials < 0] = np.inf
            #         max_logvar = torch.abs(max_logvar_eq_trials).argmin() * trial_step
            #         min_logvar_eq_trials = -logvar_trials + 1 - (-logvar_trials).exp() + cur_kl * 2
            #         min_logvar_eq_trials[min_logvar_eq_trials < 0] = np.inf
            #         min_logvar = -torch.abs(min_logvar_eq_trials).argmin() * trial_step
            #         if cur_num_embeddings > 1:
            #             logvars = torch.linspace(min_logvar, max_logvar, cur_num_embeddings // 2)
            #             means = torch.sqrt(1 + logvars - logvars.exp() + 2 * cur_kl)
            #             means_negative = -means
            #             embedding_init_positive = torch.stack([means, logvars], dim=-1)
            #             embedding_init_negative = torch.stack([means_negative, logvars], dim=-1)
            #             embedding_init = torch.cat([embedding_init_positive, embedding_init_negative], dim=0).repeat_interleave(self.embedding_dim // 2, dim=-1)
            #         else:
            #             embedding_init = torch.zeros(cur_num_embeddings, self.embedding_dim)
            #         self.embedding.data[:, cur_idx:(cur_idx+cur_num_embeddings), :] = \
            #             embedding_init.unsqueeze(0).type_as(self.embedding.data)
            #         if self.use_code_freq:
            #             self.embedding_logprob.data[:, cur_idx:(cur_idx+cur_num_embeddings)] = cur_kl
            #     cur_idx += cur_num_embeddings
            embedding, embedding_logprob = gaussian_pyramid_init(num_embeddings, embedding_dim, invert_logprob=pyramid_init_invert_logprob)
            self.embedding.data[:] = embedding.unsqueeze(0)
            if self.use_code_freq:
                self.embedding_logprob.data[:] = embedding_logprob.unsqueeze(0)
            if self.use_ema_update:
                self.ema_weight = self.embedding.clone()            
        else:
            # init scale
            self.embedding.data[:,:,(self.embedding_dim//2):] = torch.ones(1, self.num_embeddings, 1) * -math.log(self.num_embeddings)

        self.input_layer = nn.Linear(self.in_channels, self.latent_channels * 2)
        self.output_layer = nn.Linear(self.latent_channels, self.in_channels)

    # TODO: using cross entropy is simpler!
    def _pairwise_distance(self, x1 : torch.Tensor, x2 : torch.Tensor, distance_method=None):
        eps = 1e-6
        x1_repeat = x1.unsqueeze(2) #.repeat(1, 1, x2.shape[1], 1)
        x2_repeat = x2.unsqueeze(1) #.repeat(1, x1.shape[1], 1, 1)
        x1_mean, x1_logvar = x1_repeat.chunk(2, dim=-1)
        x2_mean, x2_logvar = x2_repeat.chunk(2, dim=-1)
        x1_scale = torch.exp(0.5 * x1_logvar) + eps
        x2_scale = torch.exp(0.5 * x2_logvar) + eps

        if distance_method is None:
            distance_method = self.distance_method

        if distance_method == "l2":
            # return super()._pairwise_distance(x1, x2)
            return (x1_repeat - x2_repeat).pow(2).mean(dim=-1)
        elif distance_method == "mean_l2":
            return (x1_mean - x2_mean).pow(2).mean(dim=-1)
        elif distance_method == "kl":
            # kl-based distance
            # dist1 = distributions.Normal(x1_mean, x1_scale)
            # dist2 = distributions.Normal(x2_mean, x2_scale)
            # kl = distributions.kl_divergence(dist1, dist2)
            var_ratio_log = x1_logvar - x2_logvar
            t1 = (x1_mean - x2_mean).pow(2) / torch.exp(x2_logvar)
            kl = 0.5 * (torch.exp(var_ratio_log) + t1 - 1 - var_ratio_log)

            return kl.mean(dim=-1)
        elif distance_method == "rsample_kl":
            x1_dist = distributions.Normal(x1_mean, x1_scale)
            x2_dist = distributions.Normal(x2_mean, x2_scale)
            x1_samples = x1_dist.rsample()
            # x2_samples = x2_dist.rsample()
            # return ((x2_samples - x1_mean) / x1_scale).pow(2).mean(dim=-1)
            return (x1_dist.log_prob(x1_samples) - x2_dist.log_prob(x1_samples)).mean(dim=-1)
        elif distance_method == "rsample_logp":
            x1_dist = distributions.Normal(x1_mean, x1_scale)
            x2_dist = distributions.Normal(x2_mean, x2_scale)
            x1_samples = x1_dist.rsample()
            # x2_samples = x2_dist.rsample()
            # return ((x2_samples - x1_mean) / x1_scale).pow(2).mean(dim=-1)
            return -x2_dist.log_prob(x1_samples).mean(dim=-1)
            # return (x1_samples - x2_samples).pow(2).mean(dim=-1)
        elif distance_method == "mean_rsample_logp":
            x2_dist = distributions.Normal(x2_mean, x2_scale)
            return -x2_dist.log_prob(x1_mean).mean(dim=-1)
        elif distance_method == "mean_gmm_logp":
            if self.use_code_freq:
                # normalized_freq = self.embedding_freq / self.embedding_freq.sum(-1, keepdim=True)
                # prior_entropy = torch.bmm(samples, -torch.log(normalized_freq).unsqueeze(-1)).sum() / B
                # prior_probs = torch.softmax(self.embedding_logprob, dim=-1)
                prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1)
            else:
                # prior_probs = 1 / self.num_embeddings
                prior_logits = -math.log(self.num_embeddings) * torch.ones_like(self.embedding)[:,:,0]
            # x2_dist = distributions.Normal(x2_mean.permute(0, 1, 3, 2), x2_scale.permute(0, 1, 3, 2))
            # mix = distributions.Categorical(probs=prior_probs.unsqueeze(1).unsqueeze(1))
            # x2_gmm = distributions.MixtureSameFamily(mix, x2_dist)
            # return -x2_gmm.log_prob(x1_mean.permute(0, 1, 3, 2)[:,:,:,0]).mean(dim=-1)
            x2_dist = distributions.Normal(x2_mean, x2_scale)
            x1_log_prob = x2_dist.log_prob(x1_mean)
            prior_logits = prior_logits.unsqueeze(1).unsqueeze(-1)
            comp_log_prob = prior_logits + x1_log_prob
            gmm_log_prob = torch.logsumexp(prior_logits + x1_log_prob, dim=2, keepdim=True)
            return -(comp_log_prob - gmm_log_prob).mean(dim=-1)
        elif distance_method == "x1_dist_x2_mean_logp":
            x1_dist = distributions.Normal(x1_mean, x1_scale)
            return -x1_dist.log_prob(x2_mean).mean(dim=-1)
        elif distance_method == "x1_rsample_x2_mean_l2":
            x1_dist = distributions.Normal(x1_mean, x1_scale)
            x1_samples = x1_dist.rsample()
            return (x1_samples - x2_mean).pow(2).mean(dim=-1)
        elif distance_method == "Mahalanobis":
            return torch.sqrt(((x1_mean - x2_mean) ** 2) / (torch.exp(x1_logvar) + torch.exp(x2_logvar))).mean(dim=-1)
            # return (torch.log((x1_mean - x2_mean) ** 2) - torch.log(x1_scale ** 2 + x2_scale ** 2)).mean(dim=-1)
        # https://en.wikipedia.org/wiki/Bhattacharyya_distance
        elif distance_method == "Bhattacharyya":
            return (((x1_mean - x2_mean) ** 2) / (x1_scale ** 2 + x2_scale ** 2) / 4 + \
                torch.log((x1_scale ** 2 + x2_scale ** 2) / x1_scale / x2_scale / 2) / 2).mean(dim=-1)
        elif distance_method == "Wasserstein":
            return ((x1_mean - x2_mean).pow(2) + \
                torch.exp(0.5 * x1_logvar) + torch.exp(0.5 * x2_logvar) - \
                2 * torch.exp(0.25 * (x1_logvar + x2_logvar))).mean(dim=-1)
        else:
            raise NotImplementedError(f"Unknown distance_method {distance_method}")
    
    def _distance_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self._pairwise_distance(x1, x2, distance_method=self.distance_loss_method).mean()
    
    def _logits_from_distances(self, distances):
        if self.rsample_params:
            return -distances * self.embedding_dim / 2
        else:
            return super()._logits_from_distances(distances)

    def _sample_from_param(self, param) -> torch.Tensor:
        if self.rsample_params:
            if self.rsample_params_method == "rsample":
                means, logvars = param.chunk(2, dim=-1)
                dist = distributions.Normal(means, torch.exp(0.5 * logvars))
                samples = dist.rsample()
            elif self.rsample_params_method == "mean":
                means, logvars = param.chunk(2, dim=-1)
                samples = means
            return samples
        else:
            return super()._sample_from_param(param)

    def _sample_from_embedding(self, samples) -> torch.Tensor:
        if self.rsample_params:
            # different samples for every instance
            samples_batched = samples.reshape(-1, 1, self.num_embeddings)
            embedding_samples = self._sample_from_param(self.embedding.unsqueeze(1).repeat(1, samples.shape[1], 1, 1).reshape(-1, self.num_embeddings, self.embedding_dim))
            return torch.bmm(samples_batched, embedding_samples).reshape(samples.shape[0], samples.shape[1], embedding_samples.shape[2])
            # means, logvars = self.embedding.chunk(2, dim=-1)
            # means = torch.bmm(samples, means)
            # logvars = torch.bmm(samples, logvars)
            # dist = distributions.Normal(means, torch.exp(0.5 * logvars))
            # return dist.rsample()
        else:
            return super()._sample_from_embedding(samples)

    def _manual_update_code_freq(self, samples: torch.Tensor) -> None:
        # return super()._manual_update_code_freq(samples)
        with torch.no_grad():
            embedding_mean, embedding_logvar = self.embedding.chunk(2, dim=-1)
            self.embedding_logprob.data = embedding_logvar.mean(dim=-1)

    def forward(self, input : torch.Tensor, **kwargs):
        input_shape = input.shape
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        assert(channel_size == self.in_channels)

        input = input.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
        input = self.input_layer(input)
        input = input.reshape(batch_size, -1, self.latent_channels * 2)\
                    .permute(0, 2, 1).contiguous()\
                    # .reshape(batch_size, self.latent_channels * 2, *input_shape[-2:])
        
        if self.rsample_params:
            output = super().forward(input)
        else:
            quantized_input = super().forward(input)

            quantized_input = quantized_input.view(batch_size, self.latent_dim, self.embedding_dim, -1)
            quantized_mean, quantized_logvar = quantized_input.chunk(2, dim=2)
            quantized_std = torch.exp(0.5 * quantized_logvar)

            if self.training and not 'loss_rate' in self.get_raw_cache("loss_dict"):
                # directly train encoder by KLD
                if self.gaussian_kl_from_encoder:
                    input = input.view(batch_size, self.latent_dim, self.embedding_dim, -1)
                    mean, logvar = input.chunk(2, dim=2)
                    KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1))
                else:
                    KLD = torch.sum(-0.5 * torch.sum(1 + quantized_logvar - quantized_mean ** 2 - quantized_logvar.exp(), dim=1))
            
                # NOTE: kl divergence is not true entropy here
                # self.update_cache("metric_dict",
                #     prior_entropy = KLD / input.shape[0], # normalize by batch size
                # )
                self.update_cache("loss_dict",
                    loss_rate=KLD / input_shape[0] * self.gaussian_kl_cost, # normalize by batch size
                )

            dist = distributions.Normal(quantized_mean, quantized_std)
            output = dist.rsample()
        
        output = output.reshape(batch_size, self.latent_channels, -1)
        output = output.permute(0, 2, 1).reshape(-1, self.latent_channels).contiguous()

        output = self.output_layer(output)
        output = output.reshape(batch_size, -1, channel_size).permute(0, 2, 1).reshape(*input_shape).contiguous()

        return output


class VQGaussianEmbeddingPriorCoder(MultiChannelVQPriorCoder):
    def __init__(self, latent_dim=8, num_embeddings=128, embedding_dim=32,
        logvar_init=None, freeze_logvar=False, var_scale=1.0, var_scale_anneal=False,
        **kwargs):
        super().__init__(latent_dim, num_embeddings, embedding_dim, **kwargs)
        self.embedding_logvar = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        # nn.init.uniform_(self.embedding, -1, 1)
        if logvar_init is None: logvar_init = -math.log(self.num_embeddings)
        nn.init.constant_(self.embedding_logvar, logvar_init)
        self.embedding_logvar.lr_modifier = self.embedding_lr_modifier
        if freeze_logvar:
            self.embedding_logvar.requires_grad = False

        self.var_scale_anneal = var_scale_anneal
        if var_scale_anneal:
            self.var_scale = nn.Parameter(torch.tensor(var_scale), requires_grad=False)
        else:
            self.var_scale = var_scale

    def _sample_from_embedding(self, samples) -> torch.Tensor:
        # means = torch.bmm(samples, self.embedding)
        # logvars = torch.bmm(samples, self.embedding_logvar)
        # dist = distributions.Normal(means, torch.exp(0.5 * logvars))
        # return dist.rsample()
        samples_batched = samples.reshape(-1, 1, self.num_embeddings)
        embedding_dist = distributions.Normal(
            self.embedding.unsqueeze(1).repeat(1, samples.shape[1], 1, 1).reshape(-1, self.num_embeddings, self.embedding_dim),
            torch.exp(0.5 * self.embedding_logvar.unsqueeze(1).repeat(1, samples.shape[1], 1, 1).reshape(-1, self.num_embeddings, self.embedding_dim)) * self.var_scale,
        )
        if self.var_scale_anneal:
            if self.training:
                self.update_cache("metric_dict", 
                    var_scale=self.var_scale
                )

        embedding_samples = embedding_dist.rsample()
        return torch.bmm(samples_batched, embedding_samples).reshape(samples.shape[0], samples.shape[1], embedding_samples.shape[2])

    def _pairwise_distance(self, x1 : torch.Tensor, x2 : torch.Tensor, distance_method=None):
        x1_repeat = x1.unsqueeze(2) #.repeat(1, 1, x2.shape[1], 1)
        x2_repeat = x2.unsqueeze(1) #.repeat(1, x1.shape[1], 1, 1)
        x2_logvar = self.embedding_logvar.unsqueeze(1)

        if self.use_code_freq:
            prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1)
        else:
            prior_logits = -math.log(self.num_embeddings) * torch.ones_like(self.embedding)[:,:,0]
        # x2_dist = distributions.Normal(x2_mean.permute(0, 1, 3, 2), x2_scale.permute(0, 1, 3, 2))
        # mix = distributions.Categorical(probs=prior_probs.unsqueeze(1).unsqueeze(1))
        # x2_gmm = distributions.MixtureSameFamily(mix, x2_dist)
        # return -x2_gmm.log_prob(x1_mean.permute(0, 1, 3, 2)[:,:,:,0]).mean(dim=-1)
        x2_dist = distributions.Normal(x2_repeat, torch.exp(0.5 * x2_logvar) * self.var_scale)
        x1_log_prob = x2_dist.log_prob(x1_repeat)
        prior_logits = prior_logits.unsqueeze(1).unsqueeze(-1)
        comp_log_prob = prior_logits + x1_log_prob
        gmm_log_prob = torch.logsumexp(prior_logits + x1_log_prob, dim=2, keepdim=True)
        return -(comp_log_prob - gmm_log_prob).mean(dim=-1)

    def _calculate_kl_from_dist(self, dist: distributions.Distribution, prior_logits=None):
        # KL: N, B, spatial_dim, M
        entropy_temp = max(self.entropy_temp, self.entropy_temp_min)
        if entropy_temp != 1.0:
            embedding_logvar = self.embedding_logvar.mean(-1).unsqueeze(1).unsqueeze(1)
            KL = dist.probs * ( \
                (entropy_temp - 1) * (1 + math.log(2 * math.pi) + embedding_logvar) / 2 + \
                entropy_temp * dist.logits - prior_logits)
        else:
            KL = dist.probs * (dist.logits - prior_logits)
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.mean(dim=1).sum() # mean on batch dim

        return KL


class DistributionVQPriorCoder(MultiChannelVQPriorCoder):
    def __init__(self, latent_dim=8, num_embeddings=128, embedding_dim=32,
        in_channels=256,
        # init_method="random",
        **kwargs,
        ):
        self.in_channels = in_channels
        self.latent_channels = latent_dim * embedding_dim // self.dist_num_params
        super().__init__(latent_dim, num_embeddings, embedding_dim, 
            **kwargs
        )

        # TODO: init
        self.initialize_embedding()

        self.input_layer = nn.Linear(self.in_channels, self.latent_channels) # * self.dist_num_params)
        self.output_layer = nn.Linear(self.latent_channels, self.in_channels)

    @property
    def dist_num_params(self):
        return 2

    def param_to_dist(self, x) -> distributions.Distribution:
        raise NotImplementedError()

    def initialize_embedding(self):
        pass

    def _pairwise_distance(self, x1 : torch.Tensor, x2 : torch.Tensor, distance_method=None):
        x1_repeat = x1.unsqueeze(2) #.repeat(1, 1, x2.shape[1], 1)
        x2_repeat = x2.unsqueeze(1) #.repeat(1, x1.shape[1], 1, 1)

        if self.use_code_freq:
            prior_logits = torch.log_softmax(self.embedding_logprob, dim=-1)
        else:
            prior_logits = -math.log(self.num_embeddings) * torch.ones_like(self.embedding)[:,:,0]
        # x2_dist = distributions.Normal(x2_mean.permute(0, 1, 3, 2), x2_scale.permute(0, 1, 3, 2))
        # mix = distributions.Categorical(probs=prior_probs.unsqueeze(1).unsqueeze(1))
        # x2_gmm = distributions.MixtureSameFamily(mix, x2_dist)
        # return -x2_gmm.log_prob(x1_mean.permute(0, 1, 3, 2)[:,:,:,0]).mean(dim=-1)
        x2_dist = self.param_to_dist(x2_repeat)
        x1_log_prob = x2_dist.log_prob(x1_repeat)
        prior_logits = prior_logits.unsqueeze(1)#.unsqueeze(-1)
        comp_log_prob = prior_logits + x1_log_prob
        gmm_log_prob = torch.logsumexp(prior_logits + x1_log_prob, dim=2, keepdim=True)
        return -(comp_log_prob - gmm_log_prob)

    def _logits_from_distances(self, distances):
        return -distances * self.embedding_dim / self.dist_num_params

    # def _sample_from_param(self, param) -> torch.Tensor:
    #     dist = self.param_to_dist(param)
    #     return dist.rsample()

    def _sample_from_embedding(self, samples) -> torch.Tensor:
        samples_batched = samples.reshape(-1, 1, self.num_embeddings)
        embedding_dist = self.param_to_dist(self.embedding.unsqueeze(1).repeat(1, samples.shape[1], 1, 1).reshape(-1, self.num_embeddings, self.embedding_dim))
        embedding_samples = embedding_dist.rsample()
        return torch.bmm(samples_batched, embedding_samples).reshape(samples.shape[0], samples.shape[1], embedding_samples.shape[2])

    def forward(self, input : torch.Tensor, **kwargs):
        input_shape = input.shape
        batch_size = input.shape[0]
        channel_size = input.shape[1]
        assert(channel_size == self.in_channels)

        input = input.reshape(batch_size, channel_size, -1).permute(0, 2, 1).reshape(-1, channel_size).contiguous()
        input = self.input_layer(input)
        input = input.reshape(batch_size, -1, self.latent_channels)\
                    .permute(0, 2, 1).contiguous()\
                    # .reshape(batch_size, self.latent_channels, *input_shape[-2:])
        
        output = super().forward(input)
        
        output = output.reshape(batch_size, self.latent_channels, -1)
        output = output.permute(0, 2, 1).reshape(-1, self.latent_channels).contiguous()

        output = self.output_layer(output)
        output = output.reshape(batch_size, -1, channel_size).permute(0, 2, 1).reshape(*input_shape).contiguous()

        return output


class UnivarGaussianDistributionVQPriorCoder(DistributionVQPriorCoder):
    def __init__(self, use_pyramid_init=False, **kwargs):
        self.use_pyramid_init = use_pyramid_init
        super().__init__(**kwargs)

    @property
    def dist_num_params(self):
        return 2

    def param_to_dist(self, x) -> distributions.Distribution:
        means, logvars = x.chunk(2, dim=-1)
        scales = torch.exp(0.5 * logvars)
        # return distributions.Normal(means, torch.exp(0.5 * logvars))
        return distributions.LowRankMultivariateNormal(means, torch.zeros_like(means).unsqueeze(-1), scales)

    def initialize_embedding(self):
        if self.use_pyramid_init:
            embedding, embedding_logprob = gaussian_pyramid_init(self.num_embeddings, self.embedding_dim)
            self.embedding.data[:] = embedding.unsqueeze(0)
            if self.use_code_freq:
                self.embedding_logprob.data[:] = embedding_logprob.unsqueeze(0)
            if self.use_ema_update:
                self.ema_weight = self.embedding.clone()   
        else:
            # init scale
            self.embedding.data[:,:,(self.embedding_dim//self.dist_num_params):] = torch.ones(1, self.num_embeddings, 1) * -math.log(self.num_embeddings)
            # pass

class LRMultivarGaussianDistributionVQPriorCoder(DistributionVQPriorCoder):
    def __init__(self, embedding_dim=64, dist_rank=2, use_pyramid_init=False, **kwargs):
        self.dist_rank = dist_rank
        self.use_pyramid_init = use_pyramid_init
        super().__init__(embedding_dim=embedding_dim, **kwargs)

    @property
    def dist_num_params(self):
        return self.dist_rank + 2

    def param_to_dist(self, x) -> distributions.Distribution:
        dist_dim = self.embedding_dim // self.dist_num_params
        means, cov_diag, cov_factor = x.split([dist_dim, dist_dim, dist_dim*self.dist_rank], dim=-1)
        cov_factor_reshape = tuple(means.shape) + (self.dist_rank, )
        cov_factor = cov_factor.reshape(*cov_factor_reshape)
        return distributions.LowRankMultivariateNormal(means, cov_factor, torch.exp(0.5 * cov_diag))

    def initialize_embedding(self):
        if self.use_pyramid_init:
            embedding, embedding_logprob = gaussian_pyramid_init(self.num_embeddings, self.embedding_dim*2//(self.dist_rank + 2))
            self.embedding.data[:, :, :(self.embedding_dim*2//(self.dist_rank + 2))] = embedding.unsqueeze(0)
            if self.use_code_freq:
                self.embedding_logprob.data[:] = embedding_logprob.unsqueeze(0)
            if self.use_ema_update:
                self.ema_weight = self.embedding.clone()
        else:
            # init scale
            self.embedding.data[:,:,(self.embedding_dim//(self.dist_rank + 2)):(self.embedding_dim*2//(self.dist_rank + 2))] = \
                torch.ones(1, self.num_embeddings, 1) * -math.log(self.num_embeddings)
            # pass


class SQVAEPriorCoder(NNPriorCoder):
    def __init__(self, param_var_q="gaussian_1", size_dict=512, dim_dict=64, log_param_q_init=3.0,
                 gs_temp=1.0, gs_temp_anneal=True):
        super(SQVAEPriorCoder, self).__init__()

        self.param_var_q = param_var_q
        self.size_dict = size_dict
        self.dim_dict = dim_dict

        self.gs_temp_anneal = gs_temp_anneal
        if gs_temp_anneal:
            self.gs_temp = nn.Parameter(torch.tensor(gs_temp), requires_grad=False)
        else:
            self.gs_temp = gs_temp

        # Codebook
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.log_param_q_scalar = nn.Parameter(torch.tensor(log_param_q_init))
        if self.param_var_q == "vmf":
            self.quantizer = VmfVectorQuantizer(
                self.size_dict, self.dim_dict, self.gs_temp)
        else:
            self.quantizer = GaussianVectorQuantizer(
                self.size_dict, self.dim_dict, self.gs_temp, self.param_var_q)
        
    
    def forward(self, z_from_encoder, flg_quant_det=True):
        # Encoding
        if self.param_var_q == "vmf":
            self.param_q = (self.log_param_q_scalar.exp() + torch.tensor([1.0], device="cuda"))
        else:
            self.param_q = (self.log_param_q_scalar.exp())
        
        # Quantization
        z_quantized, loss_latent, perplexity = self.quantizer(
            z_from_encoder, self.param_q, self.codebook, self.training, flg_quant_det)
        
        if self.training:
            self.update_cache("loss_dict",
                loss_rate = loss_latent,
            )
            self.update_cache("moniter_dict", 
                embedding_variance_mean=self.param_q.mean(),
            )

        self.update_cache("metric_dict", 
            perplexity=perplexity.sum() # / B
        )

        # TODO: add kl entropy metric?
        prior_entropy = math.log(self.size_dict) * (z_quantized.numel() / self.dim_dict) / z_quantized.shape[0]
        self.update_cache("metric_dict", 
            prior_entropy=prior_entropy,
        )

        return z_quantized
    
    def _calc_loss(self):
        raise NotImplementedError()
    