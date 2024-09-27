import math
import itertools
import copy
from typing import Optional, Tuple, Dict, List, Any, Iterable
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from tqdm import tqdm
import time

from pytorch_msssim import ms_ssim

from .base import EntropyCoder
from .ans import ANSEntropyCoder
from cbench.nn.base import NNTrainableModule, DynamicNNTrainableModule
from cbench.nn.utils import batched_cross_entropy
from cbench.nn.layers.param_generator import IndexParameterGenerator
from cbench.utils.bytes_ops import merge_bytes, split_merged_bytes
from cbench.benchmark.metrics.bj_delta import BJDeltaMetric
from cbench.utils.logging_utils import MetricLogger, SmoothedValue
# from cbench.benchmark.utils import bj_delta

from cbench.codecs.base import VariableRateCodecInterface, VariableComplexityCodecInterface, VariableTaskCodecInterface

# class BasicLatentGraphicalInferenceModel(DynamicNNTrainableModule):
#     def __init__(self):
#         super().__init__()

#     def forward(self, *args, **kwargs):
#         return super().forward(*args, **kwargs)
    
#     def predict(self, **kwargs):
#         raise NotImplementedError()
    
#     def sample(self, **kwargs):
#         raise NotImplementedError()


# class BasicLatentGraphicalGenerativeModel(DynamicNNTrainableModule):
#     def __init__(self):
#         super().__init__()

#     def forward(self, sample, **kwargs):
#         raise NotImplementedError()

#     def encode(self, sample, **kwargs):
#         raise NotImplementedError()

#     def decode(self, byte_string, **kwargs):
#         raise NotImplementedError()

class BasicLatentGraphicalNodeAggregatorModel(DynamicNNTrainableModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_list : List[Any], *args, **kwargs):
        raise NotImplementedError()


class AverageNodeAggregatorModel(BasicLatentGraphicalNodeAggregatorModel):
    def forward(self, input_list: List[Any], *args, **kwargs):
        return torch.stack(input_list).mean(0)


class LossyDummyEntropyCoder(EntropyCoder, NNTrainableModule):
    """A dummy sub entropy coder for lossy entropy coders. Usually for calculating distortion loss.
    """    
    def __init__(self, *args,
                 lambda_rd=1.0,
                 distortion_type="mse",
                 **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)
        self.lambda_rd = lambda_rd
        self.distortion_type = distortion_type

    # NOTE: we use (sum of loss / batch size) here to match with prior coders
    def _calc_loss_distortion(self, x_hat, x, weight=None):
        if weight is None: 
            weight = torch.ones(x.shape[0]).type_as(x) / x.shape[0]

        if self.distortion_type == "none":
            self.update_cache("metric_dict", estimated_x_epd=0)
            return None
        elif self.distortion_type == "mse":
            mse_batch = (x_hat - x).pow(2).view(x.shape[0], -1) # F.mse_loss(x_hat, x)
            loss_distortion = (mse_batch.sum(dim=-1) * weight).sum()
            self.update_cache("metric_dict", mse=mse_batch.mean())
        elif self.distortion_type == "ms-ssim":
            num_elem = x.numel() / x.shape[0]
            msssim_batch = ms_ssim(x_hat, x, data_range=1.0, size_average=False)
            loss_distortion = ((1 - msssim_batch) * weight).sum() * num_elem
            self.update_cache("metric_dict", ms_ssim=msssim_batch.mean())
        elif self.distortion_type == "ce":
            ce_batch = batched_cross_entropy(x_hat, x).view(x.shape[0], -1)
            loss_distortion = (ce_batch.sum(dim=-1) * weight).sum()
            # update coding length metric
            # if not self.training:
            self.update_cache("metric_dict", estimated_x_epd=loss_distortion)
        elif self.distortion_type == "normal":
            mean, logvar = x_hat.chunk(2, dim=1)
            mse = (mean - x).pow(2) 
            loss = mse / (2*logvar.exp()) + logvar / 2
            loss_batch = loss.reshape(x.shape[0], -1).sum(dim=-1)
            loss_distortion = (loss_batch * weight).sum()
            self.update_cache("metric_dict", mse=mse.mean())
        else:
            raise NotImplementedError("")

        return loss_distortion

    def forward(self, data, *args, prior=None, lambda_rd=None, prior_target=None, **kwargs):
        if prior_target is None:
            prior_target = data
        if lambda_rd is None:
            lambda_rd = self.lambda_rd
            
        if prior.shape[2:] != data.shape[:2]:
            for dim, size in enumerate(data.shape[2:], 2):
                prior = prior.narrow(dim, 0, size)
        loss_distortion = self._calc_loss_distortion(prior, prior_target)

        if self.training:
            losses = dict()
            if loss_distortion is not None:
                losses.update(loss_distortion=loss_distortion * lambda_rd)

            self.update_cache("loss_dict", **losses)
        if loss_distortion is not None:
            self.update_cache("metric_dict", weighted_distortion=loss_distortion * lambda_rd) # changed to log2 scale

        return prior
    
    def encode(self, data, *args, prior=None, **kwargs) -> bytes:
        return b""
        # raise NotImplementedError()
    
    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        # TODO: maximum likelihood from prior for probablistic decoding?
        return prior
        # raise NotImplementedError()


class NNBasedLossyDummyEntropyCoder(LossyDummyEntropyCoder):
    def __init__(self, *args, 
                 model : Optional[nn.Module] = None, 
                 use_perceptual_loss : bool = False, 
                 perceptual_loss_layers : Optional[List[str]] = None,
                 monitor_metric_methods : Optional[List[str]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = self._build_model() if model is None else model
        # freeze nn
        for k, p in self.nn.named_parameters():
            # p.requires_grad = False
            p.lr_modifier = 0.0
        # self.nn.eval()
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_loss_layers = perceptual_loss_layers
        self.monitor_metric_methods = monitor_metric_methods

        if self.use_perceptual_loss:
            self.perceptual_layers_cache = []
            assert isinstance(perceptual_loss_layers, list)
            modules = {k:v for k,v in self.nn.named_modules()}
            for layer_name in perceptual_loss_layers:
                layer = modules.get(layer_name)
                layer.register_forward_hook(self._append_perceptual_layers_cache)

    def _append_perceptual_layers_cache(self, model, input, output):
        self.perceptual_layers_cache.append(output)

    def _build_model(self):
        raise NotImplementedError()

    def _calc_loss_distortion(self, x_hat, x, weight=None):
        # NOTE: nn-based distortion is computation heavy! so we skip this during inference!
        if self.training:
            if self.use_perceptual_loss:
                perceptual_output_x_hat, perceptual_output_x = [], []
                if len(self.perceptual_loss_layers) == 0:
                    # get perceptual_layers from output
                    perceptual_output_x_hat = self.nn(x_hat)
                    perceptual_output_x = self.nn(x)
                else:
                    # get perceptual_layers from cache
                    output_x_hat = self.nn(x_hat)
                    perceptual_output_x_hat = [out.clone() for out in self.perceptual_layers_cache]
                    self.perceptual_layers_cache = []
                    output_x = self.nn(x)
                    perceptual_output_x = [out.clone() for out in self.perceptual_layers_cache]
                    self.perceptual_layers_cache = []

                loss_all = []
                for output_x_hat, output_x in zip(perceptual_output_x_hat, perceptual_output_x):
                    # loss_all.append(super()._calc_loss_distortion(output_x_hat, output_x, weight))
                    loss_all.append(F.mse_loss(output_x_hat, output_x))
                # NOTE: we use (sum of loss / batch size) here to match with prior coders
                loss = sum(loss_all) / len(loss_all) * x_hat.numel() / x_hat.shape[0]
            else:
                output_x_hat = self.nn(x_hat)
                output_x = self.nn(x)
                loss = super()._calc_loss_distortion(output_x_hat, output_x, weight)

            if self.monitor_metric_methods is not None:
                with torch.no_grad():
                    for method in self.monitor_metric_methods:
                        if method == "mse":
                            mse = F.mse_loss(x_hat, x)
                            self.update_cache("metric_dict", mse=mse)
                        elif method == "output_ce":
                            cross_entropy = -(F.softmax(output_x_hat, dim=-1) * F.log_softmax(output_x, dim=-1)).sum(-1).mean()
                            self.update_cache("metric_dict", output_cross_entropy=cross_entropy)
                        else:
                            raise NotImplementedError(f"monitor_metric_method {method} not implemented!")
        else:
            loss = None
        
        return loss


class CombinedLossyDummyEntropyCoder(LossyDummyEntropyCoder):
    def __init__(self, coders : List[LossyDummyEntropyCoder], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coders = nn.ModuleList(coders)

    def forward(self, data, *args, prior=None, lambda_rd=None, prior_target=None, blend_weight=None, **kwargs):
        # TODO: use different lambda_rd for each coder?
        if lambda_rd is None:
            lambda_rd = self.lambda_rd

        if blend_weight is None:
            blend_weight = [1.] * len(self.coders)

        loss_distortion_total = 0
        for i, coder in enumerate(self.coders):
            if blend_weight[i] == 0: continue
            # TODO: split prior/prior_target for each coder?
            _ = coder(data, *args, prior=prior, prior_target=prior_target, **kwargs)
            if self.training:
                loss_distortion = coder.get_raw_cache("loss_dict").pop("loss_distortion", None)
                if loss_distortion is not None:
                    loss_distortion_total = loss_distortion_total + loss_distortion * blend_weight[i]

        if self.training:
            self.update_cache("loss_dict", loss_distortion=loss_distortion_total * lambda_rd)

        return prior


class StraightForwardDummyEntropyCoder(EntropyCoder, NNTrainableModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

    def forward(self, data, *args, prior=None, **kwargs):
        return prior
    
    def encode(self, data, *args, prior=None, **kwargs) -> bytes:
        return b""
    
    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        # TODO: maximum likelihood from prior
        return prior


class ParamDictModuleWrapper(nn.Module):
    """A container to save dict parameters in module state_dict
    """    
    def __init__(self, kwargs):
        super().__init__()
        self.none_params = []
        for name, param in kwargs.items():
            # tmp fix for None param
            if param is None:
                self.none_params.append(name)
            elif isinstance(param, (list, tuple)):
                param_dict = OrderedDict()
                for i,p in enumerate(param): param_dict[str(i)] = p
                setattr(self, name, ParamListModuleWrapper(param_dict))
            elif isinstance(param, (dict)):
                setattr(self, name, ParamDictModuleWrapper(param))
            else:
                self.register_buffer(name, torch.as_tensor(param).detach())

    def forward(self, *args, **kwargs):
        param_dict = OrderedDict() 
        for name in self.none_params: param_dict[name] = None
        param_dict.update(**self._buffers)
        for name, param in self._modules.items():
            if isinstance(param, ParamListModuleWrapper):
                param_dict[name] = list(param().values())
            elif isinstance(param, ParamDictModuleWrapper):
                param_dict[name] = param()
        return param_dict

# TODO:
class ParamListModuleWrapper(ParamDictModuleWrapper):
    def __init__(self, kwargs):
        super().__init__(kwargs)


class LatentGraphicalANSEntropyCoder(ANSEntropyCoder, DynamicNNTrainableModule, VariableRateCodecInterface, VariableComplexityCodecInterface, VariableTaskCodecInterface):
    """
    LatentGraphicalANSEntropyCoder is a Bayesian graph based entropy coder. Supports both lossy and lossless compression. Supports codecs with variable rate, complexity and task.
    
    During forward, all of inference, generative and entropy coding processes are performed to get losses.
    
    During encode, all of inference, generative (except for input node) and entropy coding (encode mode) processes are performed.
    
    During decode, generative and entropy coding (decode mode) processes are performed.
    
    During post_training_process, greedy search of parameters for complexity levels could be optionally performed.
    
    Note that many functions and parameters are experimental and may not work properly!
    """
    DEFAULT_EDGE_SPLIT_SYMBOL="_"
    DEFAULT_PRIOR_KEY_NAME="prior"
    DEFAULT_PRIOR_TARGET_KEY_NAME="prior_target"
    DEFAULT_UNCONDITIONAL_NODE_NAME="u"
    # TODO: can it support multiple input nodes?
    DEFAULT_INPUT_NODE_NAME="x"
    DEFAULT_LAMBDA_FLOP_NODE_NAME="lambdaC"
    DEFAULT_RATE_LEVEL_NODE_NAME="vrlevel"
    DEFAULT_COMPLEX_LEVEL_NODE_NAME="sclevel"
    DEFAULT_TASK_INDEX_NODE_NAME="taskidx"
    
    def __init__(self, *args,
                 use_lossy_compression=True,
                 lossy_compression_lambda_rd=1.0,
                 lossy_compression_distortion_type="mse",
                #  lossy_compression_bj_delta_metric : Optional[BJDeltaMetric] = None,
                 node_generator_dict : Dict[str, DynamicNNTrainableModule] = dict(), 
                 node_generator_input_mapping : Optional[Dict[str, Dict[str, str]]] = dict(),
                 dynamic_node_generator_dict : Dict[str, IndexParameterGenerator] = dict(), 
                 latent_node_entropy_coder_dict : Dict[str, EntropyCoder] = dict(), 
                 latent_inference_dict : Dict[str, DynamicNNTrainableModule] = dict(), 
                 latent_generative_dict : Dict[str, DynamicNNTrainableModule] = dict(), 
                 latent_inference_node_aggregator_dict : Optional[Dict[str, BasicLatentGraphicalNodeAggregatorModel]] = dict(), 
                 latent_generative_node_aggregator_dict : Optional[Dict[str, BasicLatentGraphicalNodeAggregatorModel]] = dict(), 
                 latent_inference_input_mapping : Optional[Dict[str, Dict[str, str]]] = dict(),
                 latent_generative_input_mapping : Optional[Dict[str, Dict[str, str]]] = dict(),
                 latent_node_inference_topo_order : Optional[List[str]] = None,
                 latent_node_generative_topo_order : Optional[List[str]] = None,
                 normalize_loss=False,
                 use_sandwich_self_supervised_training=False,
                 moniter_node_generator_output=False,
                 complexity_metric_list=None,
                 nn_complexity_metric=None,
                 complexity_level_greedy_search=False,
                 complexity_level_greedy_search_dataset : Optional[Iterable] =None,
                 complexity_level_greedy_search_dataset_cached=False,
                 complexity_level_greedy_search_iterative=False,
                 complexity_level_greedy_search_num_levels : Optional[int] = None,
                 complexity_level_greedy_search_custom_constraint : Optional[List[float]] = None,
                 complexity_level_greedy_search_performance_metric : Optional[str] = None,
                 complexity_level_greedy_search_complexity_metric : Optional[str] = None,
                 complexity_level_greedy_search_add_controller_nodes_as_complexity_metric=True,
                 complexity_level_controller_nodes=[],
                 complexity_level_greedy_search_custom_params: Optional[Dict[str, Dict[str, int]]] = None,
                 lambda_flops=0.0,
                 auto_adjust_lambda_flops=False,
                 auto_adjust_lambda_flops_method="rejection",
                 auto_adjust_lambda_flops_rejection_weight=1e-6,
                 auto_adjust_lambda_flops_linear_loss_weight=1e-8,
                 auto_adjust_lambda_flops_default_min_flops_perdim=0,
                 auto_adjust_lambda_flops_default_max_flops_perdim=0,
                 flop_per_dim_limit=0.0,
                 use_relative_flops_loss=False,
                 relative_flops_loss_lambdas : Optional[List[float]] = None,
                 freeze_inference_modules=False,
                 freeze_inference_module_names : Optional[str] = None,
                 freeze_generative_modules=False,
                 freeze_generative_module_names : Optional[str] = None,
                 freeze_latent_entropy_coders=False,
                 freeze_latent_entropy_coder_names : Optional[str] = None,
                 gradient_clipping_group=None,
                 task_names : Optional[List[str]] = None,
                #  num_rate_levels=1,
                #  num_complex_levels=1,
                 **kwargs):
        """_summary_

        Args:
            use_lossy_compression (bool, optional): Affects whether input node is generated during encoding. Also enables the following "lossy_compression_" prefixed params. Defaults to True.
            lossy_compression_lambda_rd (float, optional): Weight on distortion loss. Only enabled when use_lossy_compression=True. Defaults to 1.0.
            lossy_compression_distortion_type (str, optional): Distortion loss type. See LossyDummyEntropyCoder. Only enabled when use_lossy_compression=True. Defaults to "mse".
            node_generator_dict (Dict[str, DynamicNNTrainableModule], optional): Node generators, which generates nodes shared between encoder/decoder. Defaults to dict().
            node_generator_input_mapping (Optional[Dict[str, Dict[str, str]]], optional): Input mapping for node generators, allows custom edge connections. Defaults to dict().
            dynamic_node_generator_dict (Dict[str, IndexParameterGenerator], optional): Dynamic node are discrete variables generated only during forward (training/validation). During coding those nodes are passed from caller or retrieved from inner parameters (from VariableRate/Complexity/Task). Defaults to dict().
            latent_node_entropy_coder_dict (Dict[str, EntropyCoder], optional): Entropy coders for Bayesian nodes. Losses are obtained from them during training. Defaults to dict().
            latent_inference_dict (Dict[str, DynamicNNTrainableModule], optional): Inference Bayesian edges. Names of those edges should be {node1}_{node2}, which represent an edge between node1 and node2. Defaults to dict().
            latent_generative_dict (Dict[str, DynamicNNTrainableModule], optional): Generative Bayesian edges. Names of those edges should be {node1}_{node2}, which represent an edge between node1 and node2. Defaults to dict().
            latent_inference_node_aggregator_dict (Optional[Dict[str, BasicLatentGraphicalNodeAggregatorModel]], optional): Multiple edge aggregator, seldom used. Usually we use input_mapping instead. Defaults to dict().
            latent_generative_node_aggregator_dict (Optional[Dict[str, BasicLatentGraphicalNodeAggregatorModel]], optional): Multiple edge aggregator, seldom used. Usually we use input_mapping instead. Defaults to dict().
            latent_inference_input_mapping (Optional[Dict[str, Dict[str, str]]], optional): Input mapping for inference edges, allows custom edge connections. Defaults to dict().
            latent_generative_input_mapping (Optional[Dict[str, Dict[str, str]]], optional): Input mapping for generative edges, allows custom edge connections. Defaults to dict().
            latent_node_inference_topo_order (Optional[List[str]], optional): Topological order for inference Bayesian nodes . Defaults to None.
            latent_node_generative_topo_order (Optional[List[str]], optional): Topological order for generative Bayesian nodes. Defaults to None.
            normalize_loss (bool, optional): Whether to normalize loss with input spatial size. Defaults to False.
            use_sandwich_self_supervised_training (bool, optional): Do not change. Defaults to False.
            moniter_node_generator_output (bool, optional): Do not change. Defaults to False.
            complexity_metric_list (_type_, optional): Do not change. Defaults to None.
            nn_complexity_metric (_type_, optional): Do not change. Defaults to None.
            complexity_level_greedy_search (bool, optional): Whether to enable complexity level greedy search during post_training_process. Defaults to False.
            complexity_level_greedy_search_dataset (Iterable, optional): The dataset used for greedy search. Defaults to None.
            complexity_level_greedy_search_dataset_cached (bool, optional): Whether to cache the dataset into memory. Defaults to False.
            complexity_level_greedy_search_iterative (bool, optional): Whether to apply iterative greedy search, may not work. Defaults to False.
            complexity_level_greedy_search_num_levels (int, optional): Num complexity levels to search. Defaults to None.
            complexity_level_greedy_search_custom_constraint (List[float], optional): Custom constraint on complexity metric. By default a linear interpolation between min and max complexity is used. Defaults to None.
            complexity_level_greedy_search_performance_metric (str, optional): Performance metric, by default the R-D loss. Defaults to None.
            complexity_level_greedy_search_complexity_metric (str, optional): Complexity metric, by default the FLOPs (or to be accurate, MACs?). Defaults to None.
            complexity_level_greedy_search_add_controller_nodes_as_complexity_metric (bool, optional): Do not change. Defaults to True.
            complexity_level_controller_nodes (list, optional): Nodes to apply greedy search. Those nodes must be implemented with IndexParameterGenerator. Defaults to [].
            complexity_level_greedy_search_custom_params (Dict[str, Dict[str, int]], optional): Manually initialize searched params. Defaults to None.
            lambda_flops (float, optional): Weight for FLOPs loss. Defaults to 0.0.
            auto_adjust_lambda_flops (bool, optional): Do not change. Defaults to False.
            auto_adjust_lambda_flops_method (str, optional): Do not change. Defaults to "rejection".
            auto_adjust_lambda_flops_rejection_weight (_type_, optional): Do not change. Defaults to 1e-6.
            auto_adjust_lambda_flops_linear_loss_weight (_type_, optional): Do not change. Defaults to 1e-8.
            auto_adjust_lambda_flops_default_min_flops_perdim (int, optional): Do not change. Defaults to 0.
            auto_adjust_lambda_flops_default_max_flops_perdim (int, optional): Do not change. Defaults to 0.
            flop_per_dim_limit (float, optional): Do not change. Defaults to 0.0.
            use_relative_flops_loss (bool, optional): Use linear interpolate flops (between min and max flops) as FLOPs loss. Defaults to False.
            relative_flops_loss_lambdas (List[float], optional): Weight for FLOPs loss for each complexity level. Defaults to None.
            freeze_inference_modules (bool, optional): Defaults to False.
            freeze_inference_module_names (str, optional): Names of modules to freeze. Defaults to None.
            freeze_generative_modules (bool, optional): Defaults to False.
            freeze_generative_module_names (str, optional): Names of modules to freeze. Defaults to None.
            freeze_latent_entropy_coders (bool, optional): Defaults to False.
            freeze_latent_entropy_coder_names (str, optional): Names of modules to freeze. Defaults to None.
            gradient_clipping_group (_type_, optional): Do not change. Defaults to None.
            task_names (Optional[List[str]], optional): Task names for VariableTaskInterface. Defaults to None.
        """        
        super().__init__(*args, **kwargs)
        DynamicNNTrainableModule.__init__(self)

        self.use_lossy_compression = use_lossy_compression
        self.lossy_compression_lambda_rd = lossy_compression_lambda_rd
        self.lossy_compression_distortion_type = lossy_compression_distortion_type
        # self.lossy_compression_bj_delta_metric = lossy_compression_bj_delta_metric
        if self.use_lossy_compression:
            if not self.DEFAULT_INPUT_NODE_NAME in latent_node_entropy_coder_dict:
                latent_node_entropy_coder_dict[self.DEFAULT_INPUT_NODE_NAME] = LossyDummyEntropyCoder(
                    lambda_rd=lossy_compression_lambda_rd,
                    distortion_type=lossy_compression_distortion_type,
                )
            # if self.lossy_compression_bj_delta_metric is not None:
            #     if not self.DEFAULT_RATE_LEVEL_NODE_NAME in latent_node_entropy_coder_dict:
            #         self.logger.warning(f"{self.DEFAULT_RATE_LEVEL_NODE_NAME} node not setup! bj_delta_metric is disabled!")
            #         self.lossy_compression_bj_delta_metric = None
            #     else:
            #         self.bj_delta_metric_logger_dict = dict()
                
        self.normalize_loss = normalize_loss

        self.node_generators = nn.ModuleDict(node_generator_dict)
        self.node_generator_input_mapping = node_generator_input_mapping
        self.dynamic_node_generators = nn.ModuleDict(dynamic_node_generator_dict)
        self.latent_node_entropy_coders = nn.ModuleDict(latent_node_entropy_coder_dict) # TODO: enable non-NN entropy coder?
        self.latent_inference_modules = nn.ModuleDict(latent_inference_dict)
        self.latent_generative_modules = nn.ModuleDict(latent_generative_dict)
        # TODO: consider using multiple inputs on entropy coder instead of using aggregators
        self.latent_inference_node_aggregator_modules = nn.ModuleDict(latent_inference_node_aggregator_dict)
        self.latent_generative_node_aggregator_modules = nn.ModuleDict(latent_generative_node_aggregator_dict)
        self.latent_inference_input_mapping = latent_inference_input_mapping
        self.latent_generative_input_mapping = latent_generative_input_mapping

        # TODO: auto determine topo order from edges
        self.latent_node_inference_topo_order = latent_node_inference_topo_order
        self.latent_node_generative_topo_order = latent_node_generative_topo_order

        self.use_sandwich_self_supervised_training = use_sandwich_self_supervised_training
        self.moniter_node_generator_output = moniter_node_generator_output
        self.complexity_metric_list = [] if complexity_metric_list is None else complexity_metric_list
        self.nn_complexity_metric = nn_complexity_metric
        
        self.complexity_level_greedy_search = complexity_level_greedy_search
        self.complexity_level_greedy_search_dataset = complexity_level_greedy_search_dataset
        self.complexity_level_greedy_search_dataset_cached = complexity_level_greedy_search_dataset_cached
        self.complexity_level_greedy_search_iterative = complexity_level_greedy_search_iterative
        self.complexity_level_greedy_search_custom_constraint = complexity_level_greedy_search_custom_constraint
        self.complexity_level_greedy_search_num_levels = complexity_level_greedy_search_num_levels
        self.complexity_level_greedy_search_performance_metric = complexity_level_greedy_search_performance_metric
        self.complexity_level_greedy_search_complexity_metric = complexity_level_greedy_search_complexity_metric
        self.complexity_level_greedy_search_add_controller_nodes_as_complexity_metric = complexity_level_greedy_search_add_controller_nodes_as_complexity_metric
        self.complexity_level_controller_nodes = complexity_level_controller_nodes
        self.complexity_level_greedy_search_custom_params = complexity_level_greedy_search_custom_params
        # TODO: check complexity_level_controller_nodes

        self.lambda_flops = lambda_flops
        self.auto_adjust_lambda_flops = auto_adjust_lambda_flops
        self.auto_adjust_lambda_flops_method = auto_adjust_lambda_flops_method
        self.auto_adjust_lambda_flops_rejection_weight = auto_adjust_lambda_flops_rejection_weight
        self.auto_adjust_lambda_flops_linear_loss_weight = auto_adjust_lambda_flops_linear_loss_weight
        self.flop_per_dim_limit = flop_per_dim_limit
        self.use_relative_flops_loss = use_relative_flops_loss
        self.relative_flops_loss_lambdas = relative_flops_loss_lambdas
        
        self.freeze_inference_modules = freeze_inference_modules
        self.freeze_generative_modules = freeze_generative_modules
        self.freeze_latent_entropy_coders = freeze_latent_entropy_coders
        self.gradient_clipping_group = gradient_clipping_group

        self.task_names = task_names

        if self.use_sandwich_self_supervised_training or self.use_relative_flops_loss:
            self.dynamic_node_range = dict()
            for name, module in self.dynamic_node_generators.items():
                # remove min and max from generation
                self.dynamic_node_range[name] = (module.min_sample, module.max_sample)
                module.min = module.min + 1
                module.max = module.max - 1

        if self.auto_adjust_lambda_flops and self.auto_adjust_lambda_flops_method == "linear":
            self.register_buffer("min_flops_perdim", torch.tensor([float(auto_adjust_lambda_flops_default_min_flops_perdim)]))
            self.register_buffer("max_flops_perdim", torch.tensor([float(auto_adjust_lambda_flops_default_max_flops_perdim)]))

        if self.freeze_inference_modules:
            modules = [self.latent_inference_modules[m] for m in freeze_inference_module_names] \
                if freeze_inference_module_names is not None else self.latent_inference_modules.values()
            for m in modules:
                for param in m.parameters():
                    param.requires_grad = False
            for param in self.latent_inference_node_aggregator_modules.parameters():
                param.requires_grad = False
        
        if self.freeze_generative_modules:
            modules = [self.latent_generative_modules[m] for m in freeze_generative_module_names] \
                if freeze_generative_module_names is not None else self.latent_generative_modules.values()
            for m in modules:
                for param in m.parameters():
                    param.requires_grad = False
            for param in self.latent_generative_node_aggregator_modules.parameters():
                param.requires_grad = False
        
        if self.freeze_latent_entropy_coders:
            modules = [self.latent_node_entropy_coders[m] for m in freeze_latent_entropy_coder_names] \
                if freeze_latent_entropy_coder_names is not None else self.latent_node_entropy_coders.values()
            for m in modules:
                for param in m.parameters():
                    param.requires_grad = False
                    
        if self.gradient_clipping_group is not None:
            for param in self.parameters():
                param.gradient_clipping_group = self.gradient_clipping_group
        
        # create module iterator according to topo order
        self.latent_inference_in_node_dict = {node_name : [] for node_name in self.latent_node_inference_topo_order}
        self.latent_inference_out_node_dict = {node_name : [] for node_name in self.latent_node_inference_topo_order}
        for edge_name, module in self.latent_inference_modules.items():
            inode, onode = edge_name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
            self.latent_inference_in_node_dict[inode].append(edge_name)
            self.latent_inference_out_node_dict[onode].append(edge_name)
        self.latent_generative_in_node_dict = {node_name : [] for node_name in self.latent_node_generative_topo_order}
        self.latent_generative_out_node_dict = {node_name : [] for node_name in self.latent_node_generative_topo_order}
        for edge_name, module in self.latent_generative_modules.items():
            inode, onode = edge_name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
            self.latent_generative_in_node_dict[inode].append(edge_name)
            self.latent_generative_out_node_dict[onode].append(edge_name)

        # TODO: assert no loops

        # assert self.DEFAULT_INPUT_NODE_NAME in self.latent_nodes
        # self.latent_nodes.remove(self.DEFAULT_INPUT_NODE_NAME)
            
            
        # vr/vc/vt codec interface impl
        self._num_rate_levels = 0
        self._num_complex_levels = 0
        self._num_tasks = 0
        if self.DEFAULT_RATE_LEVEL_NODE_NAME in self.dynamic_node_generators:
            dng = self.dynamic_node_generators[self.DEFAULT_RATE_LEVEL_NODE_NAME]
            self._num_rate_levels = dng.max_sample - dng.min_sample + 1
        if self._num_rate_levels > 0:
            self._current_rate_level = -1
            # self.register_buffer("_current_rate_level", torch.zeros(1, dtype=torch.long))
        if self.DEFAULT_TASK_INDEX_NODE_NAME in self.dynamic_node_generators:
            dng = self.dynamic_node_generators[self.DEFAULT_TASK_INDEX_NODE_NAME]
            self._num_tasks = dng.max_sample - dng.min_sample + 1
        if self._num_tasks > 0:
            self._current_task_idx = -1
            if self.task_names is None:
                self.task_names = list(range(self._num_tasks))
            assert len(self.task_names) == self._num_tasks
        
        if self.complexity_level_greedy_search:
            if self.complexity_level_greedy_search_num_levels is not None:
                self._num_complex_levels = self.complexity_level_greedy_search_num_levels
            elif self.complexity_level_greedy_search_custom_constraint is not None:
                self._num_complex_levels = len(self.complexity_level_greedy_search_custom_constraint)
            
            complexity_param_all_levels = []
            max_complexity_param = dict()
            complexity_param_valid = False
            if self.complexity_level_greedy_search_custom_params is not None:
                for level, complexity_idx in enumerate(self.complexity_level_greedy_search_custom_params):
                    complexity_param = dict()
                    for module_name, value in complexity_idx.items():
                        complexity_param[module_name] = self.node_generators[module_name](value)
                    complexity_param_all_levels.append(ParamDictModuleWrapper(complexity_param))
                # complexity_param_valid = True
            else:
                for param_name in self.complexity_level_controller_nodes:
                    module = self.node_generators[param_name]
                    assert isinstance(module, IndexParameterGenerator)
                    # if self.DEFAULT_EDGE_SPLIT_SYMBOL in name:
                    #     inode, onode = name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
                    #     param_name = onode
                    # else:
                    #     param_name = name
                    max_complexity_param[param_name] = module(module.min_sample)
                for _ in range(self._num_complex_levels):
                    complexity_param_all_levels.append(ParamDictModuleWrapper(max_complexity_param))
            # NOTE: _complexity_param_all_levels and _complexity_param_valid should be updated through post_training_process
            self._complexity_param_all_levels = nn.ModuleList(complexity_param_all_levels)
            # By default _complexity_param_valid is disabled
            self.register_buffer("_complexity_param_valid", torch.tensor([complexity_param_valid]))
            # self._complexity_param_valid = False
            # self._complexity_param_all_levels = []

            if self.complexity_level_greedy_search_complexity_metric is None:
                self.complexity_level_greedy_search_complexity_metric = "FLOPs"
            self.complexity_metric_list.append(self.complexity_level_greedy_search_complexity_metric)
            if self.complexity_level_greedy_search_performance_metric is None:
                self.complexity_level_greedy_search_performance_metric = "loss"
            self.complexity_metric_list.append(self.complexity_level_greedy_search_performance_metric)

            if self.complexity_level_greedy_search_add_controller_nodes_as_complexity_metric:
                self.complexity_metric_list.extend(self.complexity_level_controller_nodes)

        else:
            if self.DEFAULT_COMPLEX_LEVEL_NODE_NAME in self.dynamic_node_generators:
                dng = self.dynamic_node_generators[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME]
                self._num_complex_levels = dng.max_sample - dng.min_sample + 1
        
        if self._num_complex_levels > 0:
            self._current_complex_level = -1
            # self.register_buffer("_current_complex_level", torch.zeros(1, dtype=torch.long))
            if len(self.complexity_metric_list) > 0:
                # a buffer for easy retrieval of complexity during coding test
                self.register_buffer("_complexity_metric_list_cache", torch.zeros(self._num_complex_levels, len(self.complexity_metric_list)))
            elif self.nn_complexity_metric is not None:
                # a buffer for easy retrieval of complexity during coding test
                self.register_buffer("_nn_complexity_metric_cache", torch.zeros(self._num_complex_levels))

    def _get_default_node_dict(self, force_add_default_dynamic_nodes=False, **kwargs):
        output_dict = {
            self.DEFAULT_UNCONDITIONAL_NODE_NAME : None, 
            **kwargs,
        }
        # add rate/complex/task dynamic node
        if not self.training:
            if self._num_rate_levels > 0:
                if not self.DEFAULT_RATE_LEVEL_NODE_NAME in output_dict:
                    if self._current_rate_level >= 0:
                        output_dict[self.DEFAULT_RATE_LEVEL_NODE_NAME] = self._current_rate_level
                    elif force_add_default_dynamic_nodes:
                        self.logger.info("Rate level not set! Using the default one!")
                        output_dict[self.DEFAULT_RATE_LEVEL_NODE_NAME] = 0
            if self._num_tasks > 0:
                if not self.DEFAULT_TASK_INDEX_NODE_NAME in output_dict:
                    if self._current_task_idx >= 0:
                        output_dict[self.DEFAULT_TASK_INDEX_NODE_NAME] = self._current_task_idx
                    elif force_add_default_dynamic_nodes:
                        self.logger.info("Task not set! Using the default one!")
                        output_dict[self.DEFAULT_TASK_INDEX_NODE_NAME] = 0
            if self._num_complex_levels > 0:
                if not self.DEFAULT_COMPLEX_LEVEL_NODE_NAME in output_dict :
                    if self.complexity_level_greedy_search and self._complexity_param_valid:
                        # override complexity params from search
                        output_dict.update(**self._complexity_param_all_levels[self._current_complex_level]())
                        # output_dict.update(**self._complexity_param_all_levels[self._current_complex_level])
                    elif self._current_complex_level >= 0:
                        output_dict[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME] = self._current_complex_level
                    elif force_add_default_dynamic_nodes:
                        self.logger.info("Complexity level not set! Using the default one!")
                        output_dict[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME] = 0

        return output_dict

    # TODO: how to ensure same generation process during encode/decode? Setting a shared seed?
    def _node_generate_process(self, **kwargs):
        output_dict = dict(**kwargs)

        for name, module in self.node_generators.items():

            input_kwargs = dict()
            if name in self.node_generator_input_mapping:
                for node_key, input_key in self.node_generator_input_mapping[name].items():
                    input_kwargs[input_key] = output_dict[node_key]

            if self.DEFAULT_EDGE_SPLIT_SYMBOL in name:
                inode, onode = name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
                # node already known, skip generation
                if onode in output_dict:
                    continue
                if name in output_dict:
                    output_dict[onode] = output_dict.pop(name)
                    continue
                output_dict[onode] = module(output_dict[inode], **input_kwargs)
            else:
                # node already known, skip generation
                if name in output_dict:
                    continue
                output_dict[name] = module(**input_kwargs)

        if self.moniter_node_generator_output:
            for name, param in output_dict.items():
                if name not in kwargs:
                    if isinstance(param, torch.Tensor):
                        self.update_cache("moniter_dict", **{f"{name}_mean" : param.float().mean()})
                    else:
                        self.update_cache("moniter_dict", **{f"{name}" : param})
        
        return output_dict

    def _inference_process(self, input_dict, *args, **kwargs):
        output_dict = dict(**input_dict)

        for node_name in self.latent_node_inference_topo_order:
            # prepare inference kwargs input
            inference_kwargs = dict()
            if node_name in self.latent_inference_input_mapping:
                for node_key, input_key in self.latent_inference_input_mapping[node_name].items():
                    inference_kwargs[input_key] = output_dict[node_key]
            
            # iterate edges to this node
            for edge_name in self.latent_inference_out_node_dict[node_name]:
                # prepare inference kwargs input
                if edge_name in self.latent_inference_input_mapping:
                    for edge_key, input_key in self.latent_inference_input_mapping[edge_name].items():
                        inference_kwargs[input_key] = output_dict[edge_key]
                inode, onode = edge_name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
                assert onode == node_name
                node_out = self.latent_inference_modules[edge_name](output_dict[inode], **inference_kwargs)
                if onode in output_dict:
                    if not isinstance(output_dict[onode], list):
                        output_dict[onode] = [output_dict[onode]]
                    output_dict[onode].append(node_out)
                else:
                    output_dict[onode] = node_out

            # aggregate edges
            if node_name in self.latent_inference_node_aggregator_modules:
                output_dict[node_name] = self.latent_inference_node_aggregator_modules[node_name](output_dict[node_name], *args, **kwargs)

        # assert not any([isinstance(output, list) for output in output_dict.values()]), "Aggregator config incorrect!"
        
        # sampling process
        # NOTE: we follow hyperprior to sample after inference
        # for node_name, coder in self.latent_node_entropy_coders.items():
        #     coder.sample(output_dict[node_name], **kwargs)

        return output_dict

    def _generative_process(self, input_dict, *args, prior_dict : Optional[Dict[str, Dict[str, Any]]] = dict(), prior_target_dict : Optional[Dict[str, Any]] = dict(), do_encode=False, stream=None, **kwargs):
        data_dict = dict(**input_dict)
        # TODO: reconsider prior_dict and kwargs
        # prior_dict = dict()
        kwargs_dict = dict(**input_dict, **kwargs)

        latent_nodes = copy.deepcopy(self.latent_node_generative_topo_order)
        if do_encode and self.use_lossy_compression:
            latent_nodes.remove(self.DEFAULT_INPUT_NODE_NAME)

        for node_name in latent_nodes:

            # node_data = data_dict.get(node_name)
            # profiler_prefix = f"generative_{node_name}"
            # if isinstance(node_data, bytes):
            #     profiler_prefix += "_decode"
            # if do_encode:
            #     profiler_prefix += "_encode"
            # with self.profiler.start_time_profile(profiler_prefix):

            # check if encode/decode is needed
            # node_data = data_dict.get(node_name)
            # if node_name in self.latent_node_entropy_coders:
            #     if (isinstance(node_data, bytes) or do_encode) and isinstance(self.latent_node_entropy_coders[node_name], LossyDummyEntropyCoder):
            #         self.logger.warning(f"Encoding for {node_name} is skipped!")
            #         if do_encode:
            #             # add dummy bytes
            #             data_dict[node_name] = b""
            #         continue

            # prepare prior dict
            if not node_name in prior_dict:
                prior_dict[node_name] = dict()

            # aggregate edges (all prior should be ready when iterating current node)
            if node_name in self.latent_generative_node_aggregator_modules:
                prior_dict[node_name] = self.latent_generative_node_aggregator_modules[node_name](list(prior_dict[node_name].values()))

            # data entropy coding
            node_data = data_dict.get(node_name)
            if node_name in self.latent_node_entropy_coders:
                # prepare prior kwargs input
                prior_kwargs = dict()
                if node_name in self.latent_generative_input_mapping:
                    for node_key, input_key in self.latent_generative_input_mapping[node_name].items():
                        # only have 1 conditional input, redirect to "prior" key
                        if len(prior_dict[node_name]) == 1:
                            prior_kwargs.update(prior=list(prior_dict[node_name].values())[0])
                        if node_key in prior_dict[node_name]:
                            prior_kwargs[input_key] = prior_dict[node_name][node_key]
                        elif node_key in kwargs_dict:
                            prior_kwargs[input_key] = kwargs_dict[node_key]
                        else:
                            raise ValueError(f"latent_generative_input_mapping incorrect! node {node_name}, mapping {node_key} : {input_key}")
                else:
                    assert len(prior_dict[node_name]) <= 1
                    # only have 1 conditional input, redirect to "prior" key
                    if len(prior_dict[node_name]) == 1:
                        prior_kwargs.update(prior=list(prior_dict[node_name].values())[0])

                if node_name in prior_target_dict:
                    assert len(prior_target_dict[node_name]) <= 1
                    if len(prior_target_dict[node_name]) == 1:
                        prior_kwargs[self.DEFAULT_PRIOR_TARGET_KEY_NAME] = list(prior_target_dict[node_name].values())[0]

                # encode/forward/decode from bytes if neccessary
                with self.profiler.start_time_profile(f"latent_node_entropy_coders_{node_name}"):
                    if isinstance(node_data, bytes):
                        node_data = self.latent_node_entropy_coders[node_name].decode(node_data, stream=stream, **prior_kwargs)
                        data_dict[node_name] = node_data
                        kwargs_dict[node_name] = node_data
                    else:
                        # use stream to do quantize and encode simultaneously!
                        if do_encode and stream is not None:
                            node_data = self.latent_node_entropy_coders[node_name].encode(data_dict.get(node_name), stream=stream, **prior_kwargs)
                        else:
                            node_data = self.latent_node_entropy_coders[node_name](data_dict.get(node_name), **prior_kwargs)
                            if do_encode:
                                byte_string = self.latent_node_entropy_coders[node_name].encode(data_dict.get(node_name), **prior_kwargs)
                                data_dict[node_name] = byte_string
                            else:
                                data_dict[node_name] = node_data
                        kwargs_dict[node_name] = node_data # TODO: do we need to use forward(quantized) data for kwargs?

            # iterate edges from this node to generate prior
            for edge_name in self.latent_generative_in_node_dict[node_name]:
                # prepare edge kwargs input
                edge_kwargs = dict()
                if edge_name in self.latent_generative_input_mapping:
                    for edge_key, input_key in self.latent_generative_input_mapping[edge_name].items():
                        edge_kwargs[input_key] = kwargs_dict[edge_key]

                inode, onode = edge_name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
                assert inode == node_name
                # skip modules for estimating input node for lossy compression
                if do_encode and self.use_lossy_compression and onode == self.DEFAULT_INPUT_NODE_NAME:
                    continue
                
                with self.profiler.start_time_profile(f"latent_generative_modules_{edge_name}"):
                    node_out = self.latent_generative_modules[edge_name](node_data, **edge_kwargs)

                # update prior_dict and kwargs_dict
                kwargs_dict[edge_name] = node_out
                if onode in prior_dict:
                    prior_dict[onode][inode] = node_out
                else:
                    prior_dict[onode] = {inode : node_out}

        return data_dict, prior_dict
    
    # def get_current_flops(self, input=None):
    #     return super().get_current_flops(input=input)
    
    def forward(self, data, *args, prior=None, target=None, prior_target_dict=dict(), recursive_mode=False, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        # NOTE: For transform coding, the decompressed data may not be close to itself. 
        # A extra target is passes into prior_target_dict to calculate loss for LossyDummyEntropyCoder.
        if target is not None:
            prior_target_dict[self.DEFAULT_INPUT_NODE_NAME] = target

        # if self.num_rate_levels > 0:
        #     if not self.DEFAULT_RATE_LEVEL_NODE_NAME in kwargs:
        #         for rate_level in range(self.num_rate_levels):
        #             kwargs[self.DEFAULT_RATE_LEVEL_NODE_NAME] = rate_level
        #             ret = self.forward(data, *args, prior=prior, **kwargs)
        
        node_dict = self._get_default_node_dict(**kwargs)
        node_dict[self.DEFAULT_INPUT_NODE_NAME] = data

        logging_prefix = ""
        all_cache_dict = dict()
        dynamic_node_iter_dict = dict()
        is_dynamic_node_complete = True
        # iterate dynamic_node_generators 
        # if all dynamic nodes are known, proceed to the main forward step with logging_prefix
        for name, module in self.dynamic_node_generators.items():
            # node already known, skip generation and add to prefix
            if name in node_dict:
                value = node_dict[name]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                # NOTE: for fixed nodes, we should not include them in logging_prefix
                if not module.fix_for_inference:
                    logging_prefix += f"{name}={round(value)}_"
            else:
                is_dynamic_node_complete = False
                # using samples during training and full iteration during testing
                if self.training:
                    param_iter = module().tolist()
                else:
                    # TODO: tmp solution for fix_for_inference
                    if module.fix_for_inference:
                        param_iter = [module()]
                    else:
                        param_iter = list(range(module.min_sample, module.max_sample+1))
                if self.use_sandwich_self_supervised_training:
                    param_iter.insert(0, self.dynamic_node_range[name][1])
                    param_iter.append(self.dynamic_node_range[name][0])
                    
                dynamic_node_iter_dict[name] = param_iter

        if len(dynamic_node_iter_dict) > 0:

            # if self.use_lossy_compression and self.lossy_compression_bj_delta_metric is not None:
            #     if self.training:
            #         if len(self.bj_delta_metric_logger_dict) > 0:
            #             # TODO: calculate bj_delta and clear cache
            #             self.bj_delta_metric_logger_dict = dict()
            #     else:
            #         if len(self.bj_delta_metric_logger_dict) == 0:
            #             self.bj_delta_metric_logger_dict = {idx:MetricLogger() for idx in dynamic_node_iter_dict[self.DEFAULT_RATE_LEVEL_NODE_NAME]}

            for all_params in itertools.product(*dynamic_node_iter_dict.values()):
                for name, param in zip(dynamic_node_iter_dict.keys(), all_params):
                    node_dict[name] = param
                    
                if self.training and self.use_relative_flops_loss and self.DEFAULT_COMPLEX_LEVEL_NODE_NAME in dynamic_node_iter_dict:
                    # min complexity
                    node_dict[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME] = self.dynamic_node_range[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME][0]
                    (ret_data_dict, ret_prior_dict), current_all_cache_dict = self.forward(data, *args, prior=prior, prior_target_dict=prior_target_dict, recursive_mode=True, **node_dict)
                    if self.training:
                        loss_dict = current_all_cache_dict.pop("loss_dict")
                        loss_total_max = sum([loss for name, loss in loss_dict.items() if name.endswith("loss_rate") or name.endswith("loss_distortion")]).detach()
                        loss_flops_min = self.get_current_flops().detach()
                    for cache_name, cache_dict in current_all_cache_dict.items():
                        if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
                        all_cache_dict[cache_name].update(**cache_dict)
                    # max complexity
                    node_dict[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME] = self.dynamic_node_range[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME][1]
                    (ret_data_dict, ret_prior_dict), current_all_cache_dict = self.forward(data, *args, prior=prior, prior_target_dict=prior_target_dict, recursive_mode=True, **node_dict)
                    if self.training:
                        loss_dict = current_all_cache_dict.pop("loss_dict")
                        loss_total_min = sum([loss for name, loss in loss_dict.items() if name.endswith("loss_rate") or name.endswith("loss_distortion")]).detach()
                        loss_flops_max = self.get_current_flops().detach()
                    for cache_name, cache_dict in current_all_cache_dict.items():
                        if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
                        all_cache_dict[cache_name].update(**cache_dict)
                
                (ret_data_dict, ret_prior_dict), current_all_cache_dict = self.forward(data, *args, prior=prior, prior_target_dict=prior_target_dict, recursive_mode=True, **node_dict)
                # TODO: confirm if we can remove use_sandwich_self_supervised_training!
                # if self.use_sandwich_self_supervised_training and self.training and idx == 0:
                #     # detach all tensors for prior_target
                #     prior_target_dict = dict()
                #     for node, node_input_dict in ret_prior_dict.items():
                #         prior_target_dict[node] = {k:v.detach() for k,v in node_input_dict.items()}
                if self.training and self.use_relative_flops_loss and self.DEFAULT_COMPLEX_LEVEL_NODE_NAME in dynamic_node_iter_dict:
                    loss_dict = current_all_cache_dict.pop("loss_dict")
                    loss_total = sum([loss for name, loss in loss_dict.items() if name.endswith("loss_rate") or name.endswith("loss_distortion")])
                    loss_flops = self.get_current_flops()
                    loss_total = (loss_total - loss_total_min) / (loss_total_max - loss_total_min)
                    loss_flops = (loss_flops - loss_flops_min) / (loss_flops_max - loss_flops_min)
                    if self.relative_flops_loss_lambdas is not None:
                        lambda_flops = self.relative_flops_loss_lambdas[param]
                        loss_flops = loss_flops * lambda_flops
                    current_all_cache_dict["loss_dict"] = dict(
                        loss_total=loss_total,
                        loss_flops=loss_flops,
                    )
                    
                # TODO: calculate bj_delta for VR node
                # if rate_distortion_dict is not None:
                #     vrlevel = node_dict[self.DEFAULT_RATE_LEVEL_NODE_NAME]
                #     metric_dict = current_all_cache_dict.get("metric_dict")
                #     rate_metric = metric_dict.get("prior_entropy")
                #     distortion_metric = metric_dict.get(self.lossy_compression_distortion_type)
                #     rate_distortion_dict[vrlevel][node_dict] = (rate_metric, distortion_metric)

                for cache_name, cache_dict in current_all_cache_dict.items():
                    if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
                    all_cache_dict[cache_name].update(**cache_dict)
                    

            # for name, param_iter in dynamic_node_iter_dict.items():
            #     if name == self.DEFAULT_COMPLEX_LEVEL_NODE_NAME and self.use_relative_flops_loss:
            #         # min complexity
            #         node_dict[name] = self.dynamic_node_range[name][0]
            #         (ret_data_dict, ret_prior_dict), current_all_cache_dict = self.forward(data, *args, prior=prior, prior_target_dict=prior_target_dict, recursive_mode=True, **node_dict)
            #         if self.training:
            #             loss_dict = current_all_cache_dict.pop("loss_dict")
            #             loss_total_max = sum([loss for name, loss in loss_dict.items() if name.endswith("loss_rate") or name.endswith("loss_distortion")]).detach()
            #             loss_flops_min = self.get_current_flops().detach()
            #         for cache_name, cache_dict in current_all_cache_dict.items():
            #             if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
            #             all_cache_dict[cache_name].update(**cache_dict)
            #         # max complexity
            #         node_dict[name] = self.dynamic_node_range[name][1]
            #         (ret_data_dict, ret_prior_dict), current_all_cache_dict = self.forward(data, *args, prior=prior, prior_target_dict=prior_target_dict, recursive_mode=True, **node_dict)
            #         if self.training:
            #             loss_dict = current_all_cache_dict.pop("loss_dict")
            #             loss_total_min = sum([loss for name, loss in loss_dict.items() if name.endswith("loss_rate") or name.endswith("loss_distortion")]).detach()
            #             loss_flops_max = self.get_current_flops().detach()
            #         for cache_name, cache_dict in current_all_cache_dict.items():
            #             if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
            #             all_cache_dict[cache_name].update(**cache_dict)

            #     for idx, param in enumerate(param_iter):
            #         node_dict[name] = param
            #         (ret_data_dict, ret_prior_dict), current_all_cache_dict = self.forward(data, *args, prior=prior, prior_target_dict=prior_target_dict, recursive_mode=True, **node_dict)
            #         # TODO: is this proper to set prior_target as self-supervision?
            #         if self.use_sandwich_self_supervised_training and self.training and idx == 0:
            #             # detach all tensors for prior_target
            #             prior_target_dict = dict()
            #             for node, node_input_dict in ret_prior_dict.items():
            #                 prior_target_dict[node] = {k:v.detach() for k,v in node_input_dict.items()}
            #         if self.training and name == self.DEFAULT_COMPLEX_LEVEL_NODE_NAME and self.use_relative_flops_loss:
            #             loss_dict = current_all_cache_dict.pop("loss_dict")
            #             loss_total = sum([loss for name, loss in loss_dict.items() if name.endswith("loss_rate") or name.endswith("loss_distortion")])
            #             loss_flops = self.get_current_flops()
            #             loss_total = (loss_total - loss_total_min) / (loss_total_max - loss_total_min)
            #             loss_flops = (loss_flops - loss_flops_min) / (loss_flops_max - loss_flops_min)
            #             if self.relative_flops_loss_lambdas is not None:
            #                 lambda_flops = self.relative_flops_loss_lambdas[param]
            #                 loss_flops = loss_flops * lambda_flops
            #             current_all_cache_dict["loss_dict"] = dict(
            #                 loss_total=loss_total,
            #                 loss_flops=loss_flops,
            #             )
            #         for cache_name, cache_dict in current_all_cache_dict.items():
            #             if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
            #             all_cache_dict[cache_name].update(**cache_dict)

        # if not is_dynamic_node_complete:
        #     return (dict(), dict()), dict()
        
        # If all dynamic nodes are iterated in recursive_mode, update all caches in main call and return
        # TODO: better return value?
        if not recursive_mode and len(all_cache_dict) > 0: # len(logging_prefix) == 0 
            for cache_name, cache_dict in all_cache_dict.items():
                self.update_cache(cache_name, **cache_dict)
            # auto_adjust_lambda_flops
            if self.training and self.auto_adjust_lambda_flops and self.auto_adjust_lambda_flops_method == "rejection":
                cache_dict = all_cache_dict["common"]
                flops_perdim_dict = dict()
                lambda_flops_dict = dict()
                for k,v in cache_dict.items():
                    if k.endswith("flops_perdim"):
                        prefix_name = k[:-(len("flops_perdim")+1)]
                        flops_perdim_dict[prefix_name] = v
                        # self.update_cache("moniter_dict", **{f"{prefix_name}_flops_perdim" : v})
                    if k.endswith("lambda_flops"):
                        prefix_name = k[:-(len("lambda_flops")+1)]
                        lambda_flops_dict[prefix_name] = v
                        # self.update_cache("moniter_dict", **{f"{prefix_name}_lambda_flops" : v})
                # add a rejection loss
                assert len(flops_perdim_dict) > 1, "rejection requires at least 2 samples."
                for prefix_name1, prefix_name2 in itertools.combinations(flops_perdim_dict.keys(), 2):
                    flops_delta = flops_perdim_dict[prefix_name1] - flops_perdim_dict[prefix_name2]
                    lambda_delta = lambda_flops_dict[prefix_name1] - lambda_flops_dict[prefix_name2]
                    # NOTE: we need to increase lambda_delta more if flops_delta is close to 0
                    # so we set rejection_loss as lambda_delta / (|flops_delta| + |lambda_delta|) (max loss limit to 1)
                    # rejection_loss = lambda_delta / (flops_delta.abs() + lambda_delta.abs()) / 2
                    # lambda_flops_dict[prefix_name1].data += rejection_loss * self.auto_adjust_lambda_flops_rejection_weight
                    # lambda_flops_dict[prefix_name2].data -= rejection_loss * self.auto_adjust_lambda_flops_rejection_weight
                    rejection_loss = -lambda_delta.abs() / (flops_delta.abs() + lambda_delta.abs()).detach() / 2
                    self.update_cache("loss_dict", 
                                    **{f"rejection_{prefix_name1}_{prefix_name2}" : rejection_loss * self.auto_adjust_lambda_flops_rejection_weight})

            return ret_data_dict[self.DEFAULT_INPUT_NODE_NAME]

        # Complexity
        if self._num_complex_levels > 0 and self.nn_complexity_metric is not None:
            self.setup_complexity_metrics(self.nn_complexity_metric)

        node_dict = self._node_generate_process(**node_dict)
        input_dict = {
            **node_dict,
        }
        prior_dict = dict() if prior is None else {self.DEFAULT_INPUT_NODE_NAME : dict(prior=prior)}

        # inference process
        latent_dict = self._inference_process(input_dict)

        # generative process
        data_dict, prior_dict = self._generative_process(latent_dict, prior_dict=prior_dict, prior_target_dict=prior_target_dict, do_encode=False)

        # for node_name, coder in self.latent_node_entropy_coders.items():
        #     coder(data_dict[node_name], prior=prior_dict[node_name], **kwargs)

        # lambda flops
        lambda_flops = self.lambda_flops
        if self.DEFAULT_LAMBDA_FLOP_NODE_NAME in node_dict:
            lambda_flops = node_dict[self.DEFAULT_LAMBDA_FLOP_NODE_NAME]

        # flops
        if lambda_flops != 0:
            total_flops = self.get_current_flops()
            # NOTE: We only want a non-recursive iteration! 
            # DynamicNNTrainableModule.get_current_flops() could do it recursively!
            # for name, module in self.named_modules():
            # TODO: should we include self.node_generators?
            # for name, module in \
            #     itertools.chain(
            #     self.latent_node_entropy_coders.items(), 
            #     self.latent_inference_modules.items(), 
            #     self.latent_generative_modules.items(), 
            #     self.latent_inference_node_aggregator_modules.items(), 
            #     self.latent_generative_node_aggregator_modules.items()):
            #     if module == self: continue
            #     if isinstance(module, DynamicNNTrainableModule):
            #         module_flops = module.get_current_flops()
            #         if module_flops > 0:
            #             # self.update_cache("moniter_dict", **{"flops_"+name : module_flops})
            #             total_flops += module_flops
            # for name, module in self._modules.items():
            #     if name not in ["latent_node_entropy_coders", "latent_inference_modules", "latent_generative_modules", "latent_inference_node_aggregator_modules", "latent_generative_node_aggregator_modules"]:
            #         continue
            #     for sub_name, sub_module in module._modules.items():
            #         if isinstance(sub_module, DynamicNNTrainableModule):
            #             module_flops = sub_module.get_current_flops()
            #             if module_flops > 0:
            #                 self.update_cache("moniter_dict", **{f"{name}/flops_{sub_name}" : module_flops})
            #                 total_flops += module_flops

            if self.training and self.auto_adjust_lambda_flops:
                flops_perdim = total_flops / data.numel()
                self.update_cache("moniter_dict", lambda_flops=lambda_flops)
                self.update_cache("moniter_dict", flops_perdim=flops_perdim)
                if self.auto_adjust_lambda_flops_method == "rejection":
                    self.update_cache(lambda_flops=lambda_flops)
                    self.update_cache(flops_perdim=flops_perdim)
                if self.auto_adjust_lambda_flops_method == "linear":
                    if self.DEFAULT_COMPLEX_LEVEL_NODE_NAME in node_dict:
                        sclevel = node_dict[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME]
                        # adjust min/max
                        if flops_perdim < self.min_flops_perdim:
                            self.min_flops_perdim.data = flops_perdim.data
                        if flops_perdim > self.max_flops_perdim:
                            self.max_flops_perdim.data = flops_perdim.data
                        self.update_cache("moniter_dict", min_flops_perdim=self.min_flops_perdim)
                        self.update_cache("moniter_dict", max_flops_perdim=self.max_flops_perdim)
                        target_flops_perdim = self.max_flops_perdim - sclevel / (self._num_complex_levels-1) * (self.max_flops_perdim - self.min_flops_perdim)
                        # NOTE: flops_delta_weight > 0 -> flops_perdim<target_flops_perdim -> reduce lambda_flops -> larger flops_perdim
                        flops_delta_weight = (target_flops_perdim - flops_perdim.detach()) / (self.max_flops_perdim - self.min_flops_perdim)
                        linear_adjust_lambda_flop_loss = lambda_flops * flops_delta_weight * self.auto_adjust_lambda_flops_linear_loss_weight
                        self.update_cache("loss_dict", linear_adjust_lambda_flop_loss=linear_adjust_lambda_flop_loss)
            
            if self.training:
                flops_limit = self.flop_per_dim_limit * data.numel()
                # we dont want backprop lambda here!
                if isinstance(lambda_flops, torch.Tensor): lambda_flops = lambda_flops.detach()
                # normalize by batch size
                loss_flops = lambda_flops * torch.clamp_min(total_flops - flops_limit, 0) / data.shape[0]
                self.update_cache("loss_dict", loss_flops=loss_flops)
            self.update_cache("metric_dict", total_flops=total_flops)

        # metrics and logging
        # BPD
        total_prior_entropy = 0
        estimated_bpd = 0
        for name, module in self.latent_node_entropy_coders.items():
            prior_entropy = module.get_raw_cache("metric_dict").get("prior_entropy")
            if prior_entropy is not None:
                total_prior_entropy += prior_entropy
                estimated_bpd += prior_entropy / math.log(2) / (data.numel() / data.size(0))
        self.update_cache("metric_dict", prior_entropy=total_prior_entropy)
        self.update_cache("metric_dict", estimated_bpd=estimated_bpd)
        
        # distortion metric
        if not self.training and self.use_lossy_compression:
            distortion_metric = self.latent_node_entropy_coders[self.DEFAULT_INPUT_NODE_NAME].get_raw_cache("metric_dict").pop(self.lossy_compression_distortion_type, None)
            if distortion_metric is not None:
                self.update_cache("metric_dict", **{self.lossy_compression_distortion_type : distortion_metric})
        
        # normalize loss
        if self.training and self.normalize_loss:
            for name, module in self.latent_node_entropy_coders.items():
                loss_dict = module.get_raw_cache("loss_dict")
                if "loss_rate" in loss_dict:
                    loss_dict["loss_rate"] /= (data.numel() / data.size(0))
                if "loss_distortion" in loss_dict:
                    loss_dict["loss_distortion"] /= (data.numel() / data.size(0))
        
        # cache complexity (should be moved to post_training_process)
        if self.nn_complexity_metric is not None and not self.training and self._num_complex_levels > 0:
            complexity_metric = self.get_nn_complexity()
            if self.nn_complexity_metric == "FLOPs": 
                # use flops perdim
                complexity_metric = complexity_metric / data.numel()
            if self.DEFAULT_COMPLEX_LEVEL_NODE_NAME in node_dict:
                current_complex_level = node_dict[self.DEFAULT_COMPLEX_LEVEL_NODE_NAME]
            else:
                current_complex_level = self._current_complex_level
            self._nn_complexity_metric_cache[current_complex_level] = complexity_metric.item()
            complexity_metric_dict = {self.nn_complexity_metric : complexity_metric}
            self.update_cache("moniter_dict", **complexity_metric_dict)


        # apply logging_prefix in recursive_mode
        if recursive_mode: #len(logging_prefix) > 0:
            if len(logging_prefix) > 0:
                logging_prefix = logging_prefix[:-1] # remove the last _ symbol
            # move all caches in submodules to all_cache_dict and apply logging_prefix
            # all_cache_dict = self.get_all_cache(prefix=logging_prefix)
            # self.reset_all_cache()
            # for cache_name, cache_dict in all_cache_dict.items():
            #     self.update_cache(cache_name, **cache_dict)
            if is_dynamic_node_complete:
                current_all_cache_dict = self.get_all_cache(prefix=logging_prefix)
            else:
                current_all_cache_dict = all_cache_dict
            self.reset_all_cache()
            # for cache_name, cache_dict in current_all_cache_dict.items():
            #     if not cache_name in all_cache_dict: all_cache_dict[cache_name] = dict()
            #     all_cache_dict[cache_name].update(**cache_dict)
            # TODO: check if this returning-cache method works for multiple dynamic nodes
            return (data_dict, prior_dict), current_all_cache_dict

        return data_dict[self.DEFAULT_INPUT_NODE_NAME]

    def encode(self, data, *args, prior=None, **kwargs):
        with torch.no_grad():
            with self.profiler.start_time_profile("encode_prepare"):
                # handle device
                if data.device != self.device:
                    data = data.to(device=self.device)

                node_dict = self._get_default_node_dict(force_add_default_dynamic_nodes=True, **kwargs)

                # TODO: cache node_dict, node_generate_process should only be applied once during coding
                node_dict = self._node_generate_process(**node_dict)
                input_dict = {
                    self.DEFAULT_INPUT_NODE_NAME : data,
                    **node_dict,
                }
                prior_dict = dict() if prior is None else {self.DEFAULT_INPUT_NODE_NAME : dict(prior=prior)}

            # inference process
            with self.profiler.start_time_profile("encode_inference"):
                latent_dict = self._inference_process(input_dict)

            # generative process
            with self.profiler.start_time_profile("encode_generative"):
                data_dict, prior_dict = self._generative_process(latent_dict, prior_dict=prior_dict, do_encode=True)

                # TODO: use stream
                # stream = self.encoder
                # merge byte string according to generative topo order
                latent_nodes = copy.deepcopy(self.latent_node_generative_topo_order)
                if self.use_lossy_compression:
                    latent_nodes.remove(self.DEFAULT_INPUT_NODE_NAME)
                byte_string_merged = merge_bytes([data_dict[name] for name in latent_nodes], num_segments=len(latent_nodes))
            return byte_string_merged

    def decode(self, data, *args, prior=None, **kwargs):
        with torch.no_grad():
            with self.profiler.start_time_profile("decode_prepare"):
                # TODO: use stream
                # stream = self.decoder.set_stream(data)

                node_dict = self._get_default_node_dict(force_add_default_dynamic_nodes=True, **kwargs)

                # TODO: cache node_dict, node_generate_process should only be applied once during coding
                node_dict = self._node_generate_process(**node_dict)
                input_dict = dict()
                # input_dict = {
                #     self.DEFAULT_UNCONDITIONAL_NODE_NAME : None, 
                #     # **kwargs
                # }
                # split byte string
                latent_nodes = copy.deepcopy(self.latent_node_generative_topo_order)
                if self.use_lossy_compression:
                    latent_nodes.remove(self.DEFAULT_INPUT_NODE_NAME)
                byte_string_list = split_merged_bytes(data, num_segments=len(latent_nodes))
                input_dict.update(**{name : byte_string for name, byte_string in zip(latent_nodes, byte_string_list)})
                prior_dict = dict() if prior is None else {self.DEFAULT_INPUT_NODE_NAME : dict(prior=prior)} 
                if self.use_lossy_compression:
                    input_dict[self.DEFAULT_INPUT_NODE_NAME] = b"" # add a dummy bytes for input

            with self.profiler.start_time_profile("decode_generative"):
                # generative process
                data_dict, prior_dict = self._generative_process(input_dict, prior_dict=prior_dict, **node_dict)

            return data_dict[self.DEFAULT_INPUT_NODE_NAME]

    def update_state(self, *args, **kwargs) -> None:
        # TODO: cache generated nodes for faster encode/decode
        # return super().update_state(*args, **kwargs)
        for name, entropy_coder in self.latent_node_entropy_coders.items():
            entropy_coder.update_state(*args, **kwargs)

    def _test_performance_complexity(self, data, *args, **kwargs):
        if self.nn_complexity_metric is not None:
            self.setup_complexity_metrics(self.nn_complexity_metric)
        self.forward(data, *args, **kwargs)
        # TODO: custom metrics
        loss = sum(self.get_cache("loss_dict").values())
        if self.nn_complexity_metric is not None:
            # TODO: support parameter based metric
            flops = self.get_nn_complexity()
        else:
            flops = self.get_current_flops()
        self.reset_all_cache()

        return loss, flops

    def _test_dataset_complexity_performance(self, dataset, *args, performance_method=None, complexity_method=None, **kwargs):
        if performance_method is None or complexity_method is None:
            total_metric = 0
            total_complexity = 0
            total_dims = 0
            self.train() # we need to get losses
            for data in dataset:
                if self.nn_complexity_metric is not None:
                    self.setup_complexity_metrics(self.nn_complexity_metric)
                self.forward(data, *args, **kwargs)
                performance_metric = sum(self.get_cache("loss_dict").values())
                if self.nn_complexity_metric is not None:
                    # TODO: support parameter based metric
                    complexity_metric = self.get_nn_complexity()
                else:
                    complexity_metric = self.get_current_flops()
                total_metric += performance_metric.item()
                total_complexity += complexity_metric.item() if isinstance(complexity_metric, torch.Tensor) else complexity_metric
                total_dims += data.numel()
            total_complexity /= total_dims

            return total_complexity, total_metric
        else:
            performance_metric = 0
            complexity_metric = 0
            total_dims = 0
            for data in dataset:
                total_dims += data.numel()

                # if complexity_method == "FLOPs":
                #     # TODO: support parameter based metric
                #     self.setup_complexity_metrics("FLOPs")

                if performance_method == "loss":
                    # TODO: use only RD-loss!
                    self.train()
                    self.forward(data, *args, **kwargs)
                    for name, loss in self.get_cache("loss_dict").items():
                        if name.endswith("loss_rate") or name.endswith("loss_distortion"):
                            performance_metric += loss.item()
                else:
                    raise NotImplementedError()

                if complexity_method == "FLOPs":
                    # TODO: support parameter based metric
                #     complexity_metric += self.get_nn_complexity()
                # elif complexity_method == "dynamic_FLOPs":
                    complexity_metric += self.get_current_flops()
                elif complexity_method == "compress_time":
                    self.eval()
                    start_time = time.time()
                    self.encode(data, *args, **kwargs)
                    complexity_metric += time.time() - start_time
                elif complexity_method == "decompress_time":
                    self.eval()
                    byte_string = self.encode(data, *args, **kwargs)
                    start_time = time.time()
                    self.decode(byte_string, *args, **kwargs)
                    complexity_metric += time.time() - start_time
                elif complexity_method == "total_time":
                    self.eval()
                    start_time = time.time()
                    byte_string = self.encode(data, *args, **kwargs)
                    self.decode(byte_string, *args, **kwargs)
                    complexity_metric += time.time() - start_time
                else:
                    raise NotImplementedError()
                self.reset_all_cache()

            # TODO: should performance metric divide total_dims?
            complexity_metric /= total_dims
            performance_metric /= total_dims
            if isinstance(complexity_metric, torch.Tensor):
                complexity_metric = complexity_metric.item()
            if isinstance(performance_metric, torch.Tensor):
                performance_metric = performance_metric.item()

            return complexity_metric, performance_metric

    def post_training_process(self, *args, force=False, **kwargs) -> None:
        if self.complexity_level_greedy_search:
            self.update_state()                

            # create complexity levels by greedy search
            # TODO: what if lambda flops is used?
            if force or not self._complexity_param_valid:
                # NOTE: for compress/decompression metrics, this is required
                self.update_state()

                complexity_level_greedy_search_dataset = self.complexity_level_greedy_search_dataset
                if self.complexity_level_greedy_search_dataset_cached:
                    complexity_level_greedy_search_dataset = list(self.complexity_level_greedy_search_dataset)

                def _eval_complexity_idx(complexity_idx : dict, max_iter=None, return_flop_perdim=False):
                    # obtain complexity controlled node data
                    complexity_param_dict = dict()
                    for idx, name in enumerate(self.complexity_level_controller_nodes):
                        module = self.node_generators[name]
                        # assert isinstance(module, IndexParameterGenerator)
                        module_param = module(complexity_idx[name])
                        if self.DEFAULT_EDGE_SPLIT_SYMBOL in name:
                            inode, onode = name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
                            complexity_param_dict[onode] = module_param
                        else:
                            complexity_param_dict[name] = module_param

                    dataset = complexity_level_greedy_search_dataset
                    if max_iter is not None:
                        dataset = itertools.islice(dataset, max_iter)
                    return self._test_dataset_complexity_performance(dataset, 
                        performance_method=self.complexity_level_greedy_search_performance_metric,
                        complexity_method=self.complexity_level_greedy_search_complexity_metric,
                        **complexity_param_dict
                    )

                    # total_metric = 0
                    # total_complexity = 0
                    # total_dims = 0
                    # self.train() # we need to get losses
                    # for idx, data in enumerate(complexity_level_greedy_search_dataset):
                    #     if max_iter is not None and idx >= max_iter: continue
                    #     # if self.nn_complexity_metric is not None:
                    #     #     self.setup_complexity_metrics(self.nn_complexity_metric)
                    #     # self.forward(data, **complexity_param_dict)
                    #     # loss = sum(self.get_cache("loss_dict").values())
                    #     # if self.nn_complexity_metric is not None:
                    #     #     # TODO: support parameter based metric
                    #     #     flops = self.get_nn_complexity()
                    #     # else:
                    #     #     flops = self.get_current_flops()
                    #     performance_metric, complexity_metric = self._test_performance_complexity(
                    #         data, 
                    #         performance_method=self.complexity_level_greedy_search_performance_metric,
                    #         complexity_method=self.complexity_level_greedy_search_complexity_metric,
                    #         **complexity_param_dict
                    #     )
                    #     total_metric += performance_metric.item()
                    #     total_complexity += complexity_metric.item() if isinstance(complexity_metric, torch.Tensor) else complexity_metric
                    #     total_dims += data.numel()

                    # if return_flop_perdim:
                    #     total_complexity /= total_dims

                    # return total_complexity, total_metric
                
                def _recursive_reduce_complexity_idx(complexity_idx : dict, min_complexity_idx : dict, target_flops : float, target_flops_min : float, all_possible_complexity_idx : dict):
                    for reduce_name in complexity_idx:
                        if complexity_idx[reduce_name] == min_complexity_idx[reduce_name]:
                            continue
                        current_complexity_idx = copy.deepcopy(complexity_idx)
                        current_complexity_idx[reduce_name] += 1
                        # NOTE: dict is unhashable
                        if tuple(current_complexity_idx.values()) in all_possible_complexity_idx: continue
                        tmp_flops, _ = _eval_complexity_idx(current_complexity_idx, max_iter=1, return_flop_perdim=True) # just peek flops
                        if tmp_flops > target_flops:
                            all_possible_complexity_idx = _recursive_reduce_complexity_idx(current_complexity_idx, min_complexity_idx, target_flops, target_flops_min, all_possible_complexity_idx)
                        else:
                            if tmp_flops > target_flops_min:
                                all_possible_complexity_idx[tuple(current_complexity_idx.values())] = tmp_flops
                    # return list of idxs to dict
                    # all_possible_complexity_idx = [{k:v for k, v in zip(complexity_idx.keys(), complexity_idx_list)} for complexity_idx_list in all_possible_complexity_idx]
                    return all_possible_complexity_idx
                
                if self.complexity_level_greedy_search_custom_params is not None:
                    complexity_idx_all_levels = self.complexity_level_greedy_search_custom_params

                    # save complexity metric cache
                    # if len(self.complexity_metric_list) > 0:
                    #     for level, complexity_idx in enumerate(complexity_idx_all_levels):
                    #         complexity_idx_stats = _eval_complexity_idx(complexity_idx, return_flop_perdim=True)
                    #         for idx, metric_name in enumerate(self.complexity_metric_list):
                    #             if metric_name == self.complexity_level_greedy_search_complexity_metric:
                    #                 self._complexity_metric_list_cache[level][idx] = complexity_idx_stats[0]
                    #             elif metric_name == self.complexity_level_greedy_search_performance_metric:
                    #                 self._complexity_metric_list_cache[level][idx] = complexity_idx_stats[1]
                    #             # elif metric_name in self.complexity_level_controller_nodes:
                    #             #     self._complexity_metric_list_cache[level][idx] = complexity_idx_stats[metric_name]
                else:
                    if self.nn_complexity_metric is None:
                        self.logger.warning("nn_complexity_metric is not set! We only consider FLOPs metric for dynamic modules here!")
                    
                    max_complexity_idx = dict()
                    min_complexity_idx = dict()
                    all_complexity_idx = dict()
                    for param_name in self.complexity_level_controller_nodes:
                        module = self.node_generators[param_name]
                        assert isinstance(module, IndexParameterGenerator)
                        # if self.DEFAULT_EDGE_SPLIT_SYMBOL in name:
                        #     inode, onode = name.split(self.DEFAULT_EDGE_SPLIT_SYMBOL)
                        #     param_name = onode
                        # else:
                        #     param_name = name
                        max_complexity_idx[param_name] = module.min_sample
                        min_complexity_idx[param_name] = module.max_sample
                        all_complexity_idx[param_name] = list(range(module.min_sample, module.max_sample+1))

                    min_flops, max_loss = _eval_complexity_idx(min_complexity_idx, return_flop_perdim=True)
                    max_flops, min_loss = _eval_complexity_idx(max_complexity_idx, return_flop_perdim=True)
                    assert min_flops < max_flops and min_loss < max_loss, "Complexity should be configured as 0 max!"
                    
                    min_flops_total, max_loss_total = _eval_complexity_idx(min_complexity_idx, return_flop_perdim=True)
                    max_flops_total, min_loss_total = _eval_complexity_idx(max_complexity_idx, return_flop_perdim=True)
                    self.logger.info(f"{max_complexity_idx} : max complexity {max_flops_total} with loss {min_loss_total}!")
                    self.logger.info(f"{min_complexity_idx} : min complexity {min_flops_total} with loss {max_loss_total}!")
                    
                    if self.complexity_level_greedy_search_custom_constraint:
                        target_flops_list = self.complexity_level_greedy_search_custom_constraint
                        # TODO: check target_flops_list is descendent
                        complexity_idx_all_levels = []
                        flops_all_levels = []
                        sclevel_start = 0
                    else:
                        # linear interpolate target flops
                        target_flops_list = max_flops - np.arange(1, self._num_complex_levels - 1) / (self._num_complex_levels - 1) * (max_flops - min_flops)
                        complexity_idx_all_levels = [max_complexity_idx, min_complexity_idx]
                        flops_all_levels = [max_flops_total, min_flops_total]
                        sclevel_start = 1
                    
                    # iterative search
                    if self.complexity_level_greedy_search_iterative:
                        current_complexity_idx = max_complexity_idx
                        # current_flops, current_loss = max_flops_total, max_loss_total
                        self.logger.info(f"Trying to find {len(target_flops_list)} models under flop constraints {target_flops_list}")
                        # Gradually reduce complexity and find the best loss under complexity constraint
                        for sclevel, target_flops in enumerate(target_flops_list, start=sclevel_start):
                            # find best loss in all_possible_complexity_idx
                            best_loss = max_loss_total
                            best_flops = max_flops_total
                            best_complexity_idx = None
                            while best_complexity_idx is None:
                                self.logger.info(f"Start looking for complexity params for target_flops {target_flops}!")
                                all_possible_complexity_idx = _recursive_reduce_complexity_idx(current_complexity_idx, min_complexity_idx, target_flops, min_flops, dict())
                                target_flops_min = target_flops_list[sclevel-sclevel_start+1]
                                all_possible_complexity_idx_filtered = [{k:v for k, v in zip(current_complexity_idx.keys(), complexity_idx_tuple)} for complexity_idx_tuple, flops in all_possible_complexity_idx.items() if flops > target_flops_min]
                                if len(all_possible_complexity_idx_filtered) == 0:
                                    self.logger.warning(f"Cannot find proper complexity params for sclevel {sclevel}, target flops {target_flops}. Trying to reduce more flops!")
                                    all_possible_complexity_idx = [{k:v for k, v in zip(current_complexity_idx.keys(), complexity_idx_tuple)} for complexity_idx_tuple in all_possible_complexity_idx]
                                else:
                                    all_possible_complexity_idx = all_possible_complexity_idx_filtered
                                self.logger.info(f"Found {len(all_possible_complexity_idx)} complexity params for target_flops {target_flops}!")
                                for complexity_idx in tqdm(all_possible_complexity_idx):
                                    tmp_flops, tmp_loss = _eval_complexity_idx(complexity_idx, return_flop_perdim=True)
                                    if tmp_loss < best_loss:
                                        best_loss = tmp_loss
                                        best_flops = tmp_flops
                                        best_complexity_idx = complexity_idx
                                if best_complexity_idx is not None:
                                    self.logger.info(f"Found best complexity params {best_complexity_idx} with flops {best_flops} for target_flops {target_flops}!")
                                    complexity_idx_all_levels.insert(sclevel, best_complexity_idx)
                                    flops_all_levels.insert(sclevel, best_flops)
                                    current_complexity_idx = best_complexity_idx
                                else:
                                    self.logger.warning(f"Cannot find proper complexity params for sclevel {sclevel}, target flops {target_flops}. Restart search with max_complexity_idx!")
                                    current_complexity_idx = max_complexity_idx
                        complexity_idx_all_levels.append(min_complexity_idx)
                        self.logger.info(f"Iterative search success! Found complexity_idx_all_levels: {complexity_idx_all_levels}")
                    else:
                        # greedy search (may be very slow, depend on possible products)
                        self.logger.info("Processing greedy search. This may take a long time. If you want faster search, enable complexity_level_greedy_search_iterative=True!")
                        complexity_idx_stats = dict()
                        complexity_idx_stats[tuple(min_complexity_idx.values())] = (min_flops_total, max_loss_total)
                        complexity_idx_stats[tuple(max_complexity_idx.values())] = (max_flops_total, min_loss_total)
                        for complexity_idx_tuple in tqdm(itertools.product(*all_complexity_idx.values()), total=np.prod([len(l) for l in all_complexity_idx.values()])):
                            if complexity_idx_tuple in complexity_idx_stats: continue
                            complexity_idx = {k:v for k, v in zip(all_complexity_idx.keys(), complexity_idx_tuple)}
                            complexity_idx_stats[complexity_idx_tuple] = _eval_complexity_idx(complexity_idx, return_flop_perdim=True)

                        for sclevel, target_flops in enumerate(target_flops_list, start=sclevel_start):
                            best_loss = max_loss_total
                            best_flops = max_flops_total
                            best_complexity_idx_tuple = None
                            for complexity_idx_tuple, (tmp_flops, tmp_loss) in complexity_idx_stats.items():
                                if tmp_flops <= target_flops and tmp_loss <= best_loss:
                                    best_loss = tmp_loss
                                    best_flops = tmp_flops
                                    best_complexity_idx_tuple = complexity_idx_tuple
                            if best_complexity_idx_tuple is not None:
                                best_complexity_idx = {k:v for k, v in zip(all_complexity_idx.keys(), best_complexity_idx_tuple)}
                                self.logger.info(f"Found best complexity params {best_complexity_idx} with flops {best_flops} and loss {best_loss} for target_flops {target_flops}!")
                                complexity_idx_all_levels.insert(sclevel, best_complexity_idx)
                                flops_all_levels.insert(sclevel, best_flops)
                                current_complexity_idx = best_complexity_idx
                            else:
                                raise ValueError()
                            
                        # save complexity metric cache
                        if len(self.complexity_metric_list) > 0:
                            for level, complexity_idx in enumerate(complexity_idx_all_levels):
                                for idx, metric_name in enumerate(self.complexity_metric_list):
                                    if metric_name == self.complexity_level_greedy_search_complexity_metric:
                                        self._complexity_metric_list_cache[level][idx] = complexity_idx_stats[tuple(complexity_idx.values())][0]
                                    elif metric_name == self.complexity_level_greedy_search_performance_metric:
                                        self._complexity_metric_list_cache[level][idx] = complexity_idx_stats[tuple(complexity_idx.values())][1]
                                    elif metric_name in self.complexity_level_controller_nodes:
                                        self._complexity_metric_list_cache[level][idx] = complexity_idx[metric_name]

                # convert idx to params
                for level, complexity_idx in enumerate(complexity_idx_all_levels):
                    complexity_param = dict()
                    for module_name, value in complexity_idx.items():
                        complexity_param[module_name] = self.node_generators[module_name](value)
                    self._complexity_param_all_levels[level] = ParamDictModuleWrapper(complexity_param)
                    # self._complexity_param_all_levels.append(complexity_param)
        
                # cache flops
                if self.nn_complexity_metric is not None:
                    for sclevel, flops in enumerate(flops_all_levels):
                        self._nn_complexity_metric_cache[sclevel] = flops

                # finally return eval state
                self.eval()
            
            self.logger.info(f"Using {len(self._complexity_param_all_levels)} levels of complexity parameters.")
            # for idx, complexity_param_module in enumerate(self._complexity_param_all_levels):
            #     self.logger.info(f"{idx}, {complexity_param_module()}")
            self._complexity_param_valid.fill_(True)
            # for complexity_param in self._complexity_param_all_levels:
            #     self.logger.info(complexity_param)
        
        # TODO: check _nn_complexity_metric_cache for sclevel node
        # else:
        #     if self.nn_complexity_metric is not None:
        #         self._nn_complexity_metric_cache


    def set_rate_level(self, level, *args, **kwargs) -> bytes:
        if self.num_rate_levels > 0:
            self._current_rate_level = level
            # self._current_rate_level.fill_(level)

    @property
    def num_rate_levels(self):
        return self._num_rate_levels

    def set_task(self, task, *args, **kwargs) -> bool:
        success = True
        if self.num_tasks > 0:
            if task is None:
                self._current_task_idx = -1
            else:
                try:
                    self._current_task_idx = self.task_names.index(task)
                except ValueError:
                    success = False
        else:
            success = False
        return success

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    def set_complex_level(self, level, *args, **kwargs) -> None:
        if self.num_complex_levels > 0:
            self._current_complex_level = level
            # self._current_complex_level.fill_(level)

    def get_current_complex_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        if self.complexity_metric_list is not None:
            results = dict()
            for idx, metric_name in enumerate(self.complexity_metric_list):
                results[metric_name] = self._complexity_metric_list_cache[self._current_complex_level][idx].item()
            return results
        elif self.nn_complexity_metric is not None:
            return {self.nn_complexity_metric : self._nn_complexity_metric_cache[self._current_complex_level].item()}
        # elif self.complexity_level_greedy_search:
        #     return {"dynamic_FLOPs" : self._nn_complexity_metric_cache[self._current_complex_level].item()}
        else:
            # (TODO) no available cache 
            return dict()

    @property
    def num_complex_levels(self) -> int:
        return self._num_complex_levels