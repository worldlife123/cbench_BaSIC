from typing import Any, Callable, Dict, Iterator, List, Tuple, Optional, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data.dataloader import DataLoader
import torch.cuda

import pytorch_lightning as pl

# from ptflops.pytorch_engine import add_flops_counting_methods

import logging
import os
import time
import functools
from collections import OrderedDict

from cbench.utils.engine import BaseEngine
from cbench.modules.base import TrainableModuleInterface, BaseModule
from cbench.utils.logger import setup_logger
from cbench.utils.logging_utils import SmoothedValue, TimeProfiler, MetricLogger

class BasicNNTrainer(BaseEngine):
    """
    Similar to pl.Trainer. Implements BaseEngine to support logging and checkpointing. 
    """    
    def __init__(self, model : nn.Module = None,
        train_loader : DataLoader = None,
        val_loader : DataLoader = None,
        test_loader : DataLoader = None,
        on_initialize_start_hook : Callable = None,
        on_initialize_end_hook : Callable = None,
        on_train_start_hook : Callable = None,
        on_train_end_hook : Callable = None,
        checkpoint_dir="experiments", 
        float32_matmul_precision=None,
        **kwargs
    ):
        # TODO: checkpoint_dir is confusing to output_dir, consider removing this eventually!
        super().__init__(
            output_dir=checkpoint_dir,
        )
        self.set_model(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # by default testing data is the same as validation data
        self.val_loader = val_loader if val_loader is not None else test_loader
        self.test_loader = test_loader if test_loader is not None else val_loader
        # self.checkpoint_dir = checkpoint_dir
        self.on_initialize_start_hook = on_initialize_start_hook
        self.on_initialize_end_hook = on_initialize_end_hook
        self.on_train_start_hook = on_train_start_hook
        self.on_train_end_hook = on_train_end_hook

        self.extra_options = kwargs
        
        # Tensor Core config
        try:
            if float32_matmul_precision is not None:
                torch.set_float32_matmul_precision(float32_matmul_precision)
        except:
            self.logger.warning("set_float32_matmul_precision not available in this environment!")
        
        self.has_initialized = False

    def set_model(self, model : nn.Module):
        self.model = model
    
    # for compability
    @property
    def checkpoint_dir(self):
        return self.output_dir

    @property
    def _module_dict_with_state(self):
        return dict(
            model=self.model,
        )

    def state_dict(self):
        return {k:v.state_dict() for k,v in self._module_dict_with_state.items()}
    
    def load_state_dict(self, checkpoint, strict=True):
        for k,v in self._module_dict_with_state.items():
            if isinstance(v, nn.Module):
                v.load_state_dict(checkpoint[k], strict=strict)
            # else:
            #     v.load_state_dict(checkpoint[k])

    def load_model(self, checkpoint, strict=True):
        self.model.load_state_dict(checkpoint, strict=strict)
    
    """
    Called before any process start. Could include training scheduling, loading checkpoints, processing model structure, etc.
    """
    def _initialize(self, *args, **kwargs):
        raise NotImplementedError()

    """
    Called before any process end. Usually only for exiting DDP training.
    """
    def _deinitialize(self, *args, **kwargs):
        pass
        # raise NotImplementedError()

    """
    Main training process.
    """
    def _train(self, *args, **kwargs):
        raise NotImplementedError()

    """
    Main validation process. Should return a metric for comparisons.
    """
    def _validate(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    """
    Main testing process.
    """
    def _test(self, *args, **kwargs):
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        if self.has_initialized:
            self.logger.warning("Reinitializing Trainer!")
        if self.on_initialize_start_hook is not None:
            self.on_initialize_start_hook(self)
        
        # load pretrained checkpoint
        if isinstance(self.model, NNTrainableModule):
            # TODO: add pretrained kwargs in trainer
            self.model.load_checkpoint()

        self._initialize(*args, **kwargs, **self.extra_options)
        self.has_initialized = True
        self.logger.info("Trainer Initialized!")
        if self.on_initialize_end_hook is not None:
            self.on_initialize_end_hook(self)

    def deinitialize(self, *args, **kwargs):
        if self.has_initialized:
            self._deinitialize(*args, **kwargs)
            self.has_initialized = False
            self.logger.info("Trainer Deinitialized!")

    def do_train(self, *args, **kwargs):
        # NOTE: should check loader in subprocess, i.e. _train
        # assert(not self.train_loader is None)
        self.initialize()
        self.logger.info("Beginning training...")
        if self.on_train_start_hook is not None:
            self.on_train_start_hook(self)
        self._train(**self.extra_options)
        if self.on_train_end_hook is not None:
            self.on_train_end_hook(self)

    def do_validate(self, *args, **kwargs):
        # assert(not self.val_loader is None)
        self.initialize()
        self.logger.info("Beginning validation...")
        return self._validate(**self.extra_options)    
        
    def do_test(self, *args, **kwargs):
        # assert(not self.test_loader is None)
        self.initialize()
        self.logger.info("Beginning testing...")
        return self._test(**self.extra_options)


class TorchCheckpointLoader(BaseEngine):
    def __init__(self, checkpoint_file, *args, strict=True, key=None, prefix=None, filter_keys=None, **kwargs):
        self.checkpoint_file = checkpoint_file
        self.strict = strict
        self.key = key
        self.prefix = prefix
        self.filter_keys = filter_keys
        super().__init__(*args, **kwargs)

    def load(self, model : nn.Module):
        self.logger.info(f"Loading checkpoint from {self.checkpoint_file}")
        state_dict = torch.load(self.checkpoint_file)
        if self.key is not None:
            state_dict = state_dict[self.key]
        if self.prefix is not None:
            # NOTE: for python >= 3.9., use key.removeprefix()
            state_dict = {(key[len(self.prefix):] if key.startswith(self.prefix) else key) : value \
                          for key, value in state_dict.items()}
        if self.filter_keys is not None:
            for name in self.filter_keys:
                state_dict.pop(name)
        return model.load_state_dict(state_dict, strict=self.strict)


class TorchCUDATimeProfiler(TimeProfiler):
    def __init__(self, meter: SmoothedValue, include_cpu_time=False) -> None:
        self.meter = meter
        self.include_cpu_time = include_cpu_time
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    @staticmethod
    def start_profile(meter):
        return TorchCUDATimeProfiler(meter)

    def __enter__(self):
        if self.include_cpu_time:
            self.start_time = time.time()
        else:
            self.start_event.record()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        if self.include_cpu_time:
            end_time = time.time()
            self.meter.update((end_time - self.start_time) * 1000)
        else:
            self.end_event.record()
            self.meter.update(self.end_event.elapsed_time(self.start_event))


class NNCacheImpl(object):
    """
    This class implements caching functions for Modules. Should implement named_children.
    Caches are python dicts.
    """    
    def __init__(self):
        # allow direct access with name
        # self.loss_dict = dict()
        # self.metric_dict = dict()
        # self.hist_dict = dict()
        for cache_name in self.cache_names:
            # setattr(self, cache_name, dict())
            self.__dict__[cache_name] = dict()

        self.optim_state = 0

        self.logger = setup_logger(self.__class__.__name__)

    def set_optim_state(self, state : Any = 0):
        """ Set optim state for this module and all its submodules

        Args:
            state (str, optional): state name. Defaults to None.
        """        
        self.optim_state = state
        for name, module in self.named_children():
            if isinstance(module, NNTrainableModule):
                module.set_optim_state(state)

    # TODO: define value constraint on caches
    # e.g. for loss and metric, it must be a scalar tensor
    @property
    def cache_names(self) -> List[str]:
        return [
            "common", 
            "loss_dict", 
            "metric_dict", 
            "moniter_dict", 
            "hist_dict", 
            "image_dict", 
            "text_dict", 
            "figure_dict",
        ]

    def named_children(self) -> Iterator[Tuple[str, Any]]:
        """
        should be implemented in Module

        Raises:
            NotImplementedError: [description]

        Yields:
            Iterator[Tuple[str, Any]]: [description]
        """        
        raise NotImplementedError()

    def get_cache(self, cache_name="common", prefix=None, recursive=True) -> Dict[str, Any]:
        """
        Recursively get a cache .

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".
            prefix ([type], optional): Prefix of returned cache keys. If None, cache_name is used. Defaults to None.
            recursive (bool, optional): Should do recursive get. Defaults to True.

        Returns:
            Dict[str, Any]: Returned caches. All keys has "prefix" variable as prefix.
        """        
        result_dict = dict()
        cache_dict = getattr(self, cache_name)
        if prefix is None:
            prefix = cache_name
        assert isinstance(cache_dict, dict)
        if len(prefix) > 0:
            for k, v in cache_dict.items():
                kn = '/'.join((prefix, k))
                result_dict[kn] = v
        else:
            result_dict.update(cache_dict)
        # recursive get
        if recursive:
            for name, module in self.named_children():
                prefix_new = '/'.join((prefix, name)) if len(prefix) > 0 else name
                # TODO: better to implement named_children ModuleList/nn.ModuleList and ModuleDict/nn.ModuleDict
                if isinstance(module, nn.ModuleList):
                    for idx, sub_module in enumerate(module):
                        if isinstance(sub_module, NNCacheImpl):
                            prefix_sub = '/'.join((prefix_new, f"{idx}"))
                            result_dict.update(sub_module.get_cache(cache_name, prefix=prefix_sub, recursive=True))
                if isinstance(module, nn.ModuleDict):
                    for sub_name, sub_module in module.items():
                        if isinstance(sub_module, NNCacheImpl):
                            prefix_sub = '/'.join((prefix_new, sub_name))
                            result_dict.update(sub_module.get_cache(cache_name, prefix=prefix_sub, recursive=True))
                if isinstance(module, NNCacheImpl):
                    result_dict.update(module.get_cache(cache_name, prefix=prefix_new, recursive=True))
        return result_dict

    def get_raw_cache(self, cache_name="common") -> Dict[str, Any]:
        """Get raw cache data .

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".

        Returns:
            Dict[str, Any]: Raw cache dict.
        """        
        return getattr(self, cache_name)

    def get_all_cache(self, prefix=None, recursive=True) -> Dict[str, Dict[str, Any]]:
        """Get all cache values .

        Args:
            prefix ([type], optional): Prefix of returned cache keys. If None, cache_name is used. Defaults to None.
            recursive (bool, optional): Should do recursive get. Defaults to True.

        Returns:
            Dict[str, Dict[str, Any]]: All cache dicts with {cache_name : cache_dict} structure.
        """        
        result_dict = dict()
        for cache_name in self.cache_names:
            result_dict[cache_name] = self.get_cache(cache_name, prefix=prefix, recursive=recursive)
        return result_dict

    def update_cache(self, cache_name="common", **kwargs):
        """Update a cache with kwargs

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".
        """        
        cache_dict = getattr(self, cache_name)
        assert(isinstance(cache_dict, dict))
        # for kw, value in kwargs.items():
        #     # TODO: auto avoid memory issue
        #     if isinstance(value, torch.Tensor):
        #         if value.numel() == 1:
        #             value = value.item()
        #         else:
        #             value = value.cpu().detach_()
        #     cache_dict[kw] = value
        cache_dict.update(**kwargs)

    def reset_cache(self, cache_name="common", recursive=True) -> None:
        """Reset a cache to empty dict .

        Args:
            cache_name (str, optional): Should in self.cache_names. Defaults to "common".
            recursive (bool, optional): Should do recursive get. Defaults to True.
        """        
        setattr(self, cache_name, dict())
        # recursive reset
        if recursive:
            for name, module in self.named_children():
                if isinstance(module, nn.ModuleList):
                    for idx, sub_module in enumerate(module):
                        if isinstance(sub_module, NNCacheImpl):
                            sub_module.reset_cache(cache_name)
                if isinstance(module, nn.ModuleDict):
                    for sub_name, sub_module in module.items():
                        if isinstance(sub_module, NNCacheImpl):
                            sub_module.reset_cache(cache_name)
                if isinstance(module, NNCacheImpl):
                    module.reset_cache(cache_name)

    def reset_all_cache(self, recursive=True) -> None:
        """Reset all cache .

        Args:
            recursive (bool, optional): Should do recursive get. Defaults to True.
        """        
        for cache_name in self.cache_names:
            self.reset_cache(cache_name, recursive=recursive)

    # # below funcs are for backward compability
    # @property
    # def loss_dict(self):
    #     return self.loss_dict

    # @property
    # def metric_dict(self):
    #     return self.metric_dict

    def get_loss_dict(self, prefix : str = "losses") -> Dict[str, torch.Tensor]:
        return self.get_cache("loss_dict", prefix=prefix)
        # result_dict = dict()          
        # if len(prefix) > 0:
        #     for k, v in self.loss_dict.items():
        #         kn = '/'.join((prefix, k))
        #         result_dict[kn] = v
        # else:
        #     result_dict.update(self.loss_dict)
        # # recursive get
        # for name, module in self.named_children():
        #     if isinstance(module, NNCacheImpl):
        #         prefix_new = '/'.join((prefix, name))
        #         result_dict.update(module.get_loss_dict(prefix=prefix_new))
        # return result_dict

    def reset_loss_dict(self) -> None:
        self.reset_cache("loss_dict")
        # self.loss_dict = dict()
        # # recursive reset
        # for name, module in self.named_children():
        #     if isinstance(module, NNCacheImpl):
        #         module.reset_loss_dict()

    def get_metric_dict(self, prefix : str = "metrics") -> Dict[str, torch.Tensor]:
        return self.get_cache("metric_dict", prefix=prefix)
        # result_dict = dict()          
        # if len(prefix) > 0:
        #     for k, v in self.metric_dict.items():
        #         kn = '/'.join((prefix, k))
        #         result_dict[kn] = v
        # else:
        #     result_dict.update(self.metric_dict)
        # # recursive get
        # for name, module in self.named_children():
        #     if isinstance(module, NNCacheImpl):
        #         prefix_new = '/'.join((prefix, name))
        #         result_dict.update(module.get_metric_dict(prefix=prefix_new))
        # return result_dict

    def reset_metric_dict(self) -> None:
        self.reset_cache("metric_dict")
        # self.metric_dict = dict()
        # # recursive reset
        # for name, module in self.named_children():
        #     if isinstance(module, NNCacheImpl):
        #         module.reset_metric_dict()


class NNTrainableModule(nn.Module, BaseModule, NNCacheImpl, TrainableModuleInterface):
    """NNTrainableModule is an extended nn.Module that:
        1. Adds caching functions (NNCacheImpl) 
        2. Supports training interface.
        3. Adds a _device_indicator buffer which supports self.device similar to torch.Tensor.
    """    
    def __init__(self, checkpoint_loader : Optional[Union[str, TorchCheckpointLoader]] = None, **kwargs):
        self.checkpoint_loader = checkpoint_loader
        super().__init__()
        BaseModule.__init__(self)
        NNCacheImpl.__init__(self)
        self.register_buffer("_device_indicator", torch.zeros(1), persistent=False)
        self.available_complexity_metrics = ["FLOPs"]
        # cache for trainer
        # self.loss_dict = dict()
        # self.metric_dict = dict()
        # self._profiler = MetricLogger(profiler_class=functools.partial(TorchCUDATimeProfiler, include_cpu_time=profiler_include_cpu_time))

        # TODO: check attr type
        self.parameter_attributes = {k:v for k,v in kwargs.items() if k in self._parameter_attr_names}

    @property
    def _parameter_attr_names(self):
        return ["requires_grad", "lr_modifier", "weight_decay_modifier", "aux_id", "gradient_clipping_group", "optimizer_idx"]
    
    def register_parameter(self, name: str, param: nn.Parameter) -> None:
        # auto assign module-level parameter attributes before register
        for attr_name in self._parameter_attr_names:
            if attr_name in self.parameter_attributes and not hasattr(param, attr_name):
                setattr(param, attr_name, self.parameter_attributes[attr_name])
        return super().register_parameter(name, param)

    @property
    def device(self):
        return self._device_indicator.device

    def start_time_profile(self, name, profiler_include_cpu_time=False):
        profiler_class = functools.partial(TorchCUDATimeProfiler, include_cpu_time=profiler_include_cpu_time) if self.device == "cuda" else None
        return self.profiler.start_time_profile(name, profiler_class=profiler_class)

    def set_custom_state(self, state : str = None):
        """ Set custom state for this module and all its submodules

        Args:
            state (str, optional): state name. Defaults to None.
        """        
        for name, module in self.named_children():
            if isinstance(module, NNTrainableModule):
                module.set_custom_state(state)

    def load_checkpoint(self, checkpoint_loader : Optional[Union[str, TorchCheckpointLoader]] = None):
        if checkpoint_loader is None:
            checkpoint_loader = self.checkpoint_loader
        
        if checkpoint_loader is None:
            pass
        elif isinstance(checkpoint_loader, str):
            state_dict = torch.load(checkpoint_loader)
            self.load_state_dict(state_dict, strict=False)
        elif isinstance(checkpoint_loader, TorchCheckpointLoader):
            checkpoint_loader.load(self)
        else:
            raise ValueError("Unsupported checkpoint_loader!")

    # def get_loss_dict(self, prefix : str = "losses") -> Dict[str, torch.Tensor]:
    #     result_dict = dict()          
    #     if len(prefix) > 0:
    #         for k, v in self.loss_dict.items():
    #             kn = '/'.join((prefix, k))
    #             result_dict[kn] = v
    #     else:
    #         result_dict.update(self.loss_dict)
    #     # recursive get
    #     for name, module in self.named_children():
    #         if isinstance(module, NNTrainableModule):
    #             prefix_new = '/'.join((prefix, name))
    #             result_dict.update(module.get_loss_dict(prefix=prefix_new))
    #     return result_dict

    # def reset_loss_dict(self) -> None:
    #     self.loss_dict = dict()
    #     # recursive reset
    #     for name, module in self.named_children():
    #         if isinstance(module, NNTrainableModule):
    #             module.reset_loss_dict()

    # def get_metric_dict(self, prefix : str = "metrics") -> Dict[str, torch.Tensor]:
    #     result_dict = dict()          
    #     if len(prefix) > 0:
    #         for k, v in self.metric_dict.items():
    #             kn = '/'.join((prefix, k))
    #             result_dict[kn] = v
    #     else:
    #         result_dict.update(self.metric_dict)
    #     # recursive get
    #     for name, module in self.named_children():
    #         if isinstance(module, NNTrainableModule):
    #             prefix_new = '/'.join((prefix, name))
    #             result_dict.update(module.get_metric_dict(prefix=prefix_new))
    #     return result_dict

    # def reset_metric_dict(self) -> None:
    #     self.metric_dict = dict()
    #     # recursive reset
    #     for name, module in self.named_children():
    #         if isinstance(module, NNTrainableModule):
    #             module.reset_metric_dict()

    def _yield_module(self, module, recursive=False):
        if isinstance(module, NNTrainableModule):
            yield module
            if recursive:
                for name, sub in module.named_modules():
                    if sub == module: continue
                    yield from self._yield_module(sub, recursive=recursive)
        elif isinstance(module, nn.ModuleList):
            for i, vv in enumerate(module): 
                yield from self._yield_module(vv, recursive=recursive)
        elif isinstance(module, nn.ModuleDict):
            for kk, vv in module.items(): 
                yield from self._yield_module(vv, recursive=recursive)
        else:
            yield from super()._yield_module(module, recursive)

    def get_submodules(self, recursive=False):
        return self._yield_module(list(self.children()), recursive=recursive)

    def _yield_named_module(self, module_name, module, recursive=False):
        if isinstance(module, NNTrainableModule):
            yield module_name, module
            if recursive:
                for name, sub in module.named_modules(prefix=module_name):
                    if sub == module: continue
                    yield from self._yield_named_module(name, sub, recursive=recursive)
        elif isinstance(module, nn.ModuleList):
            for i, vv in enumerate(module): 
                sub_name = '.'.join((module_name, str(i)))
                yield from self._yield_named_module(sub_name, vv, recursive=recursive)
        elif isinstance(module, nn.ModuleDict):
            for kk, vv in module.items(): 
                sub_name = '.'.join((module_name, kk))
                yield from self._yield_named_module(sub_name, vv, recursive=recursive)
        else:
            yield from super()._yield_named_module(module_name, module, recursive)

    def get_named_submodules(self, recursive=False, name_prefix=""):
        return self._yield_named_module(name_prefix, {name:module for name, module in self.named_children()}, recursive=recursive)

    def get_parameters(self, *args, **kwargs) -> dict:
        return self.state_dict()

    def load_parameters(self, parameters: dict, *args, **kwargs) -> None:
        self.load_state_dict(parameters)

    def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
        return self.parameters()

    def train_full(self, dataloader, *args, **kwargs) -> None:
        for data in dataloader:
            self.train_iter(data, *args, **kwargs)

    def train_iter(self, data, *args, **kwargs) -> None:
        self.forward(data, *args, **kwargs)
        # TODO: parameter update should be handled with an inner or extra optimizer!

    # def update_state(self, *args, **kwargs) -> None:
    #     super().update_state(*args, **kwargs)
        # basically we only set model to eval mode here (this should be handled before testing)
        # self.eval()

    def forward(self, *args, **kwargs):
        self.reset_loss_dict()
        self.reset_metric_dict()
        # TODO: setup hooks for some known modules
        # if "reset_flops_count" in self:
        #     self.reset_flops_count()

    def setup_complexity_metrics(self, metrics : Union[str, List[str]]):
        self.available_complexity_metrics = metrics if isinstance(metrics, list) else [metrics]
        # TODO: ptflops does not fit well with our framework, as they count recursively as well...
        # if "FLOPs" in self.available_complexity_metrics or "Params" in self.available_complexity_metrics:
        #     if hasattr(self, "reset_flops_count"):
        #         self.reset_flops_count()
        #     else:
        #         add_flops_counting_methods(self)
        #     self.start_flops_count(ost=None, verbose=True, ignore_list=[])
        for module in self.modules():
            if module != self and isinstance(module, NNTrainableModule):
                module.setup_complexity_metrics(metrics)
    
    def get_nn_complexity(self, input : Optional[Any] = None, metric : Optional[str] = None):
        # TODO: process from input?
        if metric is None:
            metric = self.available_complexity_metrics[0]
        total_complexity = 0
        # TODO: ptflops does not fit well with our framework, as they count recursively as well...
        # if metric == "FLOPs" or metric == "Params":
        #     if hasattr(self, "stop_flops_count"):
        #         flops_count, params_count = self.compute_average_flops_cost()
        #         self.stop_flops_count()
        #         if metric == "FLOPs":
        #             total_complexity += flops_count
        #         elif metric == "Params":
        #             total_complexity += params_count
        for module in self.modules():
            if module != self and isinstance(module, NNTrainableModule):
                total_complexity += module.get_nn_complexity(metric=metric)
        # TODO: enable this warning for release version
        # if total_complexity == 0:
        #     self.logger.warning(f"No complexity counted for module {type(self)}")
        #     self.logger.warning(f"To disable this warning if you are sure this class have no complexity, implement get_nn_complexity function in {type(self)} and simply return 0.")
        return total_complexity


class DynamicNNTrainableModule(NNTrainableModule):
    def __init__(self):
        super().__init__()
        self._dynamic_parameters = OrderedDict()
        self._dynamic_parameter_hooks = OrderedDict()

    # TODO: redef this func to get_current_complexity(self, input=None, metrics=None)
    def get_current_flops(self, input=None):
        total_flops = 0 # self.get_current_flops() # avoid infinite recursive call
        for module in self.modules():
            if module != self and isinstance(module, DynamicNNTrainableModule):
                total_flops += module.get_current_flops()
        return total_flops

    def register_dynamic_parameter(self, name: str, value : torch.Tensor, hook : Optional[ Callable[[NNTrainableModule], None]] = None) -> None:
        r"""Adds a dynamic parameter to the module.

        The dynamic parameter cannot be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            value (Any): parameter to be added to the module.
            hook (Callable): function to call when the value changed
        """
        if '_dynamic_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign dynamic parameters before Module.__init__() call")

        elif not isinstance(name, str): # torch._six.string_classes): for torch2.0 compability
            raise TypeError("dynamic parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("dynamic parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("dynamic parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._dynamic_parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        elif value is not None and not isinstance(value, torch.Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(torch.typename(value), name))

        # if value is None:
        #     self._dynamic_parameters[name] = None
        # elif not isinstance(value, Parameter):
        #     raise TypeError("cannot assign '{}' object to parameter '{}' "
        #                     "(torch.nn.Parameter or None required)"
        #                     .format(torch.typename(param), name))
        # elif value.grad_fn:
        #     raise ValueError(
        #         "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
        #         "parameters must be created explicitly. To express '{0}' "
        #         "as a function of another Tensor, compute the value in "
        #         "the forward() method.".format(name))
        # else:
        # TODO: do we need to check the value type?
        self._dynamic_parameters[name] = value
        if hook is not None:
            self._dynamic_parameter_hooks[name] = hook

    def get_dynamic_parameter_value(self, name):
        return self._dynamic_parameters[name]

    def set_dynamic_parameter_value(self, name, value):
        if isinstance(value, (int, float)):
            self._dynamic_parameters[name].fill_(value)
        else:
            self._dynamic_parameters[name].data = value
        if name in self._dynamic_parameter_hooks:
            self._dynamic_parameter_hooks[name](value)

    def _save_to_dynamic_state_dict(self, destination, prefix):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._dynamic_parameters.items():
            if param is not None:
                destination[prefix + name] = param

    def dynamic_state_dict(self, destination=None, prefix='') -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
        self._save_to_dynamic_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if isinstance(module, DynamicNNTrainableModule):
                module.dynamic_state_dict(destination, prefix + name + '.')
        return destination

    def _load_from_dynamic_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = self._dynamic_parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                self._dynamic_parameters[name] = input_param
                if name in self._dynamic_parameter_hooks:
                    self._dynamic_parameter_hooks[name](input_param)
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_dynamic_state_dict(self, state_dict,
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_dynamic_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        load = None  # break load->load reference cycle

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_dynamic_parameters' in self.__dict__:
            _dynamic_parameters = self.__dict__['_dynamic_parameters']
            if name in _dynamic_parameters:
                return _dynamic_parameters[name]
        return super().__getattr__(name)
    
    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        _dynamic_parameters = self.__dict__.get('_dynamic_parameters')
        if _dynamic_parameters is not None and name in _dynamic_parameters:
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError("cannot assign '{}' as dynamic_parameter '{}' "
                                "(torch.Tensor or None expected)"
                                .format(torch.typename(value), name))
            _dynamic_parameters[name] = value
        else:
            return super().__setattr__(name, value)

    def _apply(self, fn):
        for key, param in self._dynamic_parameters.items():
            if param is not None:
                self._dynamic_parameters[key] = fn(param)
        return super()._apply(fn)

# a self-trainable module interface (TODO: better be renamed to SelfTrainableModuleImpl?)
class SelfTrainableInterface(NNCacheImpl):
    def __init__(self, trainer : BasicNNTrainer = None, output_dir=None, **kwargs):
        super().__init__()
        NNCacheImpl.__init__(self)
        self.trainer = trainer
        if trainer is not None:
            self.set_trainer(trainer)
            trainer.setup_engine(output_dir=output_dir)
            if self.is_trainer_setup():
                self.do_train()

    def set_trainer(self, trainer : BasicNNTrainer, **kwargs):
        self.trainer = trainer
        self.trainer.set_model(self)
        self.trainer_config = kwargs

    def is_trainer_valid(self):
        return self.trainer is not None
    
    def is_trainer_setup(self):
        return self.trainer is not None and self.trainer.output_dir is not None
    
    def setup_trainer_engine(self, output_dir=None, logger=None, **kwargs):
        self.trainer.setup_engine(output_dir=output_dir, logger=logger)

    def do_train(self):
        if self.trainer is not None:
            self.trainer.initialize(**self.trainer_config)
            self.trainer.do_train()


class SelfTrainableModule(NNTrainableModule, SelfTrainableInterface):
    def __init__(self, trainer : BasicNNTrainer = None, output_dir=None, **kwargs):
        super().__init__(**kwargs)
        SelfTrainableInterface.__init__(self, trainer, output_dir=output_dir)

    def train_full(self, dataloader, *args, **kwargs) -> None:
        # for data in dataloader:
        #     self.train_iter(data, *args, **kwargs)
        # if self.trainer is not None:
        #     self.trainer.initialize(*args, **kwargs, **self.trainer_config)
        self.logger.warning("SelfTrainableModule is self trained! No need to call train() function!")

    def train_iter(self, data, *args, **kwargs) -> None:
        # self.forward(data, *args, **kwargs)
        self.logger.warning("SelfTrainableModule is self trained! No need to call train() function!")

    # usually training is not required for self trained modules!
    def forward(self, *args, **kwargs):
        pass


# a self-trainable module powered by pl.LightningModule
class PLNNTrainableModule(pl.LightningModule, SelfTrainableInterface, TrainableModuleInterface):
    def __init__(self, trainer : BasicNNTrainer = None, output_dir=None, **kwargs):
        super().__init__()
        SelfTrainableInterface.__init__(self, trainer, output_dir=output_dir)

    def get_parameters(self, *args, **kwargs) -> dict:
        return self.state_dict()

    def load_parameters(self, parameters: dict, *args, **kwargs) -> None:
        self.load_state_dict(parameters)

    def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
        return self.parameters()

    def train_full(self, dataloader, *args, **kwargs) -> None:
        # for data in dataloader:
        #     self.train_iter(data, *args, **kwargs)
        # if self.trainer is not None:
        #     self.trainer.initialize(*args, **kwargs, **self.trainer_config)
        self.logger.warning("PLNNTrainableModule is self trained! No need to call train() function!")

    def train_iter(self, data, *args, **kwargs) -> None:
        # self.forward(data, *args, **kwargs)
        self.logger.warning("PLNNTrainableModule is self trained! No need to call train() function!")

    # usually training is not required for self trained modules!
    def forward(self, *args, **kwargs):
        pass


class GroupedNNTrainableModule(NNTrainableModule):
    def __init__(self, modules : List[NNTrainableModule], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grouped_modules = nn.ModuleList(modules)

    def forward(self, *args, idx=None, **kwargs):
        if idx is None:
            for module in self.grouped_modules:
                module(*args, **kwargs)
        else:
            return self.grouped_modules[idx](*args, **kwargs)
