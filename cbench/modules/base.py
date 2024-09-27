import abc
from typing import Any, Dict, Iterator, List, Mapping, Sequence

from cbench.utils.logging_utils import MetricLogger

class TrainableModuleInterface(abc.ABC):
    @abc.abstractmethod
    def train_full(self, dataloader, *args, **kwargs) -> None:
        pass
    
    @abc.abstractmethod
    def train_iter(self, data, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def get_parameters(self, *args, **kwargs) -> Any:
        pass

    # an interface for torch trainable nn, equivalent to nn.Module.parameters 
    # removed as not all class need this
    # @abc.abstractmethod
    # def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
    #     pass

    @abc.abstractmethod
    def load_parameters(self, parameters: Any, *args, **kwargs) -> None:
        pass

    # optional method to update state after training (moved to EntropyCoder)
    # def update_state(self, *args, **kwargs) -> None:
    #     pass

    def post_training_process(self, *args, **kwargs) -> None:
        pass

class BaseModule(object):
    def __init__(self, *args, **kwargs):
        self._submodules = dict()
        self._profiler = MetricLogger()

    @property
    def profiler(self):
        return self._profiler

    @profiler.setter
    def profiler(self, profiler):
        self._profiler = profiler

    def register_submodule(self, name, module):
        if name in self._submodules:
            raise ValueError("Name {} already registered!".format(name))
        self._submodules[name] = module

    def get_registered_submodules(self):
        return self._submodules.values()

    def get_registered_named_submodules(self):
        return self._submodules.items()

    def _yield_module(self, module, recursive=False):
        # TODO: support nn.Module! 
        # Currently BaseModule inside ModuleList/ModuleDict cannot be yielded because they are not Sequence/Mapping
        # NOTE: str could cause infinite recursion as str are composed of Sequences 
        if isinstance(module, str):
            pass
        else:
            if isinstance(module, BaseModule):
                yield module
                if recursive:
                    for sub in module.get_submodules(recursive=True):
                        yield sub
            elif isinstance(module, Sequence):
                for i, vv in enumerate(module): 
                    yield from self._yield_module(vv, recursive=recursive)
            elif isinstance(module, Mapping):
                for kk, vv in module.items(): 
                    yield from self._yield_module(vv, recursive=recursive)

    def get_submodules(self, recursive=False):
        return self._yield_module(self.__dict__, recursive=recursive)

        # for k, v in self.__dict__.items():
        #     if _yield_module(v, recursive=recursive):
        #         pass
        #     # if isinstance(v, BaseModule):
        #     #     yield v
        #     #     if recursive:
        #     #         for sub in v.get_submodules(recursive=True):
        #     #             yield sub
        #     elif isinstance(v, Sequence):
        #         for vv in v: 
        #             _yield_module(vv, recursive=recursive)
        #             # if isinstance(vv, BaseModule):
        #             #     yield vv
        #             #     if recursive:
        #             #         for sub in vv.get_submodules(recursive=True):
        #             #             yield sub
        #     elif isinstance(v, Mapping):
        #         for kk, vv in v.items(): 
        #             _yield_module(vv, recursive=recursive)
        #             # if isinstance(vv, BaseModule):
        #             #     yield vv
        #             #     if recursive:
        #             #         for sub in vv.get_submodules(recursive=True):
                    #             yield sub

    def _yield_named_module(self, module_name, module, recursive=False):
        # NOTE: str could cause infinite recursion as str are composed of Sequences 
        if isinstance(module, str):
            pass
        else:
            if isinstance(module, BaseModule):
                yield module_name, module
                if recursive:
                    for name, sub in module.get_named_submodules(recursive=True, name_prefix=module_name):
                        yield name, sub
            elif isinstance(module, Sequence):
                for i, vv in enumerate(module): 
                    sub_name = '.'.join((module_name, str(i)))
                    yield from self._yield_named_module(sub_name, vv, recursive=recursive)
            elif isinstance(module, Mapping):
                for kk, vv in module.items(): 
                    sub_name = '.'.join((module_name, str(kk)))
                    yield from self._yield_named_module(sub_name, vv, recursive=recursive)


    def get_named_submodules(self, recursive=False, name_prefix=""):
        return self._yield_named_module(name_prefix, self.__dict__, recursive=recursive)
        # for k, v in self.__dict__.items():
            # if self._yield_module('.'.join((name_prefix, k)), v, recursive=recursive):
            #     pass
            # # if isinstance(v, BaseModule):
            # #     module_name = '.'.join((name_prefix, k))
            # #     yield module_name, v
            # #     if recursive:
            # #         for name, sub in v.get_named_submodules(recursive=True, name_prefix=module_name):
            # #             yield name, sub
            # elif isinstance(v, Sequence):
            #     for i, vv in enumerate(v): 
            #         module_name = '.'.join((name_prefix, k, str(i)))
            #         _yield_module(module_name, vv, recursive=recursive)
            #         # if isinstance(vv, BaseModule):
            #         #     module_name = '.'.join((name_prefix, k, str(i)))
            #         #     yield module_name, vv
            #         #     if recursive:
            #         #         for name, sub in vv.get_named_submodules(recursive=True, name_prefix=module_name):
            #         #             yield name, sub
            # elif isinstance(v, Mapping):
            #     for kk, vv in v.items(): 
            #         module_name = '.'.join((name_prefix, k, kk))
            #         _yield_module(module_name, vv, recursive=recursive)
            #         # if isinstance(vv, BaseModule):
            #         #     module_name = '.'.join((name_prefix, k, kk))
            #         #     yield module_name, vv
            #         #     if recursive:
            #         #         for name, sub in vv.get_named_submodules(recursive=True, name_prefix=module_name):
            #         #             yield name, sub

    def collect_profiler_results(self, recursive=False, clear=False):
        results = dict()
        results.update(**self.profiler.get_global_average())
        if clear:
            self.profiler.clear()
        for name, module in self.get_named_submodules(recursive=recursive):
            for key, value in module.profiler.get_global_average().items():
                results[":".join([name, key])] = value
            if clear:
                module.profiler.clear()
        return results


class ModuleList(BaseModule):
    def __init__(self, *modules, **kwargs):
        super().__init__(**kwargs)
        self.module_list = modules

    def __getitem__(self, index):
        return self.module_list[index]
    
    def get_submodules(self):
        for i, module in enumerate(self.module_list):
            if isinstance(module, BaseModule):
                yield module

    def get_named_submodules(self):
        for i, module in enumerate(self.module_list):
            if isinstance(module, BaseModule):
                yield str(i), module


class ModuleDict(BaseModule):
    def __init__(self, *args, **modules):
        super().__init__(*args)
        self.module_dict = modules
    
    def __getitem__(self, index):
        return self.module_dict[index]

    def get_submodules(self):
        for key, module in self.module_dict.items():
            if isinstance(module, BaseModule):
                yield module

    def get_named_submodules(self):
        for key, module in self.module_dict.items():
            if isinstance(module, BaseModule):
                yield key, module


class BaseTrainableModule(BaseModule, TrainableModuleInterface):
    def get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
        parameters = dict()
        for name, module in self.get_named_submodules():
            if isinstance(module, TrainableModuleInterface):
                parameters[name] = (module.get_parameters(*args, **kwargs))
        return parameters

    def load_parameters(self, parameters: Dict[str, Any], *args, **kwargs) -> None:
        for name, module in self.get_named_submodules():
            if isinstance(module, TrainableModuleInterface):
                module.load_parameters(parameters[name], *args, **kwargs)
    
    # def update_state(self, *args, **kwargs) -> None:
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.update_state(*args, **kwargs)

    def train_full(self, dataloader, *args, **kwargs) -> None:
        raise NotImplementedError()
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.train_full(dataloader, *args, **kwargs)

    def train_iter(self, data, *args, **kwargs) -> None:
        raise NotImplementedError()
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.train_iter(data, *args, **kwargs)