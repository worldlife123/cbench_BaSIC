import abc
import pickle
from typing import Any, Dict, Iterator, List, Tuple, Optional, Union

from cbench.modules.base import BaseModule, TrainableModuleInterface, BaseTrainableModule
from cbench.nn.base import NNTrainableModule, PLNNTrainableModule, SelfTrainableModule, TorchCheckpointLoader
from cbench.nn.trainer import BasicNNTrainer
from cbench.utils.logging_utils import MetricLogger

class CodecInterface(abc.ABC):
    @abc.abstractmethod
    def compress(self, data, *args, **kwargs) -> bytes:
        pass

    @abc.abstractmethod
    def decompress(self, data: bytes, *args, **kwargs):
        pass

    # optional method to cache some state for faster coding
    def update_state(self, *args, **kwargs) -> None:
        pass

class PickleSerilizeFunctions(abc.ABC):
    def serialize(self, data, *args, **kwargs) -> bytes:
        return pickle.dumps(data)

    def deserialize(self, data: bytes, *args, **kwargs):
        return pickle.loads(data)


class VariableRateCodecInterface(abc.ABC):
    @abc.abstractmethod
    def set_rate_level(self, level, *args, **kwargs) -> None:
        pass

    @property
    def num_rate_levels(self) -> int:
        return 1


class VariableComplexityCodecInterface(abc.ABC):
    @abc.abstractmethod
    def set_complex_level(self, level, *args, **kwargs) -> None:
        pass

    def get_current_complex_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        return dict()

    @property
    def num_complex_levels(self) -> int:
        return 1


class VariableTaskCodecInterface(abc.ABC):
    @abc.abstractmethod
    def set_task(self, task, *args, **kwargs) -> bool:
        pass

    @property
    def num_tasks(self) -> int:
        return 1


class BaseCodec(BaseModule, CodecInterface, PickleSerilizeFunctions):
    pass
    # moved to BaseModule
    # def __init__(self, *args, **kwargs):
    #     self._profiler = MetricLogger()

    # @property
    # def profiler(self):
    #     return self._profiler

    # @profiler.setter
    # def profiler(self, profiler):
    #     self._profiler = profiler


class PickleCodec(BaseCodec):
    def compress(self, data, *args, **kwargs) -> bytes:
        return self.serialize(data)

    def decompress(self, data: bytes, *args, **kwargs):
        return self.deserialize(data)


class BaseTrainableCodec(BaseCodec, BaseTrainableModule):
    pass


class NNTrainableCodec(BaseCodec, SelfTrainableModule): # inherit from (PL)NNTrainableModule to enable nn modules
    def __init__(self, *args, trainer : BasicNNTrainer = None, **kwargs):
        super().__init__(*args, **kwargs)
        # inherit from SelfTrainableModule to allow codec-specific trainer
        SelfTrainableModule.__init__(self, *args, trainer=trainer, **kwargs)

    def forward(self, *args, **kwargs):
        # default forward pass to the compress process
        self.compress(*args, **kwargs)

    def forward_estimate_bitlen(self, *args, **kwargs) -> Tuple[Any, float]:
        return self.forward(*args, **kwargs), 0.0

    def get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
        parameters = dict()
        for name, module in self.get_named_submodules():
            if isinstance(module, TrainableModuleInterface):
                parameters[name] = (module.get_parameters(*args, **kwargs))
        return parameters

    # def iter_trainable_parameters(self, *args, **kwargs) -> Iterator:
    #     for name, module in self.get_named_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             for param in module.iter_trainable_parameters():
    #                 yield param

    def load_parameters(self, parameters: Dict[str, Any], *args, **kwargs) -> None:
        for name, module in self.get_named_submodules():
            if isinstance(module, TrainableModuleInterface):
                module.load_parameters(parameters[name], *args, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        for module in self.get_submodules():
            if isinstance(module, CodecInterface):
                module.update_state(*args, **kwargs)

    # def train_full(self, dataloader, *args, **kwargs) -> None:
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.train_full(dataloader, *args, **kwargs)

    # def train_iter(self, data, *args, **kwargs) -> None:
    #     for module in self.get_submodules():
    #         if isinstance(module, TrainableModuleInterface):
    #             module.train_iter(data, *args, **kwargs)
    

class GroupedVariableRateCodec(NNTrainableCodec, VariableRateCodecInterface, VariableComplexityCodecInterface, VariableTaskCodecInterface):
    def __init__(self, codecs : List[NNTrainableCodec], *args,
                 codec_vr_level_config : Dict[int, Tuple[int, int]] = dict(),
                 codec_sc_level_config : Dict[int, Tuple[int, int]] = dict(),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.codecs = codecs
        for i, codec in enumerate(codecs):
            self.add_module(f"codec_{i}", codec)
        self.codec_vr_level_config = codec_vr_level_config
        self.codec_sc_level_config = codec_sc_level_config

        self.active_codec_idx = 0
        self.set_rate_level(0)

    def set_rate_level(self, level, *args, **kwargs) -> None:
        sub_level = 0
        if level in self.codec_vr_level_config:
            level, sub_level = self.codec_vr_level_config[level]
        self.active_codec_idx = level
        # self.active_codec = self.codecs[level]
        if isinstance(self.active_codec, VariableRateCodecInterface):
            self.active_codec.set_rate_level(sub_level, *args, **kwargs)

    @property
    def num_rate_levels(self):
        return len(self.codecs) if len(self.codec_vr_level_config) == 0 else len(self.codec_vr_level_config)

    def set_complex_level(self, level, *args, active_only=False, **kwargs) -> None:
        if active_only:
            if isinstance(self.active_codec, VariableComplexityCodecInterface):
                self.active_codec.set_complex_level(level, *args, **kwargs)
        else:
            # NOTE: may fail if num_complex_levels is different across codec
            for codec in self.codecs:
                if isinstance(codec, VariableComplexityCodecInterface):
                    codec.set_complex_level(level, *args, **kwargs)

    def get_current_complex_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        if isinstance(self.active_codec, VariableComplexityCodecInterface):
            return self.active_codec.get_current_complex_metrics(*args, **kwargs)
        else:
            return dict()

    @property
    def num_complex_levels(self) -> int:
        return max([(codec.num_complex_levels if isinstance(codec, VariableComplexityCodecInterface) else 0) for codec in self.codecs])

    def set_task(self, task, *args, active_only=False, **kwargs) -> bool:
        if active_only:
            if isinstance(self.active_codec, VariableTaskCodecInterface):
                return self.active_codec.set_task(task, *args, **kwargs)
        else:
            # NOTE: may fail if num_complex_levels is different across codec
            success = True
            for codec in self.codecs:
                if isinstance(codec, VariableTaskCodecInterface):
                    success = (success and codec.set_task(task, *args, **kwargs))
            return success
    
    @property
    def num_tasks(self) -> int:
        return max([(codec.num_tasks if isinstance(codec, VariableTaskCodecInterface) else 0) for codec in self.codecs])

    @property
    def active_codec(self):
        return self.codecs[self.active_codec_idx]
    
    def __len__(self):
        return len(self.codecs)

    def __getitem__(self, index):
        return self.codecs[index]

    def compress(self, data, *args, **kwargs) -> bytes:
        return self.active_codec.compress(data, *args, **kwargs)

    def decompress(self, data: bytes, *args, **kwargs):
        return self.active_codec.decompress(data, *args, **kwargs)
    
    def forward(self, *args, **kwargs):
        # all codecs should be forwarded during training!
        # if self.training:
        for codec in self.codecs:
            if codec != self.active_codec:
                codec(*args, **kwargs)
        # return the active codec
        result = self.active_codec(*args, **kwargs)
        return result

    def forward_estimate_bitlen(self, *args, **kwargs) -> Tuple[Any, float]:
        return self.active_codec.forward_estimate_bitlen(*args, **kwargs)

    def update_state(self, *args, **kwargs) -> None:
        for codec in self.codecs:
            codec.update_state(*args, **kwargs)
    
    def post_training_process(self, *args, **kwargs) -> None:
        for codec in self.codecs:
            codec.post_training_process(*args, **kwargs)

    def load_checkpoint(self, checkpoint_loader : Optional[Union[str, TorchCheckpointLoader]] = None):
        # first try load all codec checkpoints
        for codec in self.codecs:
            codec.load_checkpoint()
        return super().load_checkpoint(checkpoint_loader)
