from cbench.nn.trainer import CUDADeviceParallelNNTrainer

from configs.import_utils import import_class_builder_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

from . import simple_trainer as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(CUDADeviceParallelNNTrainer)\
    .add_all_kwargs_as_param_slot()