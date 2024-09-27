from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.nn.layers.param_generator import IncreasingVectorGenerator

from . import nn_param as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(IncreasingVectorGenerator)\
    .add_all_kwargs_as_param_slot()