from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.nn.layers.param_generator import IndexSelectParameterGeneratorWrapper

from . import index as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(IndexSelectParameterGeneratorWrapper)\
    .update_args(
        batched_generator=ParamSlot(),
    )\
    .remove_args("shape")\
    .add_all_kwargs_as_param_slot()