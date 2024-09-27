from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.nn.layers.param_generator import IndexSelectParameterGenerator

from . import index as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(IndexSelectParameterGenerator)\
    .update_args(
        values=ParamSlot(),
    )\
    .remove_args("max", "shape")\
    .add_all_kwargs_as_param_slot()