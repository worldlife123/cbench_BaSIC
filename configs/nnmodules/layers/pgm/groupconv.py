from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import base as base_module

from cbench.nn.layers.pgm_layers import GroupConv2dPGMModel

config = import_class_builder_from_module(base_module)\
    .update_class(GroupConv2dPGMModel)\
    .update_args(
        in_channels=ParamSlot(), 
        in_groups=ParamSlot(), 
        out_channels=ParamSlot(), 
        out_groups=ParamSlot(),
    )\
    .add_all_kwargs_as_param_slot()
