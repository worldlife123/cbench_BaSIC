from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import slimmableconv as base_module

from cbench.nn.layers.pgm_layers import HyperpriorSlimmableConv2dPGMModel

config = import_class_builder_from_module(base_module)\
    .update_class(HyperpriorSlimmableConv2dPGMModel)\
    .add_all_kwargs_as_param_slot()
