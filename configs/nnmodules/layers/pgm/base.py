from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.pgm_layers import BasePGMLayer

config = ClassBuilder(BasePGMLayer,
).add_all_kwargs_as_param_slot()