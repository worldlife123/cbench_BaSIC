from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.param_generator import Transformer2dParameterGenerator

config = ClassBuilder(Transformer2dParameterGenerator,
).add_all_kwargs_as_param_slot()