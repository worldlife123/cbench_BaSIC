from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.param_generator import ConvTranspose2dParameterGenerator

config = ClassBuilder(ConvTranspose2dParameterGenerator,
).add_all_kwargs_as_param_slot()