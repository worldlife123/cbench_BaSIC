from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.param_generator import NNModuleParameterWrapper

config = ClassBuilder(NNModuleParameterWrapper,
    module=ParamSlot(),
).add_all_kwargs_as_param_slot()