from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.param_generator import GroupedParameterGeneratorWrapper

config = ClassBuilder(GroupedParameterGeneratorWrapper,
    batched_generator=ParamSlot(),
).add_all_kwargs_as_param_slot()