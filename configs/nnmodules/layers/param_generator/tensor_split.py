from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.param_generator import TensorSplitGenerator

config = ClassBuilder(TensorSplitGenerator,
    split_size_or_sections=ParamSlot(),
).add_all_kwargs_as_param_slot()