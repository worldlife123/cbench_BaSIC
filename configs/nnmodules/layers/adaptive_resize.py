from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.basic import AdaptiveResize2DLayer

config = ClassBuilder(AdaptiveResize2DLayer,
    in_channels=ParamSlot(),
).add_all_kwargs_as_param_slot()