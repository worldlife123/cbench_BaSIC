from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.layers.masked_conv import TopoGroupDynamicMaskConv2dContextModel

config = ClassBuilder(TopoGroupDynamicMaskConv2dContextModel,
).add_all_kwargs_as_param_slot()