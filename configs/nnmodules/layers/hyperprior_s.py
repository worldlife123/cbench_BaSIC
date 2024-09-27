from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.models.google import HyperpriorSynthesisModel

config = ClassBuilder(HyperpriorSynthesisModel,
    N=ParamSlot(),
    M=ParamSlot(),
).add_all_kwargs_as_param_slot()