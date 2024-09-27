from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.models.google import MeanScaleHyperpriorHyperSynthesisModel

config = ClassBuilder(MeanScaleHyperpriorHyperSynthesisModel,
    N=ParamSlot(),
    M=ParamSlot(),
).add_all_kwargs_as_param_slot()