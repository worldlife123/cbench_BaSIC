from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.codecs.base import GroupedVariableRateCodec


config = ClassBuilder(
    GroupedVariableRateCodec,
    codecs=ParamSlot("codecs"),
).add_all_kwargs_as_param_slot()