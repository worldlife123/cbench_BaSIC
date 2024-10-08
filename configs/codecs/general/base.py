from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.codecs.general_codec import GeneralCodec


config = ClassBuilder(
    GeneralCodec,
    preprocessor=ParamSlot("preprocessor"),
    prior_model=ParamSlot("prior_model"),
    context_model=ParamSlot("context_model"),
    entropy_coder=ParamSlot("entropy_coder"),
    prior_first=ParamSlot("prior_first", default=False),
    trainer=ParamSlot("trainer", default=None),
)