from configs.class_builder import ClassBuilder, ParamSlot
from cbench.codecs.pycodecs import PILWebPCodec

config = ClassBuilder(
    PILWebPCodec,
).add_all_kwargs_as_param_slot()