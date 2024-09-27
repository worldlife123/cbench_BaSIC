from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.models.vae import VAEDecoder

config = ClassBuilder(VAEDecoder,
    out_channels=ParamSlot(),
).add_all_kwargs_as_param_slot()