from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.models.vae import VAEEncoder

config = ClassBuilder(VAEEncoder,
    in_channels=ParamSlot(),
).add_all_kwargs_as_param_slot()