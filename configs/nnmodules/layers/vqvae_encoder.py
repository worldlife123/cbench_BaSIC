from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.models.vqvae_model_v2 import Encoder

config = ClassBuilder(Encoder,
    channels=ParamSlot(),
).add_all_kwargs_as_param_slot()