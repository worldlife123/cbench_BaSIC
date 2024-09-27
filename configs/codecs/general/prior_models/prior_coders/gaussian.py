from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder import GaussianPriorCoder

config = ClassBuilder(GaussianPriorCoder,
    in_channels=ParamSlot(),
    latent_channels=ParamSlot(),
)