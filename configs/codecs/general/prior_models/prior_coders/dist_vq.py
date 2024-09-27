from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import vq as base_module

from cbench.modules.prior_model.prior_coder import DistributionVQPriorCoder

config = import_class_builder_from_module(base_module)\
    .update_class(DistributionVQPriorCoder)\
    .update_args(
        in_channels=ParamSlot(),
        latent_dim=ParamSlot(),
        num_embeddings=ParamSlot(),
        embedding_dim=ParamSlot(),
)