from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import vq as base_module

from cbench.modules.prior_model.prior_coder import GaussianVQPriorCoder

# config = ClassBuilder(GaussianVQPriorCoder,
#     in_channels=ParamSlot(),
#     latent_channels=ParamSlot(),
#     num_embeddings=ParamSlot(),
# )
config = import_class_builder_from_module(base_module)\
    .update_class(GaussianVQPriorCoder)\
    .update_args(
        in_channels=ParamSlot(),
        latent_dim=ParamSlot(),
        num_embeddings=ParamSlot(),
        embedding_dim=ParamSlot(),
        distance_method=ParamSlot(),
        distance_loss_method=ParamSlot(),
        use_pyramid_init=ParamSlot(),
        pyramid_init_invert_logprob=ParamSlot(),
        gaussian_kl_cost=ParamSlot(),
        gaussian_kl_from_encoder=ParamSlot(),
        rsample_params=ParamSlot(),
        rsample_params_method=ParamSlot(),
)