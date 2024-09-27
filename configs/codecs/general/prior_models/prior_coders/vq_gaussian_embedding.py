from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import vq as base_module

from cbench.modules.prior_model.prior_coder import VQGaussianEmbeddingPriorCoder

# config = ClassBuilder(GaussianVQPriorCoder,
#     in_channels=ParamSlot(),
#     latent_channels=ParamSlot(),
#     num_embeddings=ParamSlot(),
# )
config = import_class_builder_from_module(base_module)\
    .update_class(VQGaussianEmbeddingPriorCoder)\
    .add_all_kwargs_as_param_slot()
#     .update_args(
# )