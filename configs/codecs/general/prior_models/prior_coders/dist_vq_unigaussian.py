from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import dist_vq as base_module

from cbench.modules.prior_model.prior_coder import UnivarGaussianDistributionVQPriorCoder

config = import_class_builder_from_module(base_module)\
    .update_class(UnivarGaussianDistributionVQPriorCoder)\
    .update_args(
        use_pyramid_init=ParamSlot(),
)