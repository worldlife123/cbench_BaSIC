from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import dist_c2vq as base_module

from cbench.modules.prior_model.prior_coder import GaussianC2VQDistributionPriorCoder

config = import_class_builder_from_module(base_module)\
    .update_class(GaussianC2VQDistributionPriorCoder)\
    .add_all_kwargs_as_param_slot()