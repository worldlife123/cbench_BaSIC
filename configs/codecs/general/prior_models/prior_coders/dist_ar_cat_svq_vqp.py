from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module

from . import dist_ar_cat_svq as base_module

from cbench.modules.prior_model.prior_coder import VQPriorStochasticVQAutoregressivePriorDistributionPriorCoder

config = import_class_builder_from_module(base_module)\
    .update_class(VQPriorStochasticVQAutoregressivePriorDistributionPriorCoder)\
    .add_all_kwargs_as_param_slot()
