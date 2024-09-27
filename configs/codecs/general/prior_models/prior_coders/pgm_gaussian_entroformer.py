from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.prior_model.prior_coder.pgm_coder import GaussianEntroFormerPriorCoder

from . import pgm_coder as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(GaussianEntroFormerPriorCoder)\
    .add_all_kwargs_as_param_slot()
