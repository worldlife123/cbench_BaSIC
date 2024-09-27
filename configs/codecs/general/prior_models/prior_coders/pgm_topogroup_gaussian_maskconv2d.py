from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.prior_model.prior_coder.pgm_coder import GaussianChannelGroupMaskConv2DTopoGroupPGMPriorCoder
from cbench.modules.prior_model.prior_coder.pgm_coder import GaussianPGMPriorCoderImpl

from . import pgm_topogroup as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(GaussianChannelGroupMaskConv2DTopoGroupPGMPriorCoder)\
    .add_all_kwargs_as_param_slot(GaussianPGMPriorCoderImpl)\
    .add_all_kwargs_as_param_slot()
