from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.prior_model.prior_coder.compressai_coder import CompressAIScaleHyperpriorCoder

from . import compressai_coder as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(CompressAIScaleHyperpriorCoder)\
    .add_all_kwargs_as_param_slot()
