from configs.import_utils import import_class_builder_from_module
from cbench.modules.prior_model.prior_coder.pgm_coder import NNTrainablePGMPriorCoder

from . import ans_coder as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(NNTrainablePGMPriorCoder)\
    .add_all_kwargs_as_param_slot()
