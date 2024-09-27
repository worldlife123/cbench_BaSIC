from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder import CategoricalGaussianPriorCoder

config = ClassBuilder(CategoricalGaussianPriorCoder)\
    .add_all_kwargs_as_param_slot()