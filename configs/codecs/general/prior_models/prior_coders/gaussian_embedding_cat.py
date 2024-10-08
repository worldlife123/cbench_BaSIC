from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder import GaussianEmbeddingCategoricalPriorCoder

config = ClassBuilder(GaussianEmbeddingCategoricalPriorCoder)\
    .add_all_kwargs_as_param_slot()