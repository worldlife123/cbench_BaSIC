from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder import Hierarchical2LayerNNPriorCoder

config = ClassBuilder(Hierarchical2LayerNNPriorCoder,
    prior_coders=ParamSlot(),
    in_channels=ParamSlot(),
    mid_channels=ParamSlot(),
)