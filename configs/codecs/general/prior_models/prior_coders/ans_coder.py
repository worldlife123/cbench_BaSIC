from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder.torch_ans import TorchANSPriorCoder

config = ClassBuilder(TorchANSPriorCoder)\
    .add_all_kwargs_as_param_slot()