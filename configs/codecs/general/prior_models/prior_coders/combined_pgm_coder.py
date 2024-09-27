from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.prior_coder.pgm_coder import CombinedNNTrainablePGMPriorCoder

config = ClassBuilder(CombinedNNTrainablePGMPriorCoder)\
    .update_args(
        coders=ParamSlot()
    )\
    .add_all_kwargs_as_param_slot()