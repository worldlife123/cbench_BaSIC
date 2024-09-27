from configs.class_builder import ClassBuilder, ParamSlot
from configs.utils.pretrained_model_builder import PretrainedModelBuilder
from cbench.modules.prior_model.autoencoder import VQVAEPreTrainedPriorModel

config = ClassBuilder(VQVAEPreTrainedPriorModel,
    model=ParamSlot("model"),
)