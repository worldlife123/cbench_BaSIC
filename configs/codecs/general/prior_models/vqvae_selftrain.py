from configs.class_builder import ClassBuilder, ParamSlot
from configs.utils.pretrained_model_builder import PretrainedModelBuilder
from cbench.modules.prior_model.autoencoder import VQVAESelfTrainedPriorModel

config = ClassBuilder(VQVAESelfTrainedPriorModel,
    trainer=ParamSlot("trainer"),
    latent_dim=ParamSlot("latent_dim"),
    num_embeddings=ParamSlot("num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
    input_shift=ParamSlot("input_shift"),
    lr=ParamSlot("lr"),
)