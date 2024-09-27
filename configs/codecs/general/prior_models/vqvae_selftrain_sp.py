from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder import SimplePyramidVQVAESelfTrainedPriorModel

config = ClassBuilder(SimplePyramidVQVAESelfTrainedPriorModel,
    trainer=ParamSlot("trainer"),
    latent_dim=ParamSlot("latent_dim"),
    pyramid_num_embeddings=ParamSlot("pyramid_num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
)