from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder import VQVAEPriorModel

config = ClassBuilder(VQVAEPriorModel,
    latent_dim=ParamSlot("latent_dim", default=1),
    num_embeddings=ParamSlot("num_embeddings", default=512),
    embedding_dim=ParamSlot("embedding_dim", default=64),
    use_gssoft_vq=ParamSlot("use_gssoft_vq", default=False),
    use_batch_norm=ParamSlot("use_batch_norm", default=True),
)