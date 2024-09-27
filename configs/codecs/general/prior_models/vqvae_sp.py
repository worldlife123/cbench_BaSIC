from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder import SimplePyramidVQVAEPriorModel

config = ClassBuilder(SimplePyramidVQVAEPriorModel,
    # embedding_dim=ParamSlot("embedding_dim", default=64),
    single_decoder = ParamSlot(default=False),
    use_batch_norm=ParamSlot("use_batch_norm", default=True),
)