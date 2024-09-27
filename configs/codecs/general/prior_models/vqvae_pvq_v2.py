from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder import PyramidVQVAEPriorModelV2
from configs.import_utils import import_all_config_from_dir

config = ClassBuilder(PyramidVQVAEPriorModelV2,
    latent_dim=ParamSlot("latent_dim"),
    pyramid_num_embeddings=ParamSlot("pyramid_num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
    use_gssoft_vq=ParamSlot("use_gssoft_vq"),
).add_param_group_slot("gs_anneal_scheme",
    import_all_config_from_dir("gs_anneal_scheme", caller_file=__file__),
    default="const"
)