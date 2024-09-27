from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder import PyramidVQVAEPriorModel
from configs.import_utils import import_all_config_from_dir

config = ClassBuilder(PyramidVQVAEPriorModel,
    latent_dim=ParamSlot("latent_dim", default=1),
    pyramid_num_embeddings=ParamSlot("pyramid_num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
    # gs_temp=ParamSlot("gs_temp", default=0.5),
    # gs_temp_min=ParamSlot("gs_temp_min", default=0.5),
    # gs_anneal=ParamSlot("gs_anneal", default=False),
    # gs_anneal_rate=ParamSlot("gs_anneal_rate", default=0.00003),
    use_gssoft_vq=ParamSlot("use_gssoft_vq"),
    use_batch_norm=ParamSlot("use_batch_norm"),
).add_param_group_slot("gs_anneal_scheme",
    import_all_config_from_dir("gs_anneal_scheme", caller_file=__file__),
    default="const"
)