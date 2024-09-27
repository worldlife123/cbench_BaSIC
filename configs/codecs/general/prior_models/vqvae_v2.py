from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder import VQVAEPriorModelV2
from configs.import_utils import import_all_config_from_dir

config = ClassBuilder(VQVAEPriorModelV2,
    latent_dim=ParamSlot("latent_dim"),
    num_embeddings=ParamSlot("num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
    use_gssoft_vq=ParamSlot("use_gssoft_vq"),
    kl_cost=ParamSlot("kl_cost"),
    use_st_gumbel=ParamSlot(),
    commitment_cost_gs=ParamSlot(),
    commitment_over_exp=ParamSlot(),
    test_sampling=ParamSlot(),
    dist_type=ParamSlot(
        choices=[
            "RelaxedOneHotCategorical", 
            "AsymptoticRelaxedOneHotCategorical", 
            "DoubleRelaxedOneHotCategorical"
        ]
    ),
    relax_temp=ParamSlot(),
    relax_temp_min=ParamSlot(),
    relax_temp_anneal=ParamSlot(),
    relax_temp_anneal_rate=ParamSlot(),
    gs_temp=ParamSlot(),
    gs_temp_min=ParamSlot(),
    gs_anneal=ParamSlot(),
    gs_anneal_rate=ParamSlot(),
).add_param_group_slot("gs_anneal_scheme",
    import_all_config_from_dir("gs_anneal_scheme", caller_file=__file__),
    default="const"
).add_param_group_slot("relax_temp_anneal_scheme",
    import_all_config_from_dir("relax_temp_anneal_scheme", caller_file=__file__),
    default="const"
)