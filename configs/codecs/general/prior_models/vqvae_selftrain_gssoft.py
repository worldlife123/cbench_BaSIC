from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_all_config_from_dir, import_class_builder_from_module
from cbench.modules.prior_model.autoencoder import VQVAEGSSOFTSelfTrainedPriorModel

from . import vqvae_selftrain as base_module

config = import_class_builder_from_module(base_module).update_class(
    VQVAEGSSOFTSelfTrainedPriorModel,
).update_args(
    training_soft_samples=ParamSlot(),
).add_param_group_slot("gs_anneal_scheme",
    import_all_config_from_dir("gs_anneal_scheme", caller_file=__file__),
    default="const"
)