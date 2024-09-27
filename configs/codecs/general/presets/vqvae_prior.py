from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from configs.import_utils import import_config_from_module

from .. import base as base_module
from ..prior_models import vqvae as prior_model

config = import_config_from_module(base_module).update_slot_params(
    prior_model=import_config_from_module(prior_model),
)
