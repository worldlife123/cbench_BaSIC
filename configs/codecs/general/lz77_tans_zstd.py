from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from configs.import_utils import import_config_from_module

from . import lz77_fse as base_module
from .entropy_models import grouped_for_lz77_zstd as entropy_coder_module

from cbench.modules.entropy_coder import GroupedEntropyCoder, FSEEntropyCoder, TANSEntropyCoder

config = import_config_from_module(base_module).update_slot_params(
    entropy_coder=import_config_from_module(entropy_coder_module),
)