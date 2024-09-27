from configs.import_utils import import_config_from_module
from . import offset as base_module

config = import_config_from_module(base_module).update_args(
    coding_table=[0, 0],
    coding_extra_symbols=[1 << bits for bits in range(32)],
    max_bits=32,
)