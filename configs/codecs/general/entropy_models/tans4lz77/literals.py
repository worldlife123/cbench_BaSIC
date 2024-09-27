from configs.import_utils import import_config_from_module
from . import basic as base_module

config = import_config_from_module(base_module).update_args(
    coding_table=[i for i in range(256)],
    coding_extra_symbols=[1] * 256,
    max_bits=8,
    table_log=11
)