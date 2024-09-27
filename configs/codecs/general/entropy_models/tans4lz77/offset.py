from configs.class_builder import ParamSlot
from configs.import_utils import import_config_from_module
from . import basic as base_module

config = import_config_from_module(base_module).update_args(
    coding_table=[0, 0],
    coding_extra_symbols=[1 << bits for bits in range(16)],
    max_bits=16,
    table_log=ParamSlot(default=8),
)