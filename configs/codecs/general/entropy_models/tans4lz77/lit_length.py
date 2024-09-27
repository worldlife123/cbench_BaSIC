from configs.import_utils import import_config_from_module
from . import basic as base_module

ll_coding_table = [ 0,  1,  2,  3,  4,  5,  6,  7,
                       8,  9, 10, 11, 12, 13, 14, 15,
                      16, 16, 17, 17, 18, 18, 19, 19,
                      20, 20, 20, 20, 21, 21, 21, 21,
                      22, 22, 22, 22, 22, 22, 22, 22,
                      23, 23, 23, 23, 23, 23, 23, 23,
                      24, 24, 24, 24, 24, 24, 24, 24,
                      24, 24, 24, 24, 24, 24, 24, 24]
ll_coding_extra_bits = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3,
    4, 6, 7, 8, 9,10,11,12,
    13,14,15,16
]
ll_coding_extra_symbols = [1 << bits for bits in ll_coding_extra_bits]

config = import_config_from_module(base_module).update_args(
    coding_table=ll_coding_table,
    coding_extra_symbols=ll_coding_extra_symbols,
    max_bits=16,
    table_log=9
)