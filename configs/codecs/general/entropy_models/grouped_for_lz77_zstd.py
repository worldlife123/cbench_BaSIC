from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder import GroupedEntropyCoder, FSEEntropyCoder
from cbench.modules.entropy_coder.fse import TANSEntropyCoder
import numpy as np

ml_coding_table = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
                    38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 ]
ml_coding_extra_bits = [
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 2, 2, 3, 3,
     4, 4, 5, 7, 8, 9,10,11,
    12,13,14,15,16
]
ml_coding_extra_symbols = [1 << bits for bits in ml_coding_extra_bits]

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

config = ClassBuilder(
    GroupedEntropyCoder,
    ClassBuilderList(
        ClassBuilder(TANSEntropyCoder, 
            coding_table=[i for i in range(256)],
            coding_extra_symbols=[0] * 256,
            max_bits=8,
            table_log=11
        ),
        ClassBuilder(TANSEntropyCoder, 
            coding_table=[0],
            coding_extra_symbols=[1 << bits for bits in range(32)],
            # table_distribution=np.ones(3200),
            max_bits=31,
            max_symbol=32,
            table_log=8
        ),
        ClassBuilder(TANSEntropyCoder, 
            coding_table=ll_coding_table,
            coding_extra_symbols=ll_coding_extra_symbols,
            max_bits=16,
            max_symbol=len(ll_coding_extra_symbols),
            table_log=9
        ),
        ClassBuilder(TANSEntropyCoder, 
            coding_table=ml_coding_table,
            coding_extra_symbols=ml_coding_extra_symbols,
            max_bits=16,
            max_symbol=len(ml_coding_extra_symbols),
            table_log=9
        ),

    )
)