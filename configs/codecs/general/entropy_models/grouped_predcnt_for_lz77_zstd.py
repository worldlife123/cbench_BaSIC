from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from configs.import_utils import import_config_from_module, import_all_config_from_dir
from cbench.modules.entropy_coder import GroupedEntropyCoder, FSEEntropyCoder
from cbench.modules.entropy_coder.fse import TANSEntropyCoder, TrainablePredCntTANSEntropyCoder
import numpy as np

from .tans4lz77 import literals as default_coder_literals
from .tans4lz77 import offset_ldm as default_coder_offset
from .tans4lz77 import lit_length as default_coder_lit_length
from .tans4lz77 import match_length as default_coder_match_length

# ml_coding_table = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
#                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
#                     32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
#                     38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
#                     40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
#                     41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
#                     42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
#                     42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 ]
# ml_coding_extra_bits = [
#      0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0,
#      1, 1, 1, 1, 2, 2, 3, 3,
#      4, 4, 5, 7, 8, 9,10,11,
#     12,13,14,15,16
# ]
# ml_coding_extra_symbols = [1 << bits for bits in ml_coding_extra_bits]

# ll_coding_table = [ 0,  1,  2,  3,  4,  5,  6,  7,
#                        8,  9, 10, 11, 12, 13, 14, 15,
#                       16, 16, 17, 17, 18, 18, 19, 19,
#                       20, 20, 20, 20, 21, 21, 21, 21,
#                       22, 22, 22, 22, 22, 22, 22, 22,
#                       23, 23, 23, 23, 23, 23, 23, 23,
#                       24, 24, 24, 24, 24, 24, 24, 24,
#                       24, 24, 24, 24, 24, 24, 24, 24]
# ll_coding_extra_bits = [
#     0, 0, 0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0, 0, 0,
#     1, 1, 1, 1, 2, 2, 3, 3,
#     4, 6, 7, 8, 9,10,11,12,
#     13,14,15,16
# ]
# ll_coding_extra_symbols = [1 << bits for bits in ll_coding_extra_bits]

config = ClassBuilder(
    GroupedEntropyCoder,
    ClassBuilderList(
        # ClassBuilder(FSEEntropyCoder,
        #     coder=FSEEntropyCoder.CODER_HUF,
        #     table_log=11,
        # ),
        # ClassBuilder(TrainablePredCntTANSEntropyCoder, 
        #     coding_table=[i for i in range(256)],
        #     coding_extra_symbols=[0] * 256,
        #     # table_distribution=2**(16 - np.ceil(np.log2(np.arange(1, 2048)))),
        #     # table_distribution=np.ones(3600),
        #     max_bits=8,
        #     # update_coding_table=True,
        #     num_predcnts=ParamSlot('num_predcnts', default=1, choices=list(range(1, 255))),
        #     # max_symbol=32,
        #     table_log=11
        # ),
        # ClassBuilder(TrainablePredCntTANSEntropyCoder, 
        #     coding_table=[0],
        #     coding_extra_symbols=[1 << bits for bits in range(32)],
        #     # table_distribution=2**(16 - np.ceil(np.log2(np.arange(1, 2048)))),
        #     # table_distribution=np.ones(3600),
        #     max_bits=16,
        #     # update_coding_table=True,
        #     num_predcnts=ParamSlot('num_predcnts', default=1, choices=list(range(1, 255))),
        #     # max_symbol=32,
        #     table_log=8
        # ),
        # ClassBuilder(TrainablePredCntTANSEntropyCoder, 
        #     coding_table=ll_coding_table,
        #     coding_extra_symbols=ll_coding_extra_symbols,
        #     max_bits=16,
        #     # update_coding_table=True,
        #     num_predcnts=ParamSlot('num_predcnts', default=1, choices=list(range(1, 255))),
        #    # max_symbol=len(ll_coding_extra_symbols),
        #     table_log=9
        # ),
        # ClassBuilder(TrainablePredCntTANSEntropyCoder, 
        #     coding_table=ml_coding_table,
        #     coding_extra_symbols=ml_coding_extra_symbols,
        #     max_bits=16,
        #     # update_coding_table=True,
        #     num_predcnts=ParamSlot('num_predcnts', default=1, choices=list(range(1, 255))),
        #     # max_symbol=len(ml_coding_extra_symbols),
        #     table_log=9
        # ),
        ParamSlot("literals", 
            default=import_config_from_module(default_coder_literals),
            # choices=import_all_config_from_dir(),
        ),
        ParamSlot("offset", 
            default=import_config_from_module(default_coder_offset),
            # choices=import_all_config_from_dir(),
        ),
        ParamSlot("lit_length", 
            default=import_config_from_module(default_coder_lit_length),
            # choices=import_all_config_from_dir(),
        ),
        ParamSlot("match_length", 
            default=import_config_from_module(default_coder_match_length),
            # choices=import_all_config_from_dir(),
        ),
    )
)