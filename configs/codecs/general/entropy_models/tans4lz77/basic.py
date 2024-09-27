from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder.fse import TANSEntropyCoder, TrainablePredCntTANSEntropyCoder

config = ClassBuilder(TrainablePredCntTANSEntropyCoder, 
    coding_table=[0],
    coding_extra_symbols=[1 << bits for bits in range(32)],
    # table_distribution=2**(16 - np.ceil(np.log2(np.arange(1, 2048)))),
    # table_distribution=np.ones(3600),
    max_bits=31,
    update_coding_table=ParamSlot('update_coding_table', default=False, choices=[False, True]),
    update_coding_table_method=ParamSlot(choices=["recursive_split", "recursive_merge"]),
    auto_adjust_max_symbol=ParamSlot(),
    target_max_symbol=ParamSlot(),
    force_log2_extra_code=ParamSlot(default=False, choices=[False, True]),
    num_predcnts=ParamSlot('num_predcnts', default=1, choices=list(range(1, 255))),
    # max_symbol=32,
    table_log=ParamSlot(default=8),
    predcnt_table_log=ParamSlot(default=8),
)