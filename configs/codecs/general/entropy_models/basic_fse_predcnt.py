from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder import FSEEntropyCoder
from cbench.modules.entropy_coder.fse import TrainablePredCntTANSEntropyCoder

config = ClassBuilder(TrainablePredCntTANSEntropyCoder, 
    coding_table=[i for i in range(256)],
    coding_extra_symbols=[0] * 256,
    max_bits=8,
    table_log=12,
    update_coding_table=ParamSlot('update_coding_table', default=False, choices=[False, True]),
    auto_adjust_max_symbol=ParamSlot(),
    target_max_symbol=ParamSlot(),
    num_predcnts=ParamSlot('num_predcnts', default=1, choices=list(range(1, 255))),
)