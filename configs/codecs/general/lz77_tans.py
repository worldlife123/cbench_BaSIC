from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from configs.import_utils import import_config_from_module

from . import lz77_fse as base_module

from cbench.codecs.general_codec import GeneralCodec
from cbench.modules.entropy_coder import GroupedEntropyCoder, FSEEntropyCoder, TANSEntropyCoder
import numpy as np
from scipy.special import softmax

config = import_config_from_module(base_module).update_slot_params(
    entropy_coder=ClassBuilder(
        GroupedEntropyCoder,
        ClassBuilderList(
            ClassBuilder(FSEEntropyCoder, coder=FSEEntropyCoder.CODER_HUF),
            ClassBuilder(TANSEntropyCoder, 
                table_distribution=softmax(np.arange(0, -65536, -1) / 65536),
                max_bits=31,
                table_log=8
            ),
            ClassBuilder(TANSEntropyCoder, 
                table_distribution=softmax(np.arange(0, -256, -1) / 256),
                max_bits=16,
                table_log=9
            ),
            ClassBuilder(TANSEntropyCoder, 
                table_distribution=softmax(np.arange(0, -256, -1) / 256),
                max_bits=16,
                table_log=9
            ),
        )
    ),
)