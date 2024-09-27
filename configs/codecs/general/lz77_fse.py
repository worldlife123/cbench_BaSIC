from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from configs.import_utils import import_config_from_module

from . import base as base_module
from .preprocessors import lz77 as preprocessor_module

from cbench.codecs.binary_codec import GeneralCodec
# from cbench.modules.preprocessor import LZ77Preprocessor
from cbench.modules.entropy_coder import GroupedEntropyCoder, FSEEntropyCoder


config = import_config_from_module(base_module).update_slot_params(
    preprocessor=import_config_from_module(preprocessor_module),
    entropy_coder=ClassBuilder(
        GroupedEntropyCoder,
        ClassBuilderList(
            ClassBuilder(FSEEntropyCoder),#, coder=FSEEntropyCoder.CODER_HUF),
            ClassBuilder(FSEEntropyCoder, table_log=8),
            ClassBuilder(FSEEntropyCoder, table_log=9),
            ClassBuilder(FSEEntropyCoder, table_log=9),
        )
    ),
)