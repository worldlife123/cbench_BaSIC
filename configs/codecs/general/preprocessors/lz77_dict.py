from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_class_builder_from_module
from cbench.modules.preprocessor import LZ77DictPreprocessor

from . import lz77 as base_module

config = import_class_builder_from_module(base_module) \
    .update_class(LZ77DictPreprocessor) \
    .update_args(
        dict_size=ParamSlot("dict_size", default=32*1024), # 32KB
        dict_relative_offset=ParamSlot("dict_relative_offset", default=False),
    )
# ClassBuilder(LZ77DictPreprocessor,
#     level=ParamSlot("level", 
#         choices={i: i for i in range(1, 23)},
#         default=3,
#     ),
# )