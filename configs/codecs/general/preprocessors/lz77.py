from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.preprocessor import LZ77Preprocessor

config = ClassBuilder(LZ77Preprocessor,
    level=ParamSlot("level", 
        choices={i: i for i in range(1, 23)},
        default=3,
    ),
    relative_offset_codes=ParamSlot(),
    relative_offset_mode=ParamSlot(),
)