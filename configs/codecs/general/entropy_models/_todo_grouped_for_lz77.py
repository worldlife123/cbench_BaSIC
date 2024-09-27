from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder import GroupedEntropyCoder, FSEEntropyCoder

# TODO: how to collaborate ClassBuilderList and ParamSlot?
config = ClassBuilder(
    GroupedEntropyCoder,
    ClassBuilderList(
        ParamSlot("literals", default=ClassBuilder(FSEEntropyCoder, coder=FSEEntropyCoder.CODER_HUF)),
        ParamSlot("offset", default=ClassBuilder(FSEEntropyCoder)),
        ParamSlot("literal_length", default=ClassBuilder(FSEEntropyCoder)),
        ParamSlot("match_length", default=ClassBuilder(FSEEntropyCoder)),
    )
)