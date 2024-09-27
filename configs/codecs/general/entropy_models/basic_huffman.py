from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder import FSEEntropyCoder

config = ClassBuilder(FSEEntropyCoder, coder=FSEEntropyCoder.CODER_HUF)