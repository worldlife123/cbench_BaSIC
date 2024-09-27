from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from cbench.modules.entropy_coder.latent_graph import LatentGraphicalANSEntropyCoder

config = ClassBuilder(LatentGraphicalANSEntropyCoder)\
    .add_all_kwargs_as_param_slot()