from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.entropy_coder.latent_graph import CombinedLossyDummyEntropyCoder
from configs.import_utils import import_class_builder_from_module

from . import lossy_dummy as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(CombinedLossyDummyEntropyCoder)\
    .update_args(
        coders=ParamSlot("coders"),
    )\
    .add_all_kwargs_as_param_slot()
