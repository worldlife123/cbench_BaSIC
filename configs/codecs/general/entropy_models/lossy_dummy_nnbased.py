from cbench.modules.entropy_coder.latent_graph import NNBasedLossyDummyEntropyCoder
from configs.import_utils import import_class_builder_from_module

from . import lossy_dummy as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(NNBasedLossyDummyEntropyCoder)\
    .add_all_kwargs_as_param_slot()
