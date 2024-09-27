from configs.class_builder import ClassBuilder, ClassBuilderList, ParamSlot
from configs.import_utils import import_config_from_module

from .. import base as base_module
from ..preprocessors import bytes2np as preprocessor_module
from ..entropy_models import basic_fse_predcnt as entropy_coder_module

config = import_config_from_module(base_module).update_slot_params(
    preprocessor=import_config_from_module(preprocessor_module),
    entropy_coder=import_config_from_module(entropy_coder_module),
)
