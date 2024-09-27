from configs.import_utils import import_config_from_module
from . import basic as base_module

config = import_config_from_module(base_module).update_args(
    use_sequences=True,
)

