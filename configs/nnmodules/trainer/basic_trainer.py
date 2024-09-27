from cbench.nn.trainer import BasicNNTrainer

from configs.import_utils import import_config_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(BasicNNTrainer,
).add_all_kwargs_as_param_slot()