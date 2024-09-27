from cbench.nn.trainer import SimpleNNTrainer

from configs.import_utils import import_config_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(SimpleNNTrainer,
    train_loader=ParamSlot(),
    val_loader=ParamSlot(),
    test_loader=ParamSlot(),
    max_epochs=ParamSlot(default=100),
).add_all_kwargs_as_param_slot()