from cbench.nn.trainer import LightningTrainer

from configs.import_utils import import_class_builder_from_module, import_all_config_from_dir
from configs.class_builder import ClassBuilder, ParamSlot

from . import basic_trainer as base_module

config = import_class_builder_from_module(base_module)\
    .update_class(LightningTrainer)\
    .add_all_kwargs_as_param_slot()\
    .update_args(
    train_loader=ParamSlot(),
    val_loader=ParamSlot(),
    test_loader=ParamSlot(),
    model_wrapper_config=ParamSlot("model_wrapper_config", 
        choices=import_all_config_from_dir("model_wrapper_configs", caller_file=__file__),
        default="empty",
    ),
    trainer_config=ParamSlot("trainer_config", 
        choices=import_all_config_from_dir("trainer_configs", caller_file=__file__),
        default="empty",
    ),
    param_scheduler_configs=ParamSlot("param_scheduler_configs",
        choices=import_all_config_from_dir("param_scheduler_configs", caller_file=__file__),
        default="empty",
    ),
    # max_epochs=ParamSlot(default=100),
)