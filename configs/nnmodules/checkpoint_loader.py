from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.base import TorchCheckpointLoader

config = ClassBuilder(TorchCheckpointLoader,
    ParamSlot("checkpoint_file"),
).add_all_kwargs_as_param_slot()