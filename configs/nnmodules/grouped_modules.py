from configs.class_builder import ClassBuilder, ParamSlot
from cbench.nn.base import GroupedNNTrainableModule

config = ClassBuilder(GroupedNNTrainableModule,
    ParamSlot("modules"),
).add_all_kwargs_as_param_slot()