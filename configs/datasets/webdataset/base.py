from configs.class_builder import ClassBuilder, ParamSlot

import torch
import webdataset as wds

config = ClassBuilder(
    wds.WebDataset,
    ParamSlot("urls"),
).add_all_kwargs_as_param_slot()

