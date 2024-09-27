from configs.import_utils import import_class_builder_from_module
from configs.env import DEFAULT_DATA_PATH
import os

from . import image_folder_dataset_wrapper as base_module

config = import_class_builder_from_module(base_module).update_args(
    root=os.path.join(DEFAULT_DATA_PATH, "Tecnick/rgb8bit"),
    enable_augmentation=False,
)