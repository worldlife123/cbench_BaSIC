from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, ClassBuilderDict, ClassBuilderObjRef, NamedParam

import configs.nnmodules.checkpoint_loader as checkpoint_loader
import configs.nnmodules.trainer.lightning_trainer as default_nn_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.imagenet_subset8000 as dataset_training
import configs.datasets.images.kodak as dataset_validation
import configs.datasets.images.kodak as dataset_testing

import configs.dataloaders.torch as default_dataloader

import torch

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 32
batch_size_gpu = batch_size_total // num_gpus if num_gpus > 0 else batch_size_total
batch_size_cpu = 1
batch_size = batch_size_gpu if gpu_available else batch_size_cpu

default_nn_training_dataset = import_class_builder_from_module(wrapper_dataset).update_slot_params(
    dataset=import_class_builder_from_module(dataset_training),
)
default_nn_training_dataloader = import_class_builder_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_training_dataset,
    batch_size=batch_size,
)

default_nn_validation_dataset = import_class_builder_from_module(wrapper_dataset).update_slot_params(
    dataset=import_class_builder_from_module(dataset_validation),
)
default_nn_validation_dataloader = import_class_builder_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_validation_dataset,
    batch_size=1,
    shuffle=False,
)

default_testing_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_testing),
)
default_testing_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_testing_dataset,
    batch_size=1,
    shuffle=False,
)

config = import_class_builder_from_module(default_nn_trainer).update_slot_params(
    train_loader=default_nn_training_dataloader,
    val_loader=default_nn_validation_dataloader,
    max_epochs=2000,
    check_val_every_n_epoch=10,
    model_wrapper_config="compressai_model",
    trainer_config="pl_gpu_clipgrad",
)