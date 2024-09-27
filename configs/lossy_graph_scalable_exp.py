from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, ClassBuilderDict, NamedParam

import torch

from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
import configs.benchmark.lossless_compression_trainable as default_benchmark
import configs.benchmark.metrics.pytorch_distortion as default_metric
import configs.benchmark.metrics.bj_delta as bj_delta_metric
import configs.trainer.nn_trainer as default_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.imagenet_subset8000 as dataset_training
import configs.datasets.images.kodak as dataset_validation
import configs.datasets.images.kodak as dataset_testing

import configs.dataloaders.torch as default_dataloader

import configs.presets.lossy_latent_graph_scalable_ar_models as default_models
import configs.presets.lossy_latent_graph_scalable_comp as comp_models
import configs.presets.lossy_latent_graph_scalable_ar_models_newbb as abl_models

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 32
batch_size_gpu = batch_size_total // num_gpus if num_gpus > 0 else batch_size_total
batch_size_cpu = 1
batch_size = batch_size_gpu if gpu_available else batch_size_cpu

num_epoch = 2000 if gpu_available else 1

default_nn_training_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_training),
)
default_nn_training_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_training_dataset,
    batch_size=batch_size,
)

default_nn_validation_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_validation),
)
default_nn_validation_dataloader = import_config_from_module(default_dataloader).update_slot_params(
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

benchmark_builder = (import_class_builder_from_module(default_benchmark)
    .update_slot_params(
        dataloader=default_testing_dataloader,
        trainer=import_class_builder_from_module(default_trainer).update_slot_params(
            dataloader_training=default_nn_training_dataloader,
            dataloader_validation=default_nn_validation_dataloader,
            num_epoch=num_epoch,
            check_val_every_n_epoch=10,
            model_wrapper_config="compressai_model",
            trainer_config="pl_gpu_clipgrad" if gpu_available else "pl_base",
            device="cuda",
            # checkpoint_config=dict(every_n_epochs=10, save_top_k=-1),
        ),
        distortion_metric=import_class_builder_from_module(default_metric).update_slot_params(
            metrics=["psnr", "ms-ssim"]
        ),
        testing_variable_rate_levels=[], # test all levels
        testing_variable_rate_bj_delta_metric=import_class_builder_from_module(bj_delta_metric).update_args(
            reference_pts=(
                [5664.227328,    9127.428096,   14834.270208,   23050.911744, 33736.851456,   47523.299328,   64263.340032,   84910.227456],
                [27.106351, 28.679134, 30.616753, 32.554935, 34.58096 , 36.720366, 38.80796 , 40.79492 ],
            ), # hyperprior on kodak
            mode=1, # BD-Rate mode
        ),
        testing_complexity_levels=[], # test all levels
        force_basic_testing=True,
        force_testing_device="cuda",
    )
).set_override_name("lossy-graph-scalable")



config = GroupedCodecBenchmarkBuilder(
    codec_group_builder = (
        import_config_from_module(default_models) +\
        import_config_from_module(abl_models) +\
        import_config_from_module(comp_models)
    ),
    benchmark_builder=benchmark_builder,
)
