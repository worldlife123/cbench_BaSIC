from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
import copy

import configs.benchmark.lossless_compression_trainable as default_benchmark
import configs.benchmark.metrics.pytorch_distortion as default_metric
import configs.benchmark.metrics.bj_delta as bj_delta_metric

import configs.codecs.pycodecs.pil_jpeg
import configs.codecs.pycodecs.pil_webp
import configs.codecs.binary_codecs.bpg

import torch
import os

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
from configs.env import DEFAULT_PRETRAINED_PATH

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.trainer.nn_trainer as default_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.kodak
import configs.datasets.images.clic_val

default_testing_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(configs.datasets.images.kodak)#.update_slot_params(train=False),
)


config = GroupedCodecBenchmarkBuilder(
    codec_group_builder = ClassBuilderList( 
        import_class_builder_from_module(configs.codecs.pycodecs.pil_jpeg),
        import_class_builder_from_module(configs.codecs.pycodecs.pil_webp),
        import_class_builder_from_module(configs.codecs.binary_codecs.bpg),
        # import_class_builder_from_module(configs.codecs.pycodecs.flif),
    ),
    benchmark_builder=
        import_config_from_module(default_benchmark)
        .set_override_name("traditional-image-lossy-codecs")
        .update_slot_params(
            dataloader=default_testing_dataset,
            # extra_testing_dataloaders=ClassBuilderList(
            #     # import_config_from_module(wrapper_dataset).update_slot_params(
            #     #     dataset=import_config_from_module(configs.datasets.images.kodak),
            #     # ),
            #     import_config_from_module(wrapper_dataset).update_slot_params(
            #         dataset=import_config_from_module(configs.datasets.images.clic_val),
            #     ),
            # ),
            distortion_metric=import_class_builder_from_module(default_metric),
            testing_variable_rate_levels=[], # test all levels
            testing_variable_rate_bj_delta_metric=import_class_builder_from_module(bj_delta_metric).update_args(
                # reference_pts=(
                #     [8511.583333333334, 10874.708333333334, 16053.0, 20789.0, 24981.458333333332, 28877.75, 32428.208333333332, 35816.541666666664, 38627.916666666664, 41762.125, 44531.083333333336, 47399.541666666664, 50989.833333333336, 55428.666666666664, 60968.833333333336, 67308.29166666667, 77295.79166666667, 91414.625, 115539.375, 167221.33333333334],
                #     [21.413169665051942, 23.786108947554737, 26.58442181916143, 28.044018471934788, 29.04052847209138, 29.78050839733066, 30.375464959215932, 30.898806375277456, 31.303703646841317, 31.699692876513407, 32.05345114516326, 32.39420904827293, 32.78513774424156, 33.231695906909174, 33.79080756749713, 34.39384837320289, 35.24049243990481, 36.33038168142427, 37.913097935378445, 40.556718720073086],
                # ), # jpeg on kodak
                reference_pts=(
                    [5664.227328,    9127.428096,   14834.270208,   23050.911744, 33736.851456,   47523.299328,   64263.340032,   84910.227456],
                    [27.106351, 28.679134, 30.616753, 32.554935, 34.58096 , 36.720366, 38.80796 , 40.79492 ],
                ), # hyperprior on kodak
                mode=1, # BD-Rate mode
            ),
            testing_complexity_levels=[], # test all levels
            force_basic_testing=True,
        ) \
)
