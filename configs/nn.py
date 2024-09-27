import math
import torch
import os

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.sync_utils import OSSUtils
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
from configs.env import DEFAULT_PRETRAINED_PATH

import configs.benchmark.lossless_compression_trainable as default_benchmark

# import configs.codecs.general.presets.vqvae_sp_prior as default_codec
import configs.codecs.general.base as general_codec
# import configs.codecs.general.prior_models.vqvae_sp as prior_model
# default_codec_config = import_config_from_module(general_codec).update_slot_params(
#     prior_model=import_config_from_module(prior_model).update_slot_params(
#         single_decoder=True
#     ),
# )
import configs.codecs.general.prior_models.vqvae
import configs.codecs.general.prior_models.vqvae_v2
import configs.codecs.general.prior_models.vqvae_sp
import configs.codecs.general.prior_models.vqvae_pvq
import configs.codecs.general.prior_models.vqvae_pvq_v2
import configs.codecs.general.prior_models.vqvae_pretrained
import configs.codecs.general.prior_models.vqvae_selftrain
import configs.codecs.general.prior_models.vqvae_selftrain_gssoft
import configs.codecs.general.prior_models.vqvae_selftrain_sp

import configs.trainer.nn_trainer as default_trainer
import configs.nnmodules.lightning_trainer as default_nn_trainer

import configs.nnmodules.pretrained.vqvae
import configs.nnmodules.pretrained.vqvae_gssoft

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.torchvision.cifar10 as dataset_training
import configs.datasets.torchvision.cifar10_test as dataset_testing
# import configs.datasets.images.random_image_generator as dataset_training
# import configs.datasets.images.random_image_generator as dataset_testing

import configs.dataloaders.torch as default_dataloader

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 256
batch_size_gpu = batch_size_total // num_gpus if num_gpus > 0 else batch_size_total
batch_size_cpu = 1
batch_size = batch_size_gpu if gpu_available else batch_size_cpu

num_epoch = 1000 if gpu_available else 1

default_nn_training_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_training),
)
default_nn_training_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_training_dataset,
    batch_size=batch_size,
)
training_data_size = len(default_nn_training_dataset.build_class())

default_nn_validation_dataset = import_config_from_module(wrapper_dataset).update_slot_params(
    dataset=import_config_from_module(dataset_testing),
)
default_nn_validation_dataloader = import_config_from_module(default_dataloader).update_slot_params(
    dataset=default_nn_validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)

self_training_nn_trainer = import_class_builder_from_module(default_nn_trainer).update_slot_params(
    train_loader=default_nn_training_dataloader,
    val_loader=default_nn_validation_dataloader,
    trainer_config="pl_gpu" if gpu_available else "pl_base",
    max_epochs=num_epoch,
)

def on_initialize_start_hook(trainer):
    rel_outdir = os.path.relpath(trainer.output_dir, os.getcwd())
    oss = OSSUtils()
    oss.download_directory(rel_outdir, rel_outdir)

def on_train_end_hook(trainer):
    rel_outdir = os.path.relpath(trainer.output_dir, os.getcwd())
    oss = OSSUtils()
    oss.upload_directory(rel_outdir, rel_outdir, force_overwrite_dir=True)

pre_training_nn_trainer = import_class_builder_from_module(default_nn_trainer).update_slot_params(
    train_loader=default_nn_training_dataloader,
    val_loader=default_nn_validation_dataloader,
    trainer_config="pl_gpu" if gpu_available else "pl_base",
    max_epochs=num_epoch,
).update_args(
    on_initialize_start_hook=on_initialize_start_hook,
    on_train_end_hook=on_train_end_hook,
)

def _exp_anneal_rate_calculate(init=1.0, min=0.01):
    anneal_scale = init / min
    num_steps = training_data_size * num_epoch * batch_size / batch_size_total
    anneal_rate = math.log(anneal_scale) / num_steps
    return anneal_rate

config = GroupedCodecBenchmarkBuilder(
    codec_group_builder=ClassBuilderList(
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
                relax_temp_anneal=True,
                relax_temp=1.0,
                relax_temp_min=0.01,
                relax_temp_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                test_sampling=True,
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
                relax_temp_anneal=True,
                relax_temp=1.0,
                relax_temp_min=0.01,
                relax_temp_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                dist_type="AsymptoticRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                dist_type="AsymptoticRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                test_sampling=True,
                dist_type="AsymptoticRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                test_sampling=True,
                dist_type="AsymptoticRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
    
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                dist_type="DoubleRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
                relax_temp_anneal=True,
                relax_temp=1.0,
                relax_temp_min=0.01,
                relax_temp_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                dist_type="DoubleRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
                relax_temp_anneal=True,
                relax_temp=1.0,
                relax_temp_min=0.01,
                relax_temp_anneal_rate=_exp_anneal_rate_calculate(),
                test_sampling=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                dist_type="DoubleRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
                relax_temp_anneal=True,
                relax_temp=1.0,
                relax_temp_min=0.01,
                relax_temp_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                test_sampling=True,
                dist_type="DoubleRelaxedOneHotCategorical",
                gs_anneal=True,
                gs_temp=1.0,
                gs_temp_min=0.01,
                gs_anneal_rate=_exp_anneal_rate_calculate(),
                relax_temp_anneal=True,
                relax_temp=1.0,
                relax_temp_min=0.01,
                relax_temp_anneal_rate=_exp_anneal_rate_calculate(),
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                test_sampling=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                test_sampling=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                kl_cost=0.0,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                kl_cost=-1.0,
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                commitment_cost_gs=0.25,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                use_st_gumbel=True,
                commitment_cost_gs=0.25,
                kl_cost=0.0,
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                commitment_cost_gs=0.25,
                commitment_over_exp=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
                use_gssoft_vq=True,
                commitment_cost_gs=0.25,
                commitment_over_exp=True,
                test_sampling=True,
            )
        ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
        #         use_gssoft_vq=True,
        #         kl_cost=0.25,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
        #         use_gssoft_vq=True,
        #         kl_cost=0.0,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
        #         latent_dim=8,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_v2).update_slot_params(
        #         latent_dim=8,
        #         use_gssoft_vq=True,
        #     )
        # ),

        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq)
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq).update_slot_params(
        #         use_gssoft_vq=True,
        #     )
        # ),        
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq).update_slot_params(
        #         gs_temp=2.0,
        #         gs_anneal=True,
        #         use_gssoft_vq=True,
        #     )
        # ),

        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq_v2).update_slot_params(
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq_v2).update_slot_params(
        #         gs_anneal_scheme="anneal",
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq_v2).update_slot_params(
        #         use_gssoft_vq=True,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq_v2).update_slot_params(
        #         use_gssoft_vq=True,
        #         gs_anneal_scheme="anneal",
        #     )
        # ),


        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain).update_slot_params(
        #         trainer=self_training_nn_trainer,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain_gssoft).update_slot_params(
        #         trainer=self_training_nn_trainer,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain_gssoft).update_slot_params(
        #         trainer=self_training_nn_trainer,
        #         training_soft_samples=False,
        #     )
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain_gssoft).update_slot_params(
        #         trainer=self_training_nn_trainer,
        #         gs_anneal_scheme="anneal",
        #     )
        # ),
        
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain_sp).update_slot_params(
        #         trainer=self_training_nn_trainer,
        #     )
        # ),

        # TODO: it seems pretrain model block training...
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pretrained).update_slot_params(
        #         model=import_config_from_module(configs.nnmodules.pretrained.vqvae).update_slot_params(
        #             trainer=pre_training_nn_trainer,
        #         )
        #     )
        # ),

        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_sp)
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_sp).update_slot_params(
        #         single_decoder=True,
        #     ),
        # ),
        # import_config_from_module(general_codec).update_slot_params(
        #     prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_sp).update_slot_params(
        #         single_decoder=True,
        #         use_batch_norm=True,
        #     ),
        # ),
    ),
    benchmark_builder=
        import_config_from_module(default_benchmark).update_slot_params(
            # codec_group=default_codec_config,
            # .batch_update_slot_params(**{
            #     # "entropy_coder.0.0.num_predcnts": ClassBuilder.SLOT_ALL_CHOICES
            #     "entropy_coder.num_predcnts": [1, 2, 4, 8, 16, 32, 64, 128]
            # }),
            dataloader=default_nn_validation_dataloader,
            # training_dataloader=import_config_from_module(default_dataloader).update_slot_params(
            #     dataset=import_config_from_module(wrapper_dataset).update_slot_params(
            #         dataset=import_config_from_module(dataset_training),
            #     ),
            #     # batch_size=256,
            # ),
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataloader,
                dataloader_validation=default_nn_validation_dataloader,
                num_epoch=num_epoch,
                model_wrapper_config="vqvae",
                trainer_config="pl_gpu" if gpu_available else "pl_base",
            ),
        ) \
        # .batch_update_slot_params(**{
        #     "codec.entropy_coder.0.0.num_predcnts": ClassBuilder.SLOT_ALL_CHOICES
        # })
)
# print(list(config.iter_parameters()))