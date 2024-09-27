import torch

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.sync_utils import OSSUtils
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder

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
import configs.codecs.general.prior_models.vqvae_selftrain
import configs.codecs.general.prior_models.vqvae_selftrain_gssoft
import configs.codecs.general.prior_models.vqvae_selftrain_sp

import configs.trainer.nn_trainer as default_trainer
import configs.nnmodules.lightning_trainer as default_nn_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.imagenet32_train as dataset_training
import configs.datasets.images.imagenet32_val as dataset_testing

import configs.dataloaders.torch as default_dataloader

batch_size_gpu = 64
batch_size_cpu = 1

default_nn_training_dataset = import_config_from_module(default_dataloader).update_slot_params(
    dataset=import_config_from_module(wrapper_dataset).update_slot_params(
        dataset=import_config_from_module(dataset_training),
    ),
    batch_size=batch_size_gpu if torch.cuda.is_available() else batch_size_cpu,
)

default_nn_validation_dataset = import_config_from_module(default_dataloader).update_slot_params(
    dataset=import_config_from_module(wrapper_dataset).update_slot_params(
        dataset=import_config_from_module(dataset_testing),
    ),
    batch_size=batch_size_gpu if torch.cuda.is_available() else batch_size_cpu,
)

self_training_nn_trainer = import_class_builder_from_module(default_nn_trainer).update_slot_params(
    train_loader=default_nn_training_dataset,
    val_loader=default_nn_validation_dataset,
    trainer_config="pl_gpu" if torch.cuda.is_available() else "pl_base",
    max_epochs=50,
)

config = GroupedCodecBenchmarkBuilder(
    codec_group_builder=ClassBuilderList(
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae).update_slot_params(
                use_batch_norm=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae).update_slot_params(
                latent_dim=8,
                use_batch_norm=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae).update_slot_params(
                latent_dim=8,
                use_gssoft_vq=True,
                use_batch_norm=True,
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq).update_slot_params(
                latent_dim=8,
                use_batch_norm=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq).update_slot_params(
                latent_dim=8,
                use_gssoft_vq=True,
                use_batch_norm=True,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_config_from_module(configs.codecs.general.prior_models.vqvae_pvq).update_slot_params(
                gs_temp=2.0,
                gs_anneal=True,
                latent_dim=8,
                use_gssoft_vq=True,
                use_batch_norm=True,
            )
        ),

        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain).update_slot_params(
                trainer=self_training_nn_trainer,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain_gssoft).update_slot_params(
                trainer=self_training_nn_trainer,
            )
        ),
        import_config_from_module(general_codec).update_slot_params(
            prior_model=import_class_builder_from_module(configs.codecs.general.prior_models.vqvae_selftrain_sp).update_slot_params(
                trainer=self_training_nn_trainer,
            )
        ),

    ),
    benchmark_builder=
        import_config_from_module(default_benchmark).update_slot_params(
            dataloader=default_nn_validation_dataset,
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataset,
                dataloader_validation=default_nn_validation_dataset,
                num_epoch=50,
                model_wrapper_config="empty",
                trainer_config="pl_gpu" if torch.cuda.is_available() else "pl_base",
            ),
        ) \
        # .batch_update_slot_params(**{
        #     "codec.entropy_coder.0.0.num_predcnts": ClassBuilder.SLOT_ALL_CHOICES
        # })
)
# print(list(config.iter_parameters()))