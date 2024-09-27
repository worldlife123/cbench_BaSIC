from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, ClassBuilderDict, NamedParam
import copy

import configs.benchmark.lossless_compression_trainable as default_benchmark
import configs.benchmark.metrics.pytorch_distortion as default_metric
import configs.benchmark.metrics.bj_delta as bj_delta_metric

import configs.codecs.general.base as general_codec
import configs.codecs.general.grouped_variable_rate as grouped_codec
import configs.codecs.pycodecs.pil_jpeg as jpeg_codec

import configs.codecs.general.prior_models.base_lossless_autoencoder
import configs.codecs.general.prior_models.lossless_autoencoder_bbv2
import configs.codecs.general.prior_models.base_lossy_autoencoder
import configs.codecs.general.prior_models.lossy_autoencoder_google
import configs.codecs.general.prior_models.lossy_autoencoder_google_slimmable
import configs.codecs.general.prior_models.aev2_vqvae_v2backbone
import configs.codecs.general.prior_models.vqvae
import configs.codecs.general.prior_models.vqvae_v2
import configs.codecs.general.prior_models.vqvae_sp
import configs.codecs.general.prior_models.vqvae_pvq
import configs.codecs.general.prior_models.vqvae_pvq_v2
import configs.codecs.general.prior_models.vqvae_pretrained
import configs.codecs.general.prior_models.vqvae_selftrain
import configs.codecs.general.prior_models.vqvae_selftrain_gssoft
import configs.codecs.general.prior_models.vqvae_selftrain_sp

import configs.codecs.general.prior_models.prior_coders.gaussian
import configs.codecs.general.prior_models.prior_coders.cat_gaussian
import configs.codecs.general.prior_models.prior_coders.gaussian_embedding_cat
import configs.codecs.general.prior_models.prior_coders.beta_bernoulli_gaussian
import configs.codecs.general.prior_models.prior_coders.vq
import configs.codecs.general.prior_models.prior_coders.gaussian_vq
import configs.codecs.general.prior_models.prior_coders.vq_gaussian_embedding
import configs.codecs.general.prior_models.prior_coders.dist_gaussian
import configs.codecs.general.prior_models.prior_coders.dist_cat
import configs.codecs.general.prior_models.prior_coders.dist_cat_embedding
import configs.codecs.general.prior_models.prior_coders.dist_cat_embedding_gp
import configs.codecs.general.prior_models.prior_coders.dist_cat_embedding_snp
import configs.codecs.general.prior_models.prior_coders.dist_cat_sbp
import configs.codecs.general.prior_models.prior_coders.dist_sb
import configs.codecs.general.prior_models.prior_coders.dist_vq_unigaussian
import configs.codecs.general.prior_models.prior_coders.dist_vq_lrmultigaussian
import configs.codecs.general.prior_models.prior_coders.dist_c2vq_gaussian
import configs.codecs.general.prior_models.prior_coders.dist_c2d_gaussian
import configs.codecs.general.prior_models.prior_coders.hierarchical
import configs.codecs.general.prior_models.prior_coders.hierarchical_2layer
import configs.codecs.general.prior_models.prior_coders.compressai_coder
import configs.codecs.general.prior_models.prior_coders.compressai_coder_slimmable
import configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder
import configs.codecs.general.prior_models.prior_coders.compressai_scalehyperprior_coder
import configs.codecs.general.prior_models.prior_coders.compressai_jointautoregressive_gaussian_coder
# import configs.codecs.general.prior_models.prior_coders.compressai_meanscalehyperprior_coder
# import configs.codecs.general.prior_models.prior_coders.compressai_jointautoregressive_coder
import configs.codecs.general.prior_models.prior_coders.mcquic_coder
import configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq
import configs.codecs.general.prior_models.prior_coders.pgm_gaussian_maskconv2d
import configs.codecs.general.prior_models.prior_coders.pgm_gaussian_channelgroup
import configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d
import configs.codecs.general.prior_models.prior_coders.combined_pgm_coder

import configs.codecs.general.entropy_models.ans
import configs.codecs.general.entropy_models.rans
import configs.codecs.general.entropy_models.vq_clustered
import configs.codecs.general.entropy_models.dist_gaussian
import configs.codecs.general.entropy_models.latent_graph
import configs.codecs.general.entropy_models.lossy_dummy

import configs.codecs.general.preprocessors.twar


import configs.nnmodules.layers.hyperprior_a
import configs.nnmodules.layers.hyperprior_s
import configs.nnmodules.layers.hyperprior_ha
import configs.nnmodules.layers.hyperprior_hs
import configs.nnmodules.layers.hyperprior_ms_ha
import configs.nnmodules.layers.hyperprior_ms_hs
import configs.nnmodules.layers.pgm.groupconv
import configs.nnmodules.layers.pgm.hyperprior_a
import configs.nnmodules.layers.pgm.hyperprior_s
import configs.nnmodules.layers.pgm.hyperprior_s_agg
import configs.nnmodules.layers.pgm.hyperprior_s_agg_v2
import configs.nnmodules.layers.pgm.hyperprior_s_agg_v2_pre
import configs.nnmodules.layers.pgm.hyperprior_s_agg_v3
import configs.nnmodules.layers.pgm.hyperprior_s_agg_out
import configs.nnmodules.layers.pgm.hyperprior_s_no_agg_out
import configs.nnmodules.layers.pgm.slimmable_hyperprior_a
import configs.nnmodules.layers.pgm.slimmable_hyperprior_s
import configs.nnmodules.layers.pgm.slimmable_hyperprior_ha
import configs.nnmodules.layers.pgm.slimmable_hyperprior_hs
import configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_ha
import configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_hs
import configs.nnmodules.layers.pgm.slimmable_cheng2020_a
import configs.nnmodules.layers.pgm.slimmable_cheng2020_s
import configs.nnmodules.layers.pgm.slimmable_cheng2020_ha
import configs.nnmodules.layers.pgm.slimmable_cheng2020_hs
import configs.nnmodules.layers.pgm.slimmable_cheng2020_noattn_a
import configs.nnmodules.layers.pgm.slimmable_cheng2020_noattn_s
import configs.nnmodules.layers.pgm.slimmable_elic_a
import configs.nnmodules.layers.pgm.slimmable_elic_s
import configs.nnmodules.layers.pgm.slimmable_elic_ha
import configs.nnmodules.layers.pgm.slimmable_elic_hs
import configs.nnmodules.layers.pgm.slimmable_elic_noattn_a
import configs.nnmodules.layers.pgm.slimmable_elic_noattn_s
import configs.nnmodules.layers.pgm.hyperprior_ha
import configs.nnmodules.layers.pgm.hyperprior_hs
import configs.nnmodules.layers.param_generator.identity
import configs.nnmodules.layers.param_generator.nn_param
import configs.nnmodules.layers.param_generator.nnmodule_param_wrapper
import configs.nnmodules.layers.param_generator.group
import configs.nnmodules.layers.param_generator.increasing_vector
import configs.nnmodules.layers.param_generator.bernoulli
import configs.nnmodules.layers.param_generator.categorical
import configs.nnmodules.layers.param_generator.categorical_to_range
import configs.nnmodules.layers.param_generator.index
import configs.nnmodules.layers.param_generator.index_select
import configs.nnmodules.layers.param_generator.index_select_wrapper
import configs.nnmodules.layers.param_generator.resnet2d
import configs.nnmodules.layers.param_generator.tensor_split
import configs.nnmodules.layers.topogroup_maskconv_context
import configs.nnmodules.layers.adaptive_resize

import torch
import os

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
from configs.env import DEFAULT_PRETRAINED_PATH

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.trainer.nn_trainer as default_trainer

import configs.nnmodules.trainer.lightning_trainer as default_nn_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.imagenet_subset8000 as dataset_training
import configs.datasets.images.kodak as dataset_validation
import configs.datasets.images.kodak as dataset_testing

import configs.dataloaders.torch as default_dataloader

from cbench.nn.base import TorchCheckpointLoader

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

hyperprior_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225, # compressai hyperprior quality 3
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=128,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_a).update_slot_params(
            N=128,
            M=192,
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_ha).update_slot_params(
            N=128,
            M=192,
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_hs).update_slot_params(
            N=128,
            M=192,
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_s).update_slot_params(
            N=128,
            M=192,
        ),
    ),
)


import torch
hyperprior_scalable_slimmable_full_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    dynamic_node_generator_dict=ClassBuilderDict(
        sclevel=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index).update_slot_params(
            shape=(1, ),
            max=5,
        ),
    ),
    node_generator_dict=ClassBuilderDict(
        sclevel_slimmask=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            )
        ),
    ),
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder_slimmable).update_slot_params(
            entropy_bottleneck_channels_list=[48, 72, 96, 144, 192],
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_a).update_slot_params(
            in_channels=3,
            out_channels=[48, 72, 96, 144, 192],
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ha).update_slot_params(
            in_channels=192,
            out_channels=[48, 72, 96, 144, 192],
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_hs).update_slot_params(
            in_channels=192,
            out_channels=[48, 72, 96, 144, 192],
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
            in_channels=192,
            out_channels=3,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_inference_input_mapping=dict(
        x_y={"slimmask" : "pgm"},
        y_z={"slimmask" : "pgm"},
    ),
    latent_generative_input_mapping=dict(
        y_x={"slimmask" : "pgm"},
        z_y={"slimmask" : "pgm"},
        z={"sclevel" : "slim_level"}, # TODO:
    ),
    moniter_node_generator_output=True,
)

hyperprior_scalable_computation_slimmable_full_master_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    dynamic_node_generator_dict=ClassBuilderDict(
        sclevel=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index).update_slot_params(
            shape=(1, ),
            max=5,
        ),
    ),
    node_generator_dict=ClassBuilderDict(
        sclevel_pgmxy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            )
        ),
        sclevel_pgmyx=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            )
        ),
        sclevel_pgmyz=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            )
        ),
        sclevel_pgmzy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            )
        ),
    ),
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_a).update_slot_params(
            in_channels=3,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ha).update_slot_params(
            in_channels=192,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_hs).update_slot_params(
            in_channels=192,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
            in_channels=192,
            out_channels=3,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_inference_input_mapping=dict(
        x_y={"pgmxy" : "pgm"},
        y_z={"pgmyz" : "pgm"},
    ),
    latent_generative_input_mapping=dict(
        y_x={"pgmyx" : "pgm"},
        z_y={"pgmzy" : "pgm"},
        y={"z" : "prior"},
    ),
    # moniter_node_generator_output=True,
)

hyperprior_scalable_computation_slimmable_full_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    node_generator_dict=ClassBuilderDict(
        pgmxy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
        pgmyx=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
        pgmyz=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
        pgmzy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 1, 1, 5),
                init_method="value",
                init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
    ),
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_a).update_slot_params(
            in_channels=3,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ha).update_slot_params(
            in_channels=192,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_hs).update_slot_params(
            in_channels=192,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
            in_channels=192,
            out_channels=3,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_inference_input_mapping=dict(
        x_y={"pgmxy" : "pgm"},
        y_z={"pgmyz" : "pgm"},
    ),
    latent_generative_input_mapping=dict(
        y_x={"pgmyx" : "pgm"},
        z_y={"pgmzy" : "pgm"},
        y={"z" : "prior"},
    ),
    # moniter_node_generator_output=True,
)

hyperprior_scalable_computation_usdecoder_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    node_generator_dict=ClassBuilderDict(
        pgmyx=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(13, 1, 1, 13),
                init_method="value",
                init_value=torch.eye(13).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
    ),
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_a).update_slot_params(
            N=192,
            M=192,
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_ha).update_slot_params(
            N=192,
            M=192,
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_hs).update_slot_params(
            N=192,
            M=192,
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
            in_channels=192,
            out_channels=3,
            mid_channels_list=list(range(48, 192+12, 12)),
            training_self_distillation=True,
            training_self_distillation_loss_type="MSE",
            # use_sandwich_rule=True,
        ),
    ),
    latent_generative_input_mapping=dict(
        y_x={"pgmyx" : "pgm"},
    ),
    # moniter_node_generator_output=True,
)

hyperprior_scalable_computation_groupconv_full_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.hyperprior_a).update_slot_params(
            in_channels=3,
            in_groups=1,
            out_channels=192,
            out_groups=4,
            mid_channels_per_group=48,
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.hyperprior_ha).update_slot_params(
            in_channels=192,
            in_groups=4,
            out_channels=192,
            out_groups=4,
            mid_channels_per_group=48,
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.hyperprior_hs).update_slot_params(
            in_channels=192,
            in_groups=4,
            out_channels=192,
            out_groups=4,
            mid_channels_per_group=48,
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.hyperprior_s).update_slot_params(
            in_channels=192,
            in_groups=4,
            out_channels=3,
            out_groups=1,
            mid_channels_per_group=48,
        ),
    ),
    # moniter_node_generator_output=True,
)

hyperprior_scalable_computation_static_backbone_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225, # compressai hyperprior quality 3
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_a).update_slot_params(
            N=192,
            M=192,
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_ha).update_slot_params(
            N=192,
            M=192,
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_hs).update_slot_params(
            N=192,
            M=192,
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_s).update_slot_params(
            N=192,
            M=192,
        ),
    ),
)

config = GroupedCodecBenchmarkBuilder(
    codec_group_builder = ClassBuilderList( 

        # Non-autoregressive

        # SlimCAE
        
        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-scalable-slimmable-full")
        # .update_slot_params(
        #     entropy_coder=hyperprior_scalable_slimmable_full_entropy_coder_config,
        # ),

        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-scalable-slimmable-full-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_scalable_slimmable_full_entropy_coder_config).update_slot_params(
                        latent_node_entropy_coder_dict=ClassBuilderDict(
                                x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                                    lambda_rd=lambda_rd,
                                ),
                                y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

                                ),
                                z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder_slimmable).update_slot_params(
                                    entropy_bottleneck_channels_list=[48, 72, 96, 144, 192],
                                    use_inner_aux_opt=True,
                                ),
                            ),
                    ),
                )
                for lambda_rd in [39.015, 75.8625, 145.2225, 281.775, 541.875]]
            )
        ),

        
        # Slimmable Dynamic
        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-sc-slimmable-full-dynamic-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        latent_node_entropy_coder_dict=ClassBuilderDict(
                            x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                                lambda_rd=lambda_rd,
                            ),
                            y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

                            ),
                            z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                                entropy_bottleneck_channels=192,
                                use_inner_aux_opt=True,
                            ),
                        ),
                        # gradient_clipping_group=idx,
                    ),
                )
                for idx, lambda_rd in enumerate([39.015, 75.8625, 145.2225, 281.775, 541.875])]
            )
        ),

        # Universal Slimmable
        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-sc-usdecoder-dynamic-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_scalable_computation_usdecoder_dynamic_entropy_coder_config).update_slot_params(
                        latent_node_entropy_coder_dict=ClassBuilderDict(
                            x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                                lambda_rd=lambda_rd,
                            ),
                            y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

                            ),
                            z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                                entropy_bottleneck_channels=192,
                                use_inner_aux_opt=True,
                            ),
                        ),
                    ),
                )
                for lambda_rd in [39.015, 75.8625, 145.2225, 281.775, 541.875]]
            )
        ),

        # Ablations
        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-sc-groupconv-full-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_scalable_computation_groupconv_full_entropy_coder_config).update_slot_params(
                        latent_node_entropy_coder_dict=ClassBuilderDict(
                            x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                                lambda_rd=lambda_rd,
                            ),
                            y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

                            ),
                            z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                                entropy_bottleneck_channels=192,
                                use_inner_aux_opt=True,
                            ),
                        ),
                    ),
                )
                for lambda_rd in [39.015, 75.8625, 145.2225, 281.775]]
            )
        ),

        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-sc-static-backbone-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_scalable_computation_static_backbone_entropy_coder_config).update_slot_params(
                        latent_node_entropy_coder_dict=ClassBuilderDict(
                            x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                                lambda_rd=lambda_rd,
                            ),
                            y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder).update_slot_params(

                            ),
                            z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                                entropy_bottleneck_channels=192,
                                use_inner_aux_opt=True,
                            ),
                        ),
                    ),
                )
                for lambda_rd in [39.015, 75.8625, 145.2225, 281.775]]
            )
        ),

    ),
    benchmark_builder=
        import_config_from_module(default_benchmark)
        .set_override_name("lossy-graph-scalable-hyperprior")
        .update_slot_params(
            dataloader=default_testing_dataloader,
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataloader,
                dataloader_validation=default_nn_validation_dataloader,
                num_epoch=num_epoch,
                check_val_every_n_epoch=10,
                model_wrapper_config="compressai_model",
                trainer_config="pl_gpu_clipgrad" if gpu_available else "pl_base",
                # checkpoint_config=dict(every_n_epochs=10, save_top_k=-1),
            ),
            distortion_metric=import_class_builder_from_module(default_metric),
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
        ) \
)
