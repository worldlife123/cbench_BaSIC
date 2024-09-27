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
# import configs.codecs.general.prior_models.prior_coders.compressai_meanscalehyperprior_coder
# import configs.codecs.general.prior_models.prior_coders.compressai_jointautoregressive_coder
import configs.codecs.general.prior_models.prior_coders.mcquic_coder
import configs.codecs.general.prior_models.prior_coders.dist_ar_cat_svq
import configs.codecs.general.prior_models.prior_coders.pgm_gaussian_maskconv2d
import configs.codecs.general.prior_models.prior_coders.pgm_gaussian_channelgroup
import configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d

import configs.codecs.general.entropy_models.rans
import configs.codecs.general.entropy_models.vq_clustered
import configs.codecs.general.entropy_models.dist_gaussian
import configs.codecs.general.entropy_models.latent_graph
import configs.codecs.general.entropy_models.lossy_dummy
import configs.codecs.general.entropy_models.sf_dummy

import configs.codecs.general.preprocessors.twar

import configs.nnmodules.layers.vae_encoder
import configs.nnmodules.layers.vae_decoder
import configs.nnmodules.layers.hyperprior_a
import configs.nnmodules.layers.hyperprior_s
import configs.nnmodules.layers.hyperprior_ha
import configs.nnmodules.layers.hyperprior_hs
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
import configs.nnmodules.layers.pgm.hyperprior_ha
import configs.nnmodules.layers.pgm.hyperprior_hs
import configs.nnmodules.layers.param_generator.identity
import configs.nnmodules.layers.param_generator.nn_param
import configs.nnmodules.layers.param_generator.increasing_vector
import configs.nnmodules.layers.param_generator.bernoulli
import configs.nnmodules.layers.param_generator.categorical
import configs.nnmodules.layers.param_generator.categorical_to_range
import configs.nnmodules.layers.param_generator.index
import configs.nnmodules.layers.param_generator.index_select
import configs.nnmodules.layers.param_generator.index_select_wrapper
import configs.nnmodules.layers.param_generator.convtranspose2d
import configs.nnmodules.layers.param_generator.resnet2d
import configs.nnmodules.layers.param_generator.transformer2d


import torch
import os

from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, NamedParam
from configs.utils.group_benchmark_builder import GroupedCodecBenchmarkBuilder
from configs.env import DEFAULT_PRETRAINED_PATH

import configs.benchmark.lossless_compression_trainable as default_benchmark

import configs.trainer.nn_trainer as default_trainer

import configs.datasets.images.image_dataset_wrapper as wrapper_dataset
import configs.datasets.images.imagenet_subset8000 as dataset_training
import configs.datasets.images.kodak as dataset_validation
import configs.datasets.images.kodak as dataset_testing

import configs.dataloaders.torch as default_dataloader

from cbench.nn.base import TorchCheckpointLoader

gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
batch_size_total = 8 * num_gpus
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

hyperprior_model_config = import_config_from_module(configs.codecs.general.prior_models.lossy_autoencoder_google).update_slot_params(
    prior_coder=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_scalehyperprior_coder).update_slot_params(
        N=128,
        M=192,
    ),
)

hyperprior_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
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

hyperprior_ar_base_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
            lambda_rd=145.2225,
        ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
            in_channels=192,
            default_topo_group_method="none",
            use_param_merger=False,
        ),
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
            M=384,
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.hyperprior_s).update_slot_params(
            N=128,
            M=192,
        ),
    ),
)

exp_name = "lossy-latent-graph-topogroup-exp"

config = GroupedCodecBenchmarkBuilder(
    codec_group_builder = ClassBuilderList( 

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior")
        # .update_slot_params(
        #     entropy_coder=hyperprior_entropy_coder_config,
        # ),

        # base models
        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-base")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config),
        ),

        # 2-stage
        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-learned-randprob0.999-g2-s2-p2-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        channel_groups=2,
                        param_merger_expand_bottleneck=True,
                        training_mc_sampling=True,
                        training_mc_loss_type="vimco",
                        training_mc_num_samples=2,
                        training_mc_sampling_share_sample_batch=True,
                        training_pgm_logits_use_random_prob=0.999,
                        topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                            out_channels=4,
                            initial_width=2,
                            num_initial_dense=2,
                            out_width=2,
                        ),
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-channelwise-g2-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        channel_groups=2,
                        default_topo_group_method="channelwise",
                        param_merger_expand_bottleneck=True,
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-checkerboard-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        default_topo_group_method="checkerboard",
                        param_merger_expand_bottleneck=True,
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-channelg2-random")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=2,
        #                 default_topo_group_method="random",
        #                 param_merger_expand_bottleneck=True,
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),


        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-learned-g2-s2-p2-ft-random")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=2,
        #                 param_merger_expand_bottleneck=True,
        #                 training_mc_sampling=True,
        #                 training_mc_loss_type="vimco",
        #                 training_mc_num_samples=2,
        #                 training_mc_sampling_share_sample_batch=True,
        #                 topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
        #                     out_channels=4,
        #                     initial_width=2,
        #                     num_initial_dense=2,
        #                     out_width=2,
        #                 ),
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-channelg2-random/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),

        
        # 4-stage

        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-channelwise-g4-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        channel_groups=4,
                        default_topo_group_method="channelwise",
                        param_merger_expand_bottleneck=True,
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-channelwise-checkerboard-g2-ft")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=2,
        #                 default_topo_group_method="channelwise-checkerboard",
        #                 param_merger_expand_bottleneck=True,
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),

        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-raster2x2-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        default_topo_group_method="raster2x2",
                        param_merger_expand_bottleneck=True,
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-channelg4-random")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=4,
        #                 default_topo_group_method="random",
        #                 param_merger_expand_bottleneck=True,
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),


        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-learned-g4-s4-p2-ft-random")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        channel_groups=4,
                        param_merger_expand_bottleneck=True,
                        training_mc_sampling=True,
                        training_mc_loss_type="vimco",
                        training_mc_num_samples=2,
                        training_mc_sampling_share_sample_batch=True,
                        topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                            out_channels=16,
                            initial_width=2,
                            num_initial_dense=2,
                            out_width=2,
                        ),
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-channelg4-random/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),
        # 8-stage

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-learned-g12-s8-p2-ft-random")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=12,
        #                 param_merger_expand_bottleneck=True,
        #                 training_mc_sampling=True,
        #                 training_mc_loss_type="vimco",
        #                 training_mc_num_samples=2,
        #                 training_mc_sampling_share_sample_batch=True,
        #                 topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
        #                     out_channels=96,
        #                     initial_width=2,
        #                     num_initial_dense=2,
        #                     out_width=2,
        #                 ),
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-channelg12-random/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),

        # 10-stage 

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-channelwise-g10-ft")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=12,
        #                 default_topo_group_method="channelwise-g10",
        #                 param_merger_expand_bottleneck=True,
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),

        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-elic-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        channel_groups=12,
                        default_topo_group_method="elic",
                        param_merger_expand_bottleneck=True,
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        # import_class_builder_from_module(general_codec)
        # .set_override_name("hyperprior-ar-channelg12-random")
        # .update_slot_params(
        #     entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
        #         latent_node_entropy_coder_dict=ClassBuilderDict(
        #             x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #                 lambda_rd=145.2225,
        #             ),
        #             y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
        #                 in_channels=192,
        #                 channel_groups=12,
        #                 default_topo_group_method="random",
        #                 param_merger_expand_bottleneck=True,
        #             ),
        #             z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
        #                 entropy_bottleneck_channels=128,
        #                 freeze_params=True,
        #             ),
        #         ),
        #         freeze_inference_modules=True,
        #         freeze_generative_modules=True,
        #     ),
        # ).update_args(
        #     checkpoint_loader=TorchCheckpointLoader(
        #         f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
        #         key="state_dict", prefix="model.", strict=False
        #     )
        # ),


        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-learned-g12-s10-p2-ft-random")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        channel_groups=12,
                        param_merger_expand_bottleneck=True,
                        training_mc_sampling=True,
                        training_mc_loss_type="vimco",
                        training_mc_num_samples=2,
                        training_mc_sampling_share_sample_batch=True,
                        topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                            out_channels=120,
                            initial_width=2,
                            num_initial_dense=2,
                            out_width=2,
                        ),
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-channelg12-random/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),

        # dynamic stage

        import_class_builder_from_module(general_codec)
        .set_override_name("hyperprior-ar-scanline-ft")
        .update_slot_params(
            entropy_coder=copy.deepcopy(hyperprior_ar_base_entropy_coder_config).update_slot_params(
                latent_node_entropy_coder_dict=ClassBuilderDict(
                    x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
                        lambda_rd=145.2225,
                    ),
                    y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                        in_channels=192,
                        default_topo_group_method="scanline",
                        param_merger_expand_bottleneck=True,
                    ),
                    z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
                        entropy_bottleneck_channels=128,
                        freeze_params=True,
                    ),
                ),
                freeze_inference_modules=True,
                freeze_generative_modules=True,
            ),
        ).update_args(
            checkpoint_loader=TorchCheckpointLoader(
                f"experiments/{exp_name}/hyperprior-ar-base/last.ckpt",
                key="state_dict", prefix="model.", strict=False
            )
        ),


    ),
    benchmark_builder=
        import_config_from_module(default_benchmark)
        .set_override_name(exp_name)
        .update_slot_params(
            dataloader=default_testing_dataloader,
            trainer=import_config_from_module(default_trainer).update_slot_params(
                dataloader_training=default_nn_training_dataloader,
                dataloader_validation=default_nn_validation_dataloader,
                num_epoch=num_epoch,
                check_val_every_n_epoch=5,
                model_wrapper_config="compressai_model",
                trainer_config="pl_gpu_clipgrad" if gpu_available else "pl_base",
                # checkpoint_config=dict(every_n_epochs=10, save_top_k=-1),
            ),
            distortion_metric=import_class_builder_from_module(default_metric),
            testing_variable_rate_levels=[], # test all levels
            testing_variable_rate_bj_delta_metric=import_class_builder_from_module(bj_delta_metric).update_args(
                reference_pts=(
                    [8511.583333333334, 10874.708333333334, 16053.0, 20789.0, 24981.458333333332, 28877.75, 32428.208333333332, 35816.541666666664, 38627.916666666664, 41762.125, 44531.083333333336, 47399.541666666664, 50989.833333333336, 55428.666666666664, 60968.833333333336, 67308.29166666667, 77295.79166666667, 91414.625, 115539.375, 167221.33333333334],
                    [21.413169665051942, 23.786108947554737, 26.58442181916143, 28.044018471934788, 29.04052847209138, 29.78050839733066, 30.375464959215932, 30.898806375277456, 31.303703646841317, 31.699692876513407, 32.05345114516326, 32.39420904827293, 32.78513774424156, 33.231695906909174, 33.79080756749713, 34.39384837320289, 35.24049243990481, 36.33038168142427, 37.913097935378445, 40.556718720073086],
                ), # jpeg on kodak
                mode=1, # BD-Rate mode
            ),
            testing_complexity_levels=[], # test all levels
            force_basic_testing=True,
            force_testing_device="cuda",
        ) \
)
