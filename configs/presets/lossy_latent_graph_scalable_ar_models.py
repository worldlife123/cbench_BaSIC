from configs.import_utils import import_config_from_module, import_class_builder_from_module
from configs.class_builder import ClassBuilder, ClassBuilderList, ClassBuilderDict, ClassBuilderObjRef, NamedParam
import copy, os
import torch

import configs.codecs.general.base as general_codec
import configs.codecs.general.grouped_variable_rate as grouped_codec

import configs.codecs.general.prior_models.prior_coders.compressai_coder
import configs.codecs.general.prior_models.prior_coders.compressai_coder_slimmable
import configs.codecs.general.prior_models.prior_coders.compressai_gaussian_coder
import configs.codecs.general.prior_models.prior_coders.compressai_scalehyperprior_coder
import configs.codecs.general.prior_models.prior_coders.compressai_jointautoregressive_gaussian_coder
import configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d
import configs.codecs.general.prior_models.prior_coders.combined_pgm_coder

import configs.codecs.general.entropy_models.ans
import configs.codecs.general.entropy_models.latent_graph
import configs.codecs.general.entropy_models.lossy_dummy

import configs.nnmodules.layers.hyperprior_a
import configs.nnmodules.layers.hyperprior_s
import configs.nnmodules.layers.hyperprior_ha
import configs.nnmodules.layers.hyperprior_hs
import configs.nnmodules.layers.hyperprior_ms_ha
import configs.nnmodules.layers.hyperprior_ms_hs
import configs.nnmodules.layers.pgm.slimmable_hyperprior_a
import configs.nnmodules.layers.pgm.slimmable_hyperprior_s
import configs.nnmodules.layers.pgm.slimmable_hyperprior_ha
import configs.nnmodules.layers.pgm.slimmable_hyperprior_hs
import configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_ha
import configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_hs
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

import configs.nnmodules.checkpoint_loader as checkpoint_loader

from . import common_trainer

stage1_trainer = import_class_builder_from_module(common_trainer)

stage2_trainer = import_class_builder_from_module(common_trainer).update_slot_params(
    max_epochs=500,
    check_val_every_n_epoch=5,
    model_wrapper_config="compressai_model_ft",
)

skip_trainer = import_class_builder_from_module(common_trainer).update_slot_params(
    max_epochs=0,
    model_wrapper_config="compressai_model_ft",
)

def get_trainer_output_param_file(trainer):
    return os.path.join(trainer.output_dir, "params.pkl")

lambda_rds = [39.015, 75.8625, 145.2225, 281.775]

common_inference_models = dict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_a).update_slot_params(
            in_channels=3,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_ha).update_slot_params(
            in_channels=192,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
)

common_generative_models = dict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_hs).update_slot_params(
            in_channels=192,
            out_channels=384,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
            in_channels=192,
            out_channels=3,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
)

common_generative_models_sd = dict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_hs).update_slot_params(
            in_channels=192,
            out_channels=384,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
            in_channels=192,
            out_channels=3,
            mid_channels_list=[48, 72, 96, 144, 192],
            training_self_distillation=True,
            training_self_distillation_loss_type="MSE",
            use_sandwich_rule=True,
        ),
)

common_entropy_coders = dict(
        # x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #     lambda_rd=145.2225,
        # ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
            in_channels=192,
            training_no_quantize_for_likelihood=True,
            training_output_straight_through=True,
            default_topo_group_method="scanline",
            topo_group_context_model=import_class_builder_from_module(configs.nnmodules.layers.topogroup_maskconv_context).update_slot_params(
                    
            ),
        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
            training_output_straight_through=True,
        ),
)

hyperprior_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
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
    use_lossy_compression=True,
    lossy_compression_lambda_rd=145.2225,
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    latent_node_entropy_coder_dict=ClassBuilderDict(
        **common_entropy_coders,
    ),
    latent_inference_dict=ClassBuilderDict(
        **common_inference_models,
    ),
    latent_generative_dict=ClassBuilderDict(
        **common_generative_models,
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

hyperprior_ar_scalable_computation_slimmable_full_dynamic_combined_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
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
        pgmy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(5, 5),
                init_method="value",
                init_value=torch.eye(5),
                fix_params=True,
            ),
            fix_for_inference=True,
            training_no_params=True,
        ),
    ),
    latent_node_inference_topo_order=["x", "y", "z"],
    latent_node_generative_topo_order=["z", "y", "x"],
    use_lossy_compression=True,
    lossy_compression_lambda_rd=145.2225,
    latent_node_entropy_coder_dict=ClassBuilderDict(
        # x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
        #     lambda_rd=145.2225,
        # ),
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.combined_pgm_coder).update_slot_params(
            coders=ClassBuilderList(
                # scanline
                import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                    in_channels=192,
                    training_no_quantize_for_likelihood=True,
                    training_output_straight_through=True,
                    default_topo_group_method="scanline",
                    topo_group_context_model=import_class_builder_from_module(configs.nnmodules.layers.topogroup_maskconv_context).update_slot_params(
                    
                    ),
                ),
                # 8-stage
                import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                    in_channels=192,
                    training_no_quantize_for_likelihood=True,
                    channel_groups=4,
                    training_mc_sampling=True,
                    training_mc_loss_type="vimco",
                    training_mc_num_samples=2,
                    training_mc_sampling_share_sample_batch=True,
                    training_pgm_logits_use_random_num_iter=5000000,
                    topo_group_context_model=import_class_builder_from_module(configs.nnmodules.layers.topogroup_maskconv_context).update_slot_params(
                    
                    ),
                    topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                        out_channels=32,
                        initial_width=2,
                        num_initial_dense=2,
                        out_width=2,
                    ),
                ),
                # 6-stage
                import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                    in_channels=192,
                    training_no_quantize_for_likelihood=True,
                    channel_groups=4,
                    training_mc_sampling=True,
                    training_mc_loss_type="vimco",
                    training_mc_num_samples=2,
                    training_mc_sampling_share_sample_batch=True,
                    training_pgm_logits_use_random_num_iter=5000000,
                    topo_group_context_model=import_class_builder_from_module(configs.nnmodules.layers.topogroup_maskconv_context).update_slot_params(
                    
                    ),
                    topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                        out_channels=24,
                        initial_width=2,
                        num_initial_dense=2,
                        out_width=2,
                    ),
                ),
                # 4-stage
                import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                    in_channels=192,
                    training_no_quantize_for_likelihood=True,
                    training_mc_sampling=True,
                    training_mc_loss_type="vimco",
                    training_mc_num_samples=2,
                    training_mc_sampling_share_sample_batch=True,
                    training_pgm_logits_use_random_num_iter=5000000,
                    topo_group_context_model=import_class_builder_from_module(configs.nnmodules.layers.topogroup_maskconv_context).update_slot_params(
                    
                    ),
                    topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                        out_channels=16,
                        initial_width=2,
                        num_initial_dense=2,
                        out_width=2,
                    ),
                ),
                # 2-stage
                import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
                    in_channels=192,
                    training_no_quantize_for_likelihood=True,
                    channel_groups=2,
                    param_merger_expand_bottleneck=True,
                    training_mc_sampling=True,
                    training_mc_loss_type="vimco",
                    training_mc_num_samples=2,
                    training_mc_sampling_share_sample_batch=True,
                    training_pgm_logits_use_random_num_iter=5000000,
                    topo_group_context_model=import_class_builder_from_module(configs.nnmodules.layers.topogroup_maskconv_context).update_slot_params(
                    
                    ),
                    topo_group_predictor=import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
                        out_channels=4,
                        initial_width=2,
                        num_initial_dense=2,
                        out_width=2,
                    ),
                ),
            ),
            training_use_max_capacity=True,
        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
            training_output_straight_through=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        **common_inference_models,
    ),
    latent_generative_dict=ClassBuilderDict(
        **common_generative_models,
    ),
    latent_inference_input_mapping=dict(
        x_y={"pgmxy" : "pgm"},
        y_z={"pgmyz" : "pgm"},
    ),
    latent_generative_input_mapping=dict(
        y_x={"pgmyx" : "pgm"},
        z_y={"pgmzy" : "pgm"},
        y={"pgmy" : "blend_weight", "z" : "prior"},
    ),
    # moniter_node_generator_output=True,
)


# hyperprior_ar_scalable_computation_slimmable_full_combined_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
#     node_generator_dict=ClassBuilderDict(
#         pgmxy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#         pgmyx=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#         pgmyz=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#         pgmzy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#     ),
#     latent_node_inference_topo_order=["x", "y", "z"],
#     latent_node_generative_topo_order=["z", "y", "x"],
#     latent_node_entropy_coder_dict=ClassBuilderDict(
#         x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
#             lambda_rd=145.2225,
#         ),
#         y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.combined_pgm_coder).update_slot_params(
#             coders=ClassBuilderList(
#                 *[import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                     in_channels=192,
#                     training_no_quantize_for_likelihood=True,
#                     training_output_straight_through=True,
#                     default_topo_group_method="scanline",
#                     param_merger_expand_bottleneck=True,
#                 ),] * 5
#             )
#         ),
#         z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.combined_pgm_coder).update_slot_params(
#             coders=ClassBuilderList(
#                 *[import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
#                     entropy_bottleneck_channels=192,
#                     use_inner_aux_opt=True,
#                     training_output_straight_through=True,
#                 ),] * 5
#             )
#         ),
#     ),
#     latent_inference_dict=ClassBuilderDict(
#         x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_a).update_slot_params(
#             in_channels=3,
#             out_channels=192,
#             mid_channels_list=[48, 72, 96, 144, 192],
#         ),
#         y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_ha).update_slot_params(
#             in_channels=192,
#             out_channels=192,
#             mid_channels_list=[48, 72, 96, 144, 192],
#         ),
#     ),
#     latent_generative_dict=ClassBuilderDict(
#         z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_ms_hs).update_slot_params(
#             in_channels=192,
#             out_channels=384,
#             mid_channels_list=[48, 72, 96, 144, 192],
#         ),
#         y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_hyperprior_s).update_slot_params(
#             in_channels=192,
#             out_channels=3,
#             mid_channels_list=[48, 72, 96, 144, 192],
#         ),
#     ),
#     latent_inference_input_mapping=dict(
#         x_y={"pgmxy" : "pgm"},
#         y_z={"pgmyz" : "pgm"},
#     ),
#     latent_generative_input_mapping=dict(
#         y_x={"pgmyx" : "pgm"},
#         z_y={"pgmzy" : "pgm"},
#         y={"pgmxy" : "blend_weight", "z" : "prior"},
#         z={"pgmyz" : "blend_weight"},
#     ),
#     # moniter_node_generator_output=True,
# )

# hyperprior_ar_scalable_computation_slimmable_full_dynamic_topogroup_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
#     node_generator_dict=ClassBuilderDict(
#         pgmxy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,

#         ),
#         pgmyx=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#         pgmyz=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#         pgmzy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                 shape=(5, 1, 1, 5),
#                 init_method="value",
#                 init_value=torch.eye(5).flip(-1).unsqueeze(1).unsqueeze(1),
#                 fix_params=True,
#             ),
#             fix_for_inference=True,
#         ),
#         pgmy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
#             batched_generator=ClassBuilderList(
#                 import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
#                     shape=(0,),
#                     no_params=True,
#                 ),
#                 import_class_builder_from_module(configs.nnmodules.layers.param_generator.group).update_slot_params(
#                     batched_generator=ClassBuilderList(
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
#                             out_channels=32,
#                             initial_width=2,
#                             num_initial_dense=2,
#                             out_width=2,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".weight",
#                             unsqueeze_params=True,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".bias",
#                             unsqueeze_params=True,
#                         ),
#                     )
#                 ),
#                 import_class_builder_from_module(configs.nnmodules.layers.param_generator.group).update_slot_params(
#                     batched_generator=ClassBuilderList(
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
#                             out_channels=16,
#                             initial_width=2,
#                             num_initial_dense=2,
#                             out_width=2,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".weight",
#                             unsqueeze_params=True,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".bias",
#                             unsqueeze_params=True,
#                         ),
#                     )
#                 ),
#                 import_class_builder_from_module(configs.nnmodules.layers.param_generator.group).update_slot_params(
#                     batched_generator=ClassBuilderList(
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
#                             out_channels=8,
#                             initial_width=2,
#                             num_initial_dense=2,
#                             out_width=2,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".weight",
#                             unsqueeze_params=True,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".bias",
#                             unsqueeze_params=True,
#                         ),
#                     )
#                 ),
#                 import_class_builder_from_module(configs.nnmodules.layers.param_generator.group).update_slot_params(
#                     batched_generator=ClassBuilderList(
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.resnet2d).update_slot_params(
#                             out_channels=4,
#                             initial_width=2,
#                             num_initial_dense=2,
#                             out_width=2,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".weight",
#                             unsqueeze_params=True,
#                         ),
#                         import_class_builder_from_module(configs.nnmodules.layers.param_generator.nnmodule_param_wrapper).update_slot_params(
#                             module=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#                                 in_channels=192,
#                                 default_topo_group_method="scanline",
#                                 param_merger_expand_bottleneck=True,
#                             ),
#                             name_filter=".bias",
#                             unsqueeze_params=True,
#                         ),
#                     )
#                 ),
#             ),
#             fix_for_inference=True,
#         ),
#     ),
#     latent_node_inference_topo_order=["x", "y", "z"],
#     latent_node_generative_topo_order=["z", "y", "x"],
#     latent_node_entropy_coder_dict=ClassBuilderDict(
#         x=import_class_builder_from_module(configs.codecs.general.entropy_models.lossy_dummy).update_slot_params(
#             lambda_rd=145.2225,
#         ),
#         y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
#             in_channels=192,
#             training_no_quantize_for_likelihood=True,
#             training_output_straight_through=True,
#             default_topo_group_method="scanline",
#             channel_groups=4,
#             param_merger_expand_bottleneck=True,
#             training_mc_sampling=True,
#             training_mc_loss_type="vimco",
#             training_mc_num_samples=2,
#             training_mc_sampling_share_sample_batch=True,
#             pgm_include_dynamic_kernel=True,
#             pgm_include_dynamic_kernel_full=True,
#             training_pgm_logits_use_random_num_iter=5000000,
#         ),
#         z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
#             entropy_bottleneck_channels=192,
#             use_inner_aux_opt=True,
#             training_output_straight_through=True,
#         ),
#     ),
#     latent_inference_dict=ClassBuilderDict(
#         **common_inference_models,
#     ),
#     latent_generative_dict=ClassBuilderDict(
#         **common_generative_models,
#     ),
#     latent_inference_input_mapping=dict(
#         x_y={"pgmxy" : "pgm"},
#         y_z={"pgmyz" : "pgm"},
#     ),
#     latent_generative_input_mapping=dict(
#         y_x={"pgmyx" : "pgm"},
#         z_y={"pgmzy" : "pgm"},
#         y={"pgmy" : "pgm", "z" : "prior"},
#     ),
#     # moniter_node_generator_output=True,
# )


stage1_trainer_shared = stage1_trainer.clone(copy_slot_data=True, share_built_object=True)

config = ClassBuilderList( 

        # Autoregressive (fixed ar)
        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-ar-sc-slimmable-full-dynamic-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        lossy_compression_lambda_rd=lambda_rd,
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            trainer=stage1_trainer_shared,
        ),

        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-ar-sc-slimmable-full-dynamic-grouped-ft-ssim")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        lossy_compression_lambda_rd=lambda_rd,
                        lossy_compression_distortion_type="ms-ssim",
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            checkpoint_loader=import_class_builder_from_module(checkpoint_loader).update_slot_params(
                checkpoint_file=ClassBuilderObjRef(stage1_trainer_shared, 
                    obj_func=get_trainer_output_param_file
                ),
                strict=False
            ),
            trainer=stage2_trainer,
        ),



        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-ar-sc-slimmable-full-dynamic-grouped-greedy-search-8level")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        lossy_compression_lambda_rd=lambda_rd,
                        complexity_level_greedy_search=True,
                        complexity_level_greedy_search_dataset=import_class_builder_from_module(common_trainer, variable_name="default_testing_dataloader"),
                        complexity_level_greedy_search_dataset_cached=True,
                        complexity_level_greedy_search_num_levels=8,
                        complexity_level_controller_nodes=["pgmxy", "pgmyz", "pgmzy", "pgmyx"]
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            checkpoint_loader=import_class_builder_from_module(checkpoint_loader).update_slot_params(
                checkpoint_file=ClassBuilderObjRef(stage1_trainer_shared, 
                    obj_func=get_trainer_output_param_file
                ),
                strict=False
            ),
            trainer=skip_trainer,
        ),

        # Autoregressive (full scalable ar)
        import_class_builder_from_module(grouped_codec)
        .set_override_name("hyperprior-ar-sc-slimmable-full-dynamic-combined-dynamic-entropy-coder-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_ar_scalable_computation_slimmable_full_dynamic_combined_dynamic_entropy_coder_config).update_slot_params(
                        lossy_compression_lambda_rd=lambda_rd,
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            trainer=stage1_trainer,
        ),


)
