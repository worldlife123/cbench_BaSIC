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

import configs.nnmodules.checkpoint_loader as checkpoint_loader

from . import common_trainer

stage1_trainer = import_class_builder_from_module(common_trainer)

stage2_trainer = import_class_builder_from_module(common_trainer).update_slot_params(
    max_epochs=500,
    check_val_every_n_epoch=5,
    model_wrapper_config="compressai_model_ft",
)

def get_trainer_output_param_file(trainer):
    return os.path.join(trainer.output_dir, "params.pkl")

lambda_rds = [39.015, 75.8625, 145.2225, 281.775]

hyperprior_cheng2020_noattn_mn128_backbone_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
    node_generator_dict=ClassBuilderDict(
        pgmxy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(4, 1, 1, 4),
                init_method="value",
                init_value=torch.eye(4).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
        pgmyx=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(4, 1, 1, 4),
                init_method="value",
                init_value=torch.eye(4).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
        pgmyz=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(4, 1, 1, 4),
                init_method="value",
                init_value=torch.eye(4).flip(-1).unsqueeze(1).unsqueeze(1),
                fix_params=True,
            ),
            fix_for_inference=True,
        ),
        pgmzy=import_class_builder_from_module(configs.nnmodules.layers.param_generator.index_select_wrapper).update_slot_params(
            batched_generator=import_class_builder_from_module(configs.nnmodules.layers.param_generator.nn_param).update_slot_params(
                shape=(4, 1, 1, 4),
                init_method="value",
                init_value=torch.eye(4).flip(-1).unsqueeze(1).unsqueeze(1),
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
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
            in_channels=128,
            default_topo_group_method="scanline",
        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=128,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_cheng2020_noattn_a).update_slot_params(
            in_channels=3,
            out_channels=128,
            mid_channels_list=[32, 64, 96, 128],
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_cheng2020_ha).update_slot_params(
            in_channels=128,
            out_channels=128,
            mid_channels_list=[32, 64, 96, 128],
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_cheng2020_hs).update_slot_params(
            in_channels=128,
            out_channels=256,
            mid_channels_list=[32, 64, 96, 128],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_cheng2020_noattn_s).update_slot_params(
            in_channels=128,
            out_channels=3,
            mid_channels_list=[32, 64, 96, 128],
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


hyperprior_elic_noattn_mn192_backbone_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config = import_class_builder_from_module(configs.codecs.general.entropy_models.latent_graph).update_slot_params(
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
        y=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.pgm_topogroup_gaussian_maskconv2d).update_slot_params(
            in_channels=192,
            default_topo_group_method="scanline",
        ),
        z=import_class_builder_from_module(configs.codecs.general.prior_models.prior_coders.compressai_coder).update_slot_params(
            entropy_bottleneck_channels=192,
            use_inner_aux_opt=True,
        ),
    ),
    latent_inference_dict=ClassBuilderDict(
        x_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_elic_noattn_a).update_slot_params(
            in_channels=3,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_z=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_elic_ha).update_slot_params(
            in_channels=192,
            out_channels=192,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
    ),
    latent_generative_dict=ClassBuilderDict(
        z_y=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_elic_hs).update_slot_params(
            in_channels=192,
            out_channels=384,
            mid_channels_list=[48, 72, 96, 144, 192],
        ),
        y_x=import_class_builder_from_module(configs.nnmodules.layers.pgm.slimmable_elic_noattn_s).update_slot_params(
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
)

stage1_trainer_shared = stage1_trainer.clone(copy_slot_data=True, share_built_object=True)
stage1_trainer_shared2 = stage1_trainer.clone(copy_slot_data=True, share_built_object=True)

config = ClassBuilderList( 

        # Autoregressive (Cheng2020 backbone)
        # dynamic
        import_class_builder_from_module(grouped_codec)
        .set_override_name("cheng2020-noattn-mn128-sc-slimmable-full-dynamic-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_cheng2020_noattn_mn128_backbone_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        **{
                            "latent_node_entropy_coder_dict.x.lambda_rd" : lambda_rd,
                        }
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            trainer=stage1_trainer_shared,
        ),

        import_class_builder_from_module(grouped_codec)
        .set_override_name("cheng2020-noattn-mn128-sc-slimmable-full-dynamic-grouped-ft-ssim")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_cheng2020_noattn_mn128_backbone_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        **{
                            "latent_node_entropy_coder_dict.x.lambda_rd" : lambda_rd,
                            "latent_node_entropy_coder_dict.x.distortion_type" : "ms-ssim",
                        }
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

        # Autoregressive (ELIC backbone)
        # dynamic
        import_class_builder_from_module(grouped_codec)
        .set_override_name("elic-noattn-mn192-sc-slimmable-full-dynamic-grouped")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_elic_noattn_mn192_backbone_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        **{
                            "latent_node_entropy_coder_dict.x.lambda_rd" : lambda_rd,
                        }
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            trainer=stage1_trainer_shared2,
        ),

        import_class_builder_from_module(grouped_codec)
        .set_override_name("elic-noattn-mn192-sc-slimmable-full-dynamic-grouped-ft-ssim")
        .update_slot_params(
            codecs=ClassBuilderList(*[
                import_class_builder_from_module(general_codec)
                .update_slot_params(
                    entropy_coder=copy.deepcopy(hyperprior_elic_noattn_mn192_backbone_ar_scalable_computation_slimmable_full_dynamic_entropy_coder_config).update_slot_params(
                        **{
                            "latent_node_entropy_coder_dict.x.lambda_rd" : lambda_rd,
                            "latent_node_entropy_coder_dict.x.distortion_type" : "ms-ssim",
                        }
                    ),
                )
                for lambda_rd in lambda_rds]
            )
        ).update_args(
            checkpoint_loader=import_class_builder_from_module(checkpoint_loader).update_slot_params(
                checkpoint_file=ClassBuilderObjRef(stage1_trainer_shared2, 
                    obj_func=get_trainer_output_param_file
                ),
                strict=False
            ),
            trainer=stage2_trainer,
        ),

)
