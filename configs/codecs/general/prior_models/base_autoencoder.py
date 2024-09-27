from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder_v2 import AutoEncoderPriorModel

config = ClassBuilder(AutoEncoderPriorModel,
    distortion_type=ParamSlot(),
    train_mc_sampling=ParamSlot(),
    test_mc_sampling=ParamSlot(),
    mc_sampling_size=ParamSlot(),
    mc_sampling_use_kl_weight=ParamSlot(),
    train_simulated_annealing=ParamSlot(),
    anneal_temperature_param_name=ParamSlot(),
    use_vamp_prior=ParamSlot(),
    vamp_input_size=ParamSlot(),
).add_all_kwargs_as_param_slot()