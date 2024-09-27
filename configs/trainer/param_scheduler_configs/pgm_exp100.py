import math

def _em_loss_beta_var(epoch):
    return math.log(epoch/100) \
           if epoch > 100 else 0.0

def _mc_loss_weight_var(epoch):
    return 1.0 if epoch > 100 else 0.0

def _random_mask_weight_var(epoch):
    return 0.0 if epoch > 100 else 1.0

config = [
    ("model.prior_model.prior_coder.gs_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.entropy_coder.prior_coders.0.gs_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.entropy_coder.prior_coders.1.gs_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.entropy_coder.latent_node_entropy_coders.y.gs_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.prior_model.prior_coder.relax_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.entropy_coder.prior_coders.0.relax_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.entropy_coder.prior_coders.1.relax_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),
    ("model.entropy_coder.latent_node_entropy_coders.y.relax_temp", dict(
        scheduler_type="ExponentialLR",
        scheduler_config=dict(gamma=math.exp(math.log(0.5) / 100)),
    )),

    ("model.entropy_coder.prior_coders.em_loss_beta", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_em_loss_beta_var),
    )),
    ("model.entropy_coder.prior_coders.0.em_loss_beta", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_em_loss_beta_var),
    )),
    ("model.entropy_coder.prior_coders.1.em_loss_beta", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_em_loss_beta_var),
    )),
    ("model.entropy_coder.prior_coders.mc_loss_weight", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_mc_loss_weight_var),
    )),
    ("model.entropy_coder.prior_coders.0.mc_loss_weight", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_mc_loss_weight_var),
    )),
    ("model.entropy_coder.prior_coders.1.mc_loss_weight", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_mc_loss_weight_var),
    )),
    ("model.entropy_coder.prior_coders.random_mask_weight", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_random_mask_weight_var),
    )),
    ("model.entropy_coder.prior_coders.0.random_mask_weight", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_random_mask_weight_var),
    )),
    ("model.entropy_coder.prior_coders.1.random_mask_weight", dict(
        scheduler_type="LambdaLR",
        scheduler_config=dict(lr_lambda=_random_mask_weight_var),
    )),
]
