from typing import List, Optional, Dict, Any, Sequence, Tuple, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, EarlyStopping

import time
import os
import logging
from tqdm import tqdm

from cbench.utils.logger import setup_logger

from .base import BasicNNTrainer, NNTrainableModule, SelfTrainableInterface

from cbench.benchmark.trainer import BasicTrainer
from cbench.data.base import DataLoaderInterface
from cbench.modules.base import BaseModule, TrainableModuleInterface
from cbench.utils.logging_utils import SmoothedValue

class ParamSchedulerWrapper(object):
    def __init__(self, param : nn.Parameter, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.param = param
        self.lr_scheduler = lr_scheduler
        self.optimizer = self.lr_scheduler.optimizer
        # initialize param
        self.param_initial = param.detach()
        value = self.lr_scheduler.get_last_lr()[0]
        self.param.fill_(self.param_initial * value)

    def state_dict(self):
        return dict(
            param=self.param.data,
            lr_scheduler=self.lr_scheduler.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.param.data = state_dict['param']
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def step(self, *args, **kwargs):
        self.optimizer.step() # reduce warning from lr_scheduler
        self.lr_scheduler.step(*args, **kwargs)
        value = self.lr_scheduler.get_last_lr()[0]
        self.param.fill_(self.param_initial * value)


def make_optimizer(
        model : nn.Module,
        optimizer_type="Adam",
        base_lr=0.001,
        weight_decay=0,
        bias_lr_factor=1.0,
        weight_decay_bias=None,
        aux_id=-1,
        param_name_filter="",
        logger=None,
        **kwargs
    ) -> torch.optim.Optimizer:
    logger_func = logger.debug if logger is not None else print
    if optimizer_type == "None":
        return None
    params = []

    logger_func("Params to optimize for " + ("aux optimizer {}:".format(aux_id) if aux_id >= 0 else "main optimizer:"))
    # param_name_filter = re.compile(param_name_filter)
    for key, value in model.named_parameters():
        if not param_name_filter in key: continue
        if not value.requires_grad:
            continue
        p_aux_id = value.aux_id if hasattr(value, "aux_id") else -1
        if p_aux_id != aux_id:
            continue
        lr = base_lr
        weight_decay = weight_decay
        if "bias" in key:
            lr *= bias_lr_factor
            weight_decay = weight_decay if weight_decay_bias is None else weight_decay_bias
        # lr modifier defined in the parameter
        if hasattr(value, "lr_modifier"):
            if value.lr_modifier == 0: continue
            lr *= value.lr_modifier
        if hasattr(value, "weight_decay_modifier"):
            weight_decay *= value.weight_decay_modifier
        gradient_clipping_group = value.gradient_clipping_group if hasattr(value, "gradient_clipping_group") else -1
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "gradient_clipping_group" : gradient_clipping_group}]
        logger_func("{}: \t size {} \t lr {} \t weight_decay {}".format(key, value.numel(), lr, weight_decay))
    
    if len(params) == 0:
        logger.warning("No params to optimize!")
        return None

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, base_lr, **kwargs)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, base_lr, **kwargs)
    elif optimizer_type == "Adabelief":
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(params, base_lr, **kwargs)
    else:
        raise NotImplementedError('solver %s not supported' % optimizer_type)
    return optimizer


def make_scheduler(optimizer, scheduler_type="None", logger=None, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    logger_func = logger.debug if logger is not None else print
    if optimizer is None: 
        scheduler = None
    elif scheduler_type == "None":
        scheduler = None # torch.optim.lr_scheduler.ConstantLR(optimizer)
    else:
        # TODO: support for custom scheduler
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **kwargs)

    return scheduler


def make_optimizer_with_scheduler(
        model,
        scheduler_type="None",
        scheduler_config=dict(), 
        logger=None,
        **kwargs,
    ):
    optimizer = make_optimizer(model, logger=logger, **kwargs)
    scheduler = make_scheduler(optimizer, scheduler_type, logger=logger, **scheduler_config)

    return optimizer, scheduler


def make_dummy_optimizer_with_param_scheduler(
    model,
    param_name="",
    strict=False,
    scheduler_type="None",
    scheduler_config=dict(), 
    logger=None,
    **kwargs,
):
    logger_func = logger.warn if logger is not None else print
    param = None
    names_all = []
    for key, value in model.named_parameters():
        names_all.append(key)
        if key == param_name:
            param = value
            continue
    if param is None:
        if strict:
            raise KeyError(f"Parameter {param_name} not found! Avaiable parameters are: {names_all}")
        else:
            logger_func(f"Parameter {param_name} not found! Skipping param scheduler!")
            return None, None
    elif param.requires_grad == True:
        raise ValueError(f"Parameter {param_name} has gradient! Set requires_grad=False to use a param scheduler!")
    
    # trick: set lr as the parameter to make lr_scheduler to adjust the parameter
    dummy_optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = make_scheduler(dummy_optimizer, 
        scheduler_type=scheduler_type, 
        logger=logger,
        **scheduler_config
    )

    param_scheduler = ParamSchedulerWrapper(param, scheduler)
    return dummy_optimizer, param_scheduler


class TorchGeneralTrainer(BasicTrainer):
    def __init__(self, dataloader_training: DataLoaderInterface, *args, 
        dataloader_validation=None,
        dataloader_testing=None,
        model_wrapper_config=dict(),
        trainer_config=dict(),
        param_scheduler_configs : List[Tuple[List[str], Dict[str, Any]]] = [],
        checkpoint_config=dict(),
        # optimizer_class=None,
        # optimizer_config: Optional[Dict[str, Any]] = None,
        # scheduler_class=None,
        # scheduler_config: Optional[Dict[str, Any]] = None,
        num_epoch=100, 
        check_val_every_n_epoch=1,
        device="cpu",
        float32_matmul_precision=None,
        **kwargs) -> None:
        super().__init__(dataloader_training, *args, 
            use_iterative_training=True, 
            num_epoch=num_epoch, 
            **kwargs
        )
        self.dataloader_training = self.dataloader # alias
        # by default testing data is the same as validation data
        self.dataloader_validation = dataloader_validation if dataloader_validation is not None else dataloader_testing
        self.dataloader_testing = dataloader_testing if dataloader_testing is not None else dataloader_validation

        self.model_wrapper_config = model_wrapper_config
        self.trainer_config = trainer_config
        # self.trainer_config["max_epochs"]=9999
        # self.trainer_config["num_sanity_val_steps"]=0
        # self.trainer_config["strategy"]=None
        self.param_scheduler_configs = param_scheduler_configs
        # self.trainer_config.update(max_epochs=self.num_epoch) # num_epoch parameter
        # self.optimizer_config = optimizer_config
        # self.scheduler_config = scheduler_config
        self.checkpoint_config = checkpoint_config

        self.check_val_every_n_epoch = check_val_every_n_epoch
        # NOTE: in most cases device is managed by self.trainer (BasicNNTrainer)
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Tensor Core config
        try:
            if float32_matmul_precision is not None:
                torch.set_float32_matmul_precision(float32_matmul_precision)
        except:
            self.logger.warning("set_float32_matmul_precision not available in this environment!")

        self.trainer = None # not initialized!

    def initialize(self, module: TrainableModuleInterface, *args,
            load_checkpoint=True,
            save_checkpoint=True,
            checkpoint_file="params.pkl",
            initial_seed=None,
            use_tflogger=True,
            tflogger_name="training_logs",
            **kwargs
        ):
        # optimizer = None
        # # TODO: optmizer and schedular class input?
        # if self.optimizer_config is not None:
        #     parameters = list(module.iter_trainable_parameters())
        #     if len(parameters) > 0:
        #         optimizer = make_optimizer(**self.optimizer_config)
        
        # scheduler = None
        # if self.scheduler_config is not None:
        #     if len(parameters) > 0:
        #         scheduler = make_scheduler(optimizer=optimizer, **self.scheduler_config)

        self._load_checkpoint = load_checkpoint
        self._save_checkpoint = save_checkpoint
        self._checkpoint_file = checkpoint_file

        # reproducibility
        if initial_seed is not None:
            pl.seed_everything(initial_seed, workers=True)

        # search for self-trained modules and do training
        # TODO: how to deal with recursive trainable modules? Maybe we should do a bottom-up search?
        # if isinstance(module, BaseModule):
        #     for name, submodule in module.get_named_submodules():
        #         if isinstance(submodule, SelfTrainableInterface) and submodule.is_trainer_valid():
        #             trainer_output_dir = os.path.join(self.output_dir, name)
        #             if not submodule.is_trainer_setup():
        #                 submodule.setup_trainer_engine(
        #                     output_dir=trainer_output_dir,
        #                     logger=setup_logger(name + ":trainer", outdir=trainer_output_dir, label='trainer_log')
        #                 )
        #             submodule.do_train()
        #             # freeze parameters in self-trained modules to avoid further update!
        #             # for param in submodule.parameters():
        #             #     param.requires_grad = False
        #             # submodule.freeze()

        # use inner trainer for SelfTrainableModule and skip training
        if isinstance(module, SelfTrainableInterface) and module.is_trainer_valid():
            self.logger.warning("SelfTrainableModule trains on its own! Will skip training...")
            if not module.is_trainer_setup():
                module.setup_trainer_engine(
                    output_dir=self.output_dir, 
                )
                module.do_train()
            self.trainer = module.trainer
        else:
            if isinstance(module, nn.Module):
                self.trainer = LightningTrainer(module, *args, 
                    model_wrapper_config=self.model_wrapper_config,
                    trainer_config=self.trainer_config,
                    param_scheduler_configs=self.param_scheduler_configs,
                    checkpoint_config=self.checkpoint_config,
                    train_loader=self.dataloader_training,
                    val_loader=self.dataloader_validation,
                    test_loader=self.dataloader_testing,
                    checkpoint_dir=self.output_dir,
                    max_epochs=self.num_epoch,
                    check_val_every_n_epoch=self.check_val_every_n_epoch,
                    use_tflogger=use_tflogger,
                    tflogger_name=tflogger_name,
                    # NOTE: load_checkpoint and save_checkpoint are original params in BasicTrainer
                    # which affects self.load_checkpoint and self.save_checkpoint
                    # LightningTrainer enables load_checkpoint and save_checkpoint by default
                    # TODO: maybe add special options for LightningTrainer checkpointing?
                    # load_checkpoint=load_checkpoint,
                    # save_checkpoint=save_checkpoint,
                    # **kwargs
                )
            else:
                self.logger.warning("Module not trainable nn! Will skip training...")
                self.trainer = None
                # raise NotImplementedError("module not trainable nn!")


    def load_checkpoint(self, module: TrainableModuleInterface, *args, checkpoint_file="params.pkl", **kwargs) -> Any:
        # TODO: should we initialize here?
        # if self.trainer is None:
        #     self.initialize(module, *args, checkpoint_file=checkpoint_file, **kwargs)
        load_success = False
        if isinstance(module, nn.Module):
            # assure checkpoint to be device consistent
            module = module.to(device=self.device)
            checkpoint_file_path = os.path.join(self.output_dir, checkpoint_file)
            try:
                if os.path.exists(checkpoint_file_path):
                    self.logger.info("Loading checkpoint from {} ...".format(checkpoint_file_path))
                    state_dict = torch.load(checkpoint_file_path)
                    module.load_state_dict(state_dict)
                    load_success = True
                else:
                    self.logger.warning("Checkpoint {} not found!".format(checkpoint_file_path))
            except:
                self.logger.warning("Checkpoint {} loading failed!".format(checkpoint_file_path))
        return load_success
    
    def save_checkpoint(self, module: TrainableModuleInterface, *args, checkpoint_file="params.pkl", **kwargs) -> Any:
        # TODO: proper way to handle this through BasicNNTrainer?
        if isinstance(module, nn.Module):
            # assure checkpoint to be device consistent
            module = module.to(device=self.device)
            checkpoint_file_path = os.path.join(self.output_dir, checkpoint_file)
            self.logger.info("Saving checkpoint to {} ...".format(checkpoint_file_path))
            # NOTE: save write, avoid conflict in multiprocessing
            with self.open_file_safe(file_dir=checkpoint_file_path, mode="wb") as f:
                if f is None:
                    self.logger.warning(f"Safe open {checkpoint_file_path} failed! Check if lock is already acquired!")
                    return
                torch.save(module.state_dict(), f)
        # return super().save_checkpoint(module, *args, checkpoint_file=checkpoint_file, **kwargs)

    def train_module(self, module: TrainableModuleInterface, *args,
            initial_seed=None,
            **kwargs
         ):
        self.initialize(module, *args, 
            initial_seed=initial_seed, 
            tflogger_name="training_logs",
            tflogger_version=0, # disable auto versioning as versions are handled seperately
            **kwargs
        )

        if self.trainer is None: return

        # NOTE: Note that LightningTrainer will handle device properly during train/val/test! This will only affect the checkpointing and post_training_process. 
        # NOTE: This cause module storing on cuda:0 multiple times when using multiple GPUs!
        # if isinstance(module, nn.Module):
        #     module = module.to(device=self.device)

        if isinstance(self.trainer, BasicNNTrainer):
            self.trainer.do_train()
            if isinstance(self.trainer, LightningTrainer):
                if self.trainer.trainer.global_rank != 0:
                    return
            # self.trainer.deinitialize()
        else:
            self.logger.warning("NN Trainer is not properly setup! Skipping training...")


        # finally update model state for benchmark (moved to coding test phase)
        # module.update_state()
            
        # run post_training_process
        if isinstance(module, nn.Module):
            module = module.to(device=self.device)
        module.post_training_process()

        # save a final checkpoint (TODO: may conflict with BasicNNTrainer checkpointer implementation! Maybe add a choice param?)
        # NOTE: tmp disabled because of multiprocessing issue
        self.save_checkpoint(module, checkpoint_file=self._checkpoint_file)

        # super().train_module(module, *args,
        #     train_loader=self.dataloader,
        #     test_loader=self.dataloader_validation,
        #     # optimizer=optimizer,
        #     # scheduler=scheduler,
        #     **kwargs
        # )

    # def train_module_one_epoch(self, module: TrainableModuleInterface, *args, 
    #     optimizer=None,
    #     scheduler=None,
    #     **kwargs) -> None:
    #     # TODO:
    #     return super().train_module_one_epoch(module, *args, **kwargs)

    def test_module(self, module: TrainableModuleInterface, *args,
            initial_seed=None,
            **kwargs
        ):
        self.initialize(module, *args, 
            initial_seed=initial_seed, 
            # use_tflogger=False, # do not produce tf logs when testing
            tflogger_name="test_logs",
            tflogger_version=0, # disable auto versioning as versions are handled seperately
            **kwargs
        )

        if self.trainer is None: return

        # NOTE: Note that LightningTrainer will handle device properly during train/val/test! This will only affect the checkpointing and post_training_process. 
        # NOTE: This cause module storing on cuda:0 multiple times when using multiple GPUs!
        # if isinstance(module, nn.Module):
        #     module = module.to(device=self.device)

        # TODO: is it still needed as load_checkpoint is done?
        # NOTE: LightningTrainer use callbacks for checkpointing. We force it to load its checkpoint here!
        if isinstance(self.trainer, LightningTrainer):
            self.trainer._initialize(use_tflogger=False)
            self.trainer._load_weights()
            load_success = True
        else:
            # load the final checkpoint (TODO: may conflict with BasicNNTrainer checkpointer implementation! Maybe add a choice param?)
            load_success = self.load_checkpoint(module, checkpoint_file=self._checkpoint_file)

        if isinstance(self.trainer, BasicNNTrainer):
            results = self.trainer.do_test()
            if isinstance(self.trainer, LightningTrainer):
                if self.trainer.trainer.global_rank != 0:
                    return
            # self.trainer.deinitialize()
        else:
            self.logger.warning("NN Trainer is not properly setup! Skipping testing...")
            #
            results = dict()

        # TODO: (is this needed?) run post_training_process in case that training is skipped
        # if not load_success:
        #     if isinstance(module, nn.Module):
        #         module = module.to(device=self.device)
        #     module.post_training_process()
        #     self.save_checkpoint(module, checkpoint_file=self._checkpoint_file)

        return results

    def should_end_process(self):
        if isinstance(self.trainer, LightningTrainer):
            if self.trainer.trainer.global_rank != 0:
                return True
        return super().should_end_process()

class SimpleNNTrainer(BasicNNTrainer):
    def __init__(self, *args, 
        model: NNTrainableModule = None,
        optimizer_config=dict(),
        scheduler_config=dict(),
        max_epochs=100,
        device="cuda",
        training_logging_interval=100,
        training_log_cache=True,
        training_gradient_clip_val=None,
        validation_logging_interval=100,
        validation_interval_num_epoch=1,
        **kwargs):
        super().__init__(model, *args, **kwargs)
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.max_epochs = max_epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # configs
        self.training_logging_interval = training_logging_interval
        self.training_log_cache = training_log_cache
        self.training_gradient_clip_val = training_gradient_clip_val
        self.validation_logging_interval = validation_logging_interval
        self.validation_interval_num_epoch = validation_interval_num_epoch
        
        # placeholder
        self.tflogger = None

    def _log_metrics(self):
        metric_dict = self.model.get_metric_dict()
        metric_logs = []
        for k,v in metric_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            metric_logs.append(f"{k}: {v}")
        if len(metric_logs) > 0:
            self.logger.info("Metrics: " + ", ".join(metric_logs))
        self.model.reset_metric_dict()

    def _log_cache(self, prefix=None, log_nonscalar=False):
        if self.tflogger is None: return

        # log losses
        loss_dict = self.model.get_cache("loss_dict")
        for k, v in loss_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.tflogger.add_scalar(f"{prefix}/"+k, v, global_step=self.global_step)

        # log metrics if any
        metric_dict = self.model.get_cache("metric_dict")
        for k, v in metric_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.tflogger.add_scalar(f"{prefix}_metrics/"+k, v, global_step=self.global_step)

        # log moniters if any
        moniter_dict = self.model.get_cache("moniter_dict")
        for k, v in moniter_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.tflogger.add_scalar(f"{prefix}_moniters/"+k, v, global_step=self.global_step)

        if log_nonscalar:
            # log histograms
            hist_dict = self.model.get_cache("hist_dict")
            for k, v in hist_dict.items():
                self.tflogger.add_histogram(f"{prefix}_hist/"+k, v, global_step=self.global_step)
            # log images
            image_dict = self.model.get_cache("image_dict")
            for k, v in image_dict.items():
                # logger.add_images(f"{prefix}_image/"+k, v, global_step=global_step)
                if v.ndim == 4:
                    v = make_grid(v)
                self.tflogger.add_image(f"{prefix}_image/"+k, v, global_step=self.global_step)
            # log text
            text_dict = self.model.get_cache("text_dict")
            for k, v in text_dict.items():
                self.tflogger.add_text(f"{prefix}_text/"+k, v, global_step=self.global_step)
            # log figures
            figure_dict = self.model.get_cache("figure_dict")
            for k, v in figure_dict.items():
                self.tflogger.add_figure(f"{prefix}_figure/"+k, v, global_step=self.global_step)

    def _clip_grad(self, optimizer : torch.optim.Optimizer, gradient_clip_val = None):
        gradient_clip_info = {-1:[]}
        # for group_idx, gradient_clip_info in enumerate(self.gradient_clipping_table):
        group_gradient_clip_val = gradient_clip_val
        for param_group in optimizer.param_groups:
            gradient_clipping_group_idx = param_group.get("gradient_clipping_group", -1)
            if not gradient_clipping_group_idx in gradient_clip_info:
                gradient_clip_info[gradient_clipping_group_idx] = []
            gradient_clip_info[gradient_clipping_group_idx].extend(param_group['params'])
        
        for group_idx, param_list in gradient_clip_info.items():
            if group_gradient_clip_val is not None:
                nn.utils.clip_grad_norm_(param_list, group_gradient_clip_val)


    def _initialize(self, *args, 
                    use_tflogger=True, 
                    tflogger_name='training_logs',
                    **kwargs):
        optimizer = None
        if self.optimizer_config is not None:
            parameters = list(self.model.parameters())
            if len(parameters) > 0:
                optimizer = make_optimizer(self.model, logger=self.logger, **self.optimizer_config)
        self.optimizer = optimizer
        
        scheduler = None
        if self.scheduler_config is not None:
            if len(parameters) > 0:
                scheduler = make_scheduler(optimizer=optimizer, logger=self.logger, **self.scheduler_config)
        self.scheduler = scheduler
        
        if use_tflogger:
            self.tflogger = SummaryWriter(os.path.join(self.output_dir, tflogger_name))

        self.global_step = 0

    def _train(self, *args, **kwargs):
        self.model.to(device=self.device)
        interval_logger = SmoothedValue()
        
        for epoch_idx in range(self.max_epochs):
            self.model.train()
            last_time = time.time()
            for i, data in enumerate(self.train_loader):
                data = data.to(device=self.device)

                _ = self.model(data)

                loss_dict = self.model.get_loss_dict()
                if len(loss_dict) > 0:
                    loss = sum(loss_dict.values())

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.training_gradient_clip_val is not None:
                        self._clip_grad(self.optimizer, gradient_clip_val=self.training_gradient_clip_val)
                    self.optimizer.step()
                self.model.reset_loss_dict()


                self.global_step += 1

                if i % self.training_logging_interval == 0:
                    max_steps = self.max_epochs * len(self.train_loader)
                    eta_total_seconds = interval_logger.avg * (max_steps - self.global_step)
                    self.logger.info(f"Training Epoch [{epoch_idx}/{self.max_epochs}]: Step [{i}/{len(self.train_loader)}] Loss {loss.item():2f} ETA(h) {(eta_total_seconds/3600):2f}")

                    # self._log_metrics()
                    if self.training_log_cache:
                        self._log_cache(prefix="training")

                self.model.reset_all_cache()
                last_step_interval = time.time() - last_time
                interval_logger.update(last_step_interval)
                last_time = time.time()

            if epoch_idx % self.validation_interval_num_epoch == 0:
                self._validate()
                
            self.scheduler.step()

    def _validate(self, *args, **kwargs) -> Any:
        self.model.to(device=self.device)
        self.model.eval()

        loss_total = 0
        for i, data in enumerate(self.val_loader):
            data = data.to(device=self.device)
            with torch.no_grad():
                _, loss = self.model(data)
            loss_total += loss.item()

            # TODO: custom validation metrics?
            if i % 100 == 0:
                self.logger.info(f"Validation Step [{i}/{len(self.val_loader)}] Loss {loss.item():2f}")
                self._log_metrics()
    
        return loss_total / len(self.val_loader)

    def _test(self, *args, **kwargs):
        self.model.to(device=self.device)
        self.model.eval()

        loss_total = 0
        for i, data in enumerate(self.test_loader):
            data = data.to(device=self.device)
            with torch.no_grad():
                _, loss = self.model(data)
            loss_total += loss.item()

            # TODO: custom validation metrics?
            # TODO: tflogger?
            if i % self.validation_logging_interval == 0:
                self.logger.info(f"Test Step [{i}/{len(self.test_loader)}] Loss {loss.item():2f}")
                self._log_metrics()

        return loss_total / len(self.test_loader)


class CUDADeviceParallelNNTrainer(SimpleNNTrainer):
    def __init__(self, *args, device_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device_list = [i for i in range(torch.cuda.device_count())] if device_list is None else device_list
        assert len(self.device_list) > 0
    
    def set_model(self, model: nn.Module):
        # NOTE: model should be iterable
        assert len(model) > 0
        return super().set_model(model)
        
    def _initialize(self, *args, 
                    use_tflogger=True, 
                    tflogger_name='training_logs',
                    **kwargs):

        if len(self.model) % len(self.device_list) != 0:
            self.logger.warning(f"{len(self.device_list)} CUDA Devices are used for {len(self.model)} parallel modules which is not balanced!")

        # self.model_idx_device_mapping = []
        self.device_module_mapping = {device_idx : [] for device_idx in self.device_list}
        for idx, module in enumerate(self.model):
            # self.model_idx_device_mapping[idx] = self.device_list[idx % len(self.device_list)]
            self.device_module_mapping[self.device_list[idx % len(self.device_list)]].append(module)

        optimizer = None
        if self.optimizer_config is not None:
            parameters = list(self.model.parameters())
            if len(parameters) > 0:
                optimizer = make_optimizer(self.model, logger=self.logger, **self.optimizer_config)
        self.optimizer = optimizer
        
        scheduler = None
        if self.scheduler_config is not None:
            if len(parameters) > 0:
                scheduler = make_scheduler(optimizer=optimizer, logger=self.logger, **self.scheduler_config)
        self.scheduler = scheduler
        
        if use_tflogger:
            self.tflogger = SummaryWriter(os.path.join(self.output_dir, tflogger_name))

        self.global_step = 0
        
    def _train(self, *args, **kwargs):
        
        for epoch_idx in range(self.max_epochs):
            for device_idx, modules in self.device_module_mapping.items():
                for module in modules:
                    module.to(device=f"cuda:{device_idx}")
            self.model.train()

            last_time = time.time()
            interval_logger = SmoothedValue()
            for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                self.optimizer.zero_grad()
                for device_idx, modules in self.device_module_mapping.items():
                    data = data.to(device=f"cuda:{device_idx}")
                    for module in modules:
                        _ = module(data)

                        loss_dict = module.get_loss_dict()
                        if len(loss_dict) > 0:
                            loss = sum(loss_dict.values())
                            loss.backward()
                        module.reset_loss_dict()
                
                if self.training_gradient_clip_val is not None:
                    self._clip_grad(self.optimizer, gradient_clip_val=self.training_gradient_clip_val)
                self.optimizer.step()

                self.global_step += 1

                if i % self.training_logging_interval == 0:
                    max_steps = self.max_epochs * len(self.train_loader)
                    eta_total_seconds = interval_logger.avg * (max_steps - self.global_step)
                    self.logger.info(f"Training Epoch [{epoch_idx}/{self.max_epochs}]: Step [{i}/{len(self.train_loader)}] Loss {loss.item():2f} ETA(h) {(eta_total_seconds/3600):2f}")
                    
                    # self._log_metrics()
                    if self.training_log_cache:
                        self._log_cache(prefix="training")

                self.model.reset_all_cache()

                last_step_interval = time.time() - last_time
                interval_logger.update(last_step_interval)
                last_time = time.time()

            if epoch_idx % self.validation_interval_num_epoch == 0:
                self._validate()
                
            self.scheduler.step()

class ParamSchedulerLightningCallback(pl.Callback):
    def __init__(self, param_scheduler_configs : List[Tuple[List[str], Dict[str, Any]]] = [], step_policy="epoch") -> None:
        self.param_scheduler_configs = param_scheduler_configs
        self.step_policy = step_policy
        assert step_policy in ["epoch", "batch"]

        self.param_schedulers = []
        self.is_param_schedulers_setup = False

        self.pl_logger = setup_logger("pytorch_lightning")

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self.is_param_schedulers_setup: return

        for param_names, config in self.param_scheduler_configs:
            if not isinstance(param_names, Sequence):
                param_names = [param_names]
            param_scheduler_config = dict(scheduler_config=dict(), scheduler_extra_config=dict()) # default
            param_scheduler_config.update(config)
            dummy_optimizer, param_scheduler = make_dummy_optimizer_with_param_scheduler(pl_module, param_names, logger=self.pl_logger, **param_scheduler_config)
            if param_scheduler is not None:
                self.param_schedulers.append(param_scheduler)

        self.is_param_schedulers_setup = True

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.step_policy == "epoch":
            for scheduler in self.param_schedulers:
                scheduler.step()

    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.step_policy == "batch":
            for scheduler in self.param_schedulers:
                scheduler.step()

    # TODO: override load_state_dict for pytorch_lightning >= 1.8
    # def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]) -> None:
    #     if 'param_schedulers' in callback_state:
    #         param_schedulers_states = callback_state.pop('param_schedulers')
    #         for ps, ps_state in zip(self.param_schedulers, param_schedulers_states):
    #             ps.load_state_dict(ps_state)
    #     # return super().on_load_checkpoint(trainer, pl_module, callback_state)

    # TODO: override state_dict for pytorch_lightning >= 1.8
    # def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> dict:
    #     # checkpoint = super().on_save_checkpoint(trainer, pl_module, checkpoint)
    #     callback_state = dict()
    #     if self.is_param_schedulers_setup:
    #         callback_state['param_schedulers'] = [ps.state_dict() for ps in self.param_schedulers]
    #     return callback_state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if 'param_schedulers' in state_dict:
            param_schedulers_states = state_dict.pop('param_schedulers')
            for ps, ps_state in zip(self.param_schedulers, param_schedulers_states):
                ps.load_state_dict(ps_state)
        return super().load_state_dict(state_dict)

    def state_dict(self) -> Dict[str, Any]:
        callback_state = dict()
        if self.is_param_schedulers_setup:
            callback_state['param_schedulers'] = [ps.state_dict() for ps in self.param_schedulers]
        return callback_state

class _LightningBasicModelWrapper(pl.LightningModule):
    '''
    Wrapper for nn.Module with pl.LightningModule
    Args:
        model (nn.Module): torch nn.Module to be trained
        optimizer_type (string): refer to make_optimizer
        optimizer_config (dict): parameters for make_optimizer
        scheduler_type (string): refer to make_scheduler
        scheduler_config (dict): parameters for make_scheduler
        scheduler_extra_config (dict): extra parameters for LightningModule configure_optimizers
        metric_weight_table (dict): weight table for monitoring validation metric.
        loss_weight_table (dict): weight table for loss calculation.
        loss_aux_table (dict): Auxiliary optimizer index to run loss backward process.
    '''
    def __init__(self, model : NNTrainableModule, 
        optimizer_type="Adam",
        optimizer_config : Dict[str, Any] = dict(),
        scheduler_type="None",
        scheduler_config : Dict[str, Any] = dict(),
        scheduler_extra_config : Dict[str, Any] = {
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
            'monitor': 'val_metric', # Metric for ReduceLROnPlateau to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for LearningRateMonitor to use)
        },
        multiopt_configs : List[Dict[str, Any]] = [],
        use_double_precision=False,
        # param_scheduler_configs : List[Tuple[List[str], Dict[str, Any]]] = [],
        metric_weight_table=dict(val_metric=1.0),
        loss_weight_table=dict(),
        loss_aux_table=dict(),
        gradient_clipping_table=dict(),
        logger=None,
        training_log_cache=True,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config
        self.scheduler_type = scheduler_type
        self.scheduler_config = scheduler_config
        self.scheduler_extra_config = scheduler_extra_config
        self.multiopt_configs = multiopt_configs
        self.use_double_precision = use_double_precision
        # self.param_scheduler_configs = param_scheduler_configs
        # self.param_scheduler_configs = [
        #     ("prior_model.prior_coder.gs_temp", dict(
        #         scheduler_type="ExponentialLR",
        #         scheduler_config=dict(gamma=1e-1),
        #     )),
        # ]
        # self.param_schedulers = []
        # self.is_param_schedulers_setup = False

        self.metric_weight_table = metric_weight_table
        self.loss_weight_table = loss_weight_table
        self.loss_aux_table = loss_aux_table
        self.gradient_clipping_table = gradient_clipping_table

        self.last_time = time.time()
        self.interval_logger = SmoothedValue()

        # self.pl_logger = logging.getLogger("pytorch_lightning")
        # self.pl_logger.setLevel(logging.INFO)
        self.pl_logger = setup_logger("pytorch_lightning") if logger is None else logger
        
        self.training_log_cache = training_log_cache

    def forward(self, x):
        x = self.model(x)
        return x

    # def on_train_epoch_end(self) -> None:
    #     for scheduler in self.param_schedulers:
    #         scheduler.step()

    # def setup(self, stage: Optional[str] = None) -> None:
    #     if self.is_param_schedulers_setup: return

    #     for param_names, config in self.param_scheduler_configs:
    #         if not isinstance(param_names, Sequence):
    #             param_names = [param_names]
    #         param_scheduler_config = dict(scheduler_config=dict(), scheduler_extra_config=dict()) # default
    #         param_scheduler_config.update(config)
    #         dummy_optimizer, param_scheduler = make_dummy_optimizer_with_param_scheduler(self.model, param_names, logger=self.pl_logger, **param_scheduler_config)
    #         if param_scheduler is not None:
    #             self.param_schedulers.append(param_scheduler)

    #     self.is_param_schedulers_setup = True

    # def _save_to_state_dict(self, destination, prefix, keep_vars):
    #     super()._save_to_state_dict(destination, prefix, keep_vars)
    #     if self.is_param_schedulers_setup:
    #         destination['param_schedulers'] = [ps.state_dict() for ps in self.param_schedulers]

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #     if 'param_schedulers' in state_dict:
    #         param_schedulers_states = state_dict.pop('param_schedulers')
    #         for ps, ps_state in zip(self.param_schedulers, param_schedulers_states):
    #             ps.load_state_dict(ps_state)
    #             # NOTE: it seems that the checkpoint is saved before calling ps.step()
    #             # so we need to compensate here
    #             ps.step()
    #     super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


    def _log_cache(self, prefix=None, log_nonscalar=False, is_training=True):
        # log losses
        loss_dict = self.model.get_cache("loss_dict")
        for k, v in loss_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.log(f"{prefix}/"+k, v, sync_dist=(not is_training)) # sync across devices for epoch level logging (during validation)
        # self.log_dict(ret)
        self.model.reset_loss_dict()

        # log metrics if any
        metric_dict = self.model.get_cache("metric_dict")
        for k, v in metric_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.log(f"{prefix}_metrics/"+k, v, sync_dist=(not is_training)) # sync across devices for epoch level logging (during validation)
        self.model.reset_metric_dict()

        # log moniters if any
        moniter_dict = self.model.get_cache("moniter_dict")
        for k, v in moniter_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.log(f"{prefix}_moniters/"+k, v, sync_dist=(not is_training)) # sync across devices for epoch level logging (during validation)
        self.model.reset_cache("moniter_dict")

        # TODO: this seems to cause error on newer version of pl/tensorboard logger (reported pl 1.7.7)
        logger = self.logger.experiment
        if log_nonscalar and isinstance(logger, SummaryWriter):
            if self.trainer:
                # use logging interval during training (and log every batch for validation/testing)
                if is_training and self.global_step % self.trainer.log_every_n_steps != 0:
                    pass
                else:
                    # log histograms
                    hist_dict = self.model.get_cache("hist_dict")
                    for k, v in hist_dict.items():
                        logger.add_histogram(f"{prefix}_hist/"+k, v, global_step=self.global_step)
                    # log images
                    image_dict = self.model.get_cache("image_dict")
                    for k, v in image_dict.items():
                        # logger.add_images(f"{prefix}_image/"+k, v, global_step=global_step)
                        if v.ndim == 4:
                            v = make_grid(v)
                        logger.add_image(f"{prefix}_image/"+k, v, global_step=self.global_step)
                    # log text
                    text_dict = self.model.get_cache("text_dict")
                    for k, v in text_dict.items():
                        logger.add_text(f"{prefix}_text/"+k, v, global_step=self.global_step)
                    # log figures
                    figure_dict = self.model.get_cache("figure_dict")
                    for k, v in figure_dict.items():
                        logger.add_figure(f"{prefix}_figure/"+k, v, global_step=self.global_step)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # TODO: this could be migrated into progress bar callback
        # total eta
        if self.trainer is not None:
            last_step_interval = time.time() - self.last_time
            self.interval_logger.update(last_step_interval)
            max_steps = self.trainer.max_epochs * len(self.trainer.train_dataloader) if self.trainer.max_steps < 0 else self.trainer.max_steps
            eta_total_seconds = self.interval_logger.avg * (max_steps - self.global_step)
            # eta_total_string = str(datetime.timedelta(seconds=int(eta_total)))
            self.log("eta(h)", eta_total_seconds / 3600, prog_bar=True, logger=False)
        self.last_time = time.time()

        ret = dict()

        self.model.set_optim_state(optimizer_idx)
        
        # forward + backward + optimize
        # inputs, labels = batch
        if isinstance(batch, dict):
            outputs = self.model(**batch)
        elif isinstance(batch, (list, tuple)):
            outputs = self.model(*batch)
        else:
            if self.use_double_precision:
                self.model.double()
                batch = batch.double()
            outputs = self.model(batch)

        # log bottleneck 
        # if self.global_step % 1000 == 0 and "bottleneck_prior" in outputs:
        #     tensorboard = self.logger.experiment
        #     bottleneck_prior = outputs["bottleneck_prior"]
        #     tensorboard.add_histogram("bottleneck_hist", bottleneck_prior, global_step=self.global_step)

        # outputs = self.model(inputs, labels=labels)
        loss_dict = self.model.get_loss_dict() # criterion(outputs, labels)
        # ret.update(**loss_dict)
        loss_dict_weighted = {
            k : (v * (self.loss_weight_table[k] if k in self.loss_weight_table else 1.0))
            for k, v in loss_dict.items()
        }
        losses = [v for k, v in loss_dict_weighted.items()]

        if optimizer_idx == 0:
            # main losses
            losses = [v for k, v in loss_dict_weighted.items() if (not k in self.loss_aux_table or self.loss_aux_table[k]<0)]
            # losses = list(loss_dict_weighted.values())
        else:
            # aux losses
            losses = [v for k, v in loss_dict_weighted.items() if (k in self.loss_aux_table and self.loss_aux_table[k]==optimizer_idx-1)]
        
        # avoid back propagating float 0 when no loss exists
        loss = torch.scalar_tensor(0, device=self.device, requires_grad=True)
        loss = loss + sum(losses)
        # training metrics if any
        # metric_dict = self.model.get_metric_dict()
        # ret.update(**metric_dict)

        # log losses
        self.log("training/loss_opt{}".format(optimizer_idx), loss)
        self.log("loss", loss)

        # for k, v in loss_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v.item()
        #     self.log("training/"+k, v)
        # # self.log_dict(ret)
        # self.model.reset_loss_dict()

        # # log metrics if any
        # metric_dict = self.model.get_metric_dict()
        # for k, v in metric_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v.item()
        #     self.log("training_metrics/"+k, v)
        # self.model.reset_metric_dict()

        if self.training_log_cache:
            self._log_cache(prefix="training")
        self.model.reset_all_cache()

        # if self.global_step % 100 == 0:
        #     self.pl_logger.info("Iter {} Loss {:.4f}".format(self.global_step, loss))

        # output final loss
        ret.update(loss=loss)
        return ret

    # def on_epoch_start(self) -> None:
    #     return super().on_epoch_start()

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        
        if self.optimizer_type == "None":
            optimizer = None
        else:
            optimizer = make_optimizer(self.model, self.optimizer_type, logger=self.pl_logger, **self.optimizer_config) # , logger=self.logger
        
        if optimizer is None:
            return None
        
        scheduler = make_scheduler(optimizer, self.scheduler_type, logger=self.pl_logger, **self.scheduler_config)
        opt_config = dict(optimizer=optimizer, frequency=None) # TODO: frequency
        if not scheduler is None:
            opt_config.update(lr_scheduler=dict(scheduler=scheduler, **self.scheduler_extra_config))
        optimizers.append(opt_config)
        # optimizers.append(optimizer)
        # schedulers.append(dict(scheduler=scheduler, **self.scheduler_extra_config))

        # aux optimizers
        for aux_id, config in enumerate(self.multiopt_configs):
            aux_config = dict(optimizer_config=dict(), scheduler_config=dict(), scheduler_extra_config=dict()) # default
            aux_config.update(config)
            optimizer = make_optimizer(self.model, logger=self.pl_logger, aux_id=aux_id, **aux_config["optimizer_config"])
            if optimizer is None:
                self.pl_logger.warning(f"No params defined for aux optimizer {aux_id}. Skipping this optimizer!")
                continue
            scheduler = make_scheduler(optimizer, logger=self.pl_logger, **aux_config["scheduler_config"])
            aux_opt_config = dict(optimizer=optimizer, frequency=None) # TODO: frequency
            if scheduler is not None:
                aux_opt_config.update(lr_scheduler=dict(scheduler=scheduler, **aux_config["scheduler_extra_config"]))
            optimizers.append(aux_opt_config)
            # optimizers.append(optimizer)
            # schedulers.append(dict(scheduler=scheduler, **aux_config["scheduler_extra_config"]))

        # param schedulers
        # for param_names, config in self.param_scheduler_configs:
        #     if not isinstance(param_names, Sequence):
        #         param_names = [param_names]
        #     param_scheduler_config = dict(scheduler_config=dict(), scheduler_extra_config=dict()) # default
        #     param_scheduler_config.update(config)
        #     dummy_optimizer, param_scheduler = make_dummy_optimizer_with_param_scheduler(self.model, param_names, logger=self.pl_logger, **param_scheduler_config)
        #     if not dummy_optimizer is None:
        #         param_scheduler = dict(scheduler=param_scheduler, **param_scheduler_config['scheduler_extra_config'])
        #         param_scheduler_opt_config = dict(optimizer=dummy_optimizer, lr_scheduler=param_scheduler)
        #         optimizers.append(param_scheduler_opt_config)
        
        return optimizers #, schedulers


    def configure_gradient_clipping(self, optimizer: torch.optim.Optimizer, optimizer_idx: int, gradient_clip_val = None, gradient_clip_algorithm = None):
        # return super().configure_gradient_clipping(optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm)
        # TODO: gradient_clip_algorithm?
        gradient_clip_info = {-1:[]}
        # for group_idx, gradient_clip_info in enumerate(self.gradient_clipping_table):
        group_gradient_clip_val = gradient_clip_val
        for param_group in optimizer.param_groups:
            gradient_clipping_group_idx = param_group.get("gradient_clipping_group", -1)
            if not gradient_clipping_group_idx in gradient_clip_info:
                gradient_clip_info[gradient_clipping_group_idx] = []
            gradient_clip_info[gradient_clipping_group_idx].extend(param_group['params'])
        
        for group_idx, param_list in gradient_clip_info.items():
            group_gradient_clip_val = self.gradient_clipping_table.get(group_idx, group_gradient_clip_val)
            if group_gradient_clip_val is not None:
                nn.utils.clip_grad_norm_(param_list, group_gradient_clip_val)

    def validation_step(self, batch, batch_idx):
        # state_dict = torch.load("experiments/aii_kube/vq-svq-comp/VQ/epoch=772-step=151507.ckpt", map_location=torch.device('cpu'))["state_dict"]
        # state_dict = {k.strip("model."):v for k,v in state_dict.items()}
        # state_dict["prior_model.prior_coder.gs_temp"] = torch.tensor([0.01])
        # state_dict["prior_model.prior_coder.embedding_variance"] = torch.tensor([0.4]).log()
        # self.model.load_state_dict(state_dict, strict=False)
        metric_table = dict()

        # inputs, labels = batch
        if isinstance(batch, dict):
            outputs = self.model(**batch)
        elif isinstance(batch, (list, tuple)):
            outputs = self.model(*batch)
        else:
            if self.use_double_precision:
                self.model.double()
                batch = batch.double()
            outputs = self.model(batch)
        # outputs = self.model(inputs, labels=labels)

        metric_dict = self.model.get_metric_dict()
        # for k, v in metric_dict.items():
        #     if not k in metric_table:
        #         metric_table[k] = []
        #     # this is required for proper GC
        #     if isinstance(v, torch.Tensor):
        #         v = v.item()
        #     metric_table[k].append(v)
        
        metric_value = 0. # sum(metric_values)

        for k, v in metric_dict.items():
            if k in self.metric_weight_table:
                metric_value += v.item() if isinstance(v, torch.Tensor) else v
        # self.log_dict(metric_table)
        self.log("val_metric", metric_value, sync_dist=True)

        # NOTE: to enable image and hist logging, enable log_nonscalar!
        self._log_cache(prefix="validation", log_nonscalar=True, is_training=False)
        self.model.reset_all_cache()

        return metric_value
    
    # TODO: postprocess metrics
    # def validation_epoch_end(self, outputs: List[torch.Tensor | Dict[str, Any]]) -> None:
    #     return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # deprecated in pl v1.5
    # def get_progress_bar_dict(self):
    #     # don't show the version number
    #     items = super().get_progress_bar_dict()
    #     items.pop("v_num", None)
    #     return items

    # def load_state_dict(self, *args, **kwargs):
    #     super().load_state_dict(*args, **kwargs)


class LightningTrainer(BasicNNTrainer):
    '''
    Main trainer with pytorch_lightning
    Args:
        model (nn.Module): torch.nn.Module or pl.LightningModule to be trained. 
            If a torch.nn.Module is used, _LightningBasicModelWrapper will be initialized using model_wrapper_config.
        model_wrapper_config (dict): parameters for _LightningBasicModelWrapper
        trainer_config (dict): parameters for :meth:`pytorch_lightning.Trainer`
        checkpoint_config (dict): parameters for :meth:`pytorch_lightning.callbacks.ModelCheckpoint`
    '''
    def __init__(self, 
        model : Union[pl.LightningModule, nn.Module] = None,
        model_wrapper_config : Dict[str, Any] = dict(),
        param_scheduler_configs : List[Tuple[List[str], Dict[str, Any]]] = [],
        trainer_config : Dict[str, Any] = dict(),
        checkpoint_config : Dict[str, Any] = dict(), # = inspect_default_value_dict(pl.Trainer),
        extra_callback_configs=[],
        max_epochs=100,
        check_val_every_n_epoch=1,
        training_log_cache=True,
        **kwargs
    ):
        # if not isinstance(model, pl.LightningModule):
        #     model = _LightningBasicModelWrapper(model, **model_wrapper_config)
        self.model_wrapper_config = model_wrapper_config
        self.param_scheduler_configs = param_scheduler_configs
        self.trainer_config = trainer_config
        self.checkpoint_config = checkpoint_config

        self.extra_callback_configs = extra_callback_configs
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.training_log_cache = training_log_cache
        # self.set_model(model) # set model in super function
        super().__init__(model=model,
            **kwargs
        )

    def set_model(self, model: nn.Module):
        if not isinstance(model, pl.LightningModule):
            model_wrapper_config = dict()
            model_wrapper_config.update(**self.model_wrapper_config)
            # load pretrained checkpoint (NOTE: moved to _initialize to avoid conflict with SelfTrainableInterface)
            # if isinstance(model, NNTrainableModule):
            #     # TODO: add pretrained kwargs in trainer
            #     model.load_checkpoint()
            model = _LightningBasicModelWrapper(model, logger=self.logger,
                training_log_cache=self.training_log_cache,
                **model_wrapper_config
            )
        return super().set_model(model)

    def _initialize(self, *args, 
        use_tflogger=True, 
        tflogger_name='training_logs',
        tflogger_version=None,
        # use_latest=True, 
        # manual_load_weights=False, 
        # resume_from_checkpoint=None, 
        load_checkpoint=True,
        save_checkpoint=True,
        load_best_checkpoint=False,
        **kwargs):

        if use_tflogger:
            logger = pl.loggers.TensorBoardLogger(
                save_dir=self.checkpoint_dir,
                name=tflogger_name,
                version=tflogger_version,
            )
        else:
            logger = None

        # default to save the last checkpoint
        checkpoint_config = dict(
            save_last=True
        )
        checkpoint_config.update(self.checkpoint_config)
        
        # callbacks
        callbacks = [
            pl.callbacks.LearningRateMonitor(), # TODO: options
            pl.callbacks.TQDMProgressBar(), # TODO: options
            pl.callbacks.EarlyStopping("loss", patience=self.max_epochs, check_finite=True), # TODO: options
            ParamSchedulerLightningCallback(self.param_scheduler_configs),
        ]
        if save_checkpoint:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    **checkpoint_config
                )
            )
        
        # TODO: "last.ckpt" is defined in pl. Maybe could be customized?
        if load_checkpoint:
            resume_from_checkpoint = os.path.join(self.checkpoint_dir, "last.ckpt")
            if not os.path.exists(resume_from_checkpoint):
                self.logger.warning("Checkpoint {} not found! Training from beginning...".format(resume_from_checkpoint))
                resume_from_checkpoint = None
        else:
            resume_from_checkpoint = None

        # if resume_from_checkpoint is not None:
        #     self.model.load_from_checkpoint(resume_from_checkpoint)

        # TODO: extra_callback_configs
        trainer_options = dict(
            max_epochs=self.max_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )
        trainer_options.update(self.trainer_config)
        # fix for cross platform accelerator
        if trainer_options.get('accelerator') == "gpu" and not torch.cuda.is_available():
            self.logger.warning("GPU is not available on this machine! Switching to auto accelerator!")
            trainer_options['accelerator'] = 'auto'
            trainer_options['devices'] = None
        
        # from  pytorch_lightning.profiler import PyTorchProfiler
        self.trainer = pl.Trainer(
            # *args, 
            logger=logger, 
            default_root_dir=self.checkpoint_dir, 
            # resume_from_checkpoint=resume_from_checkpoint, # deprecated
            callbacks=callbacks, 
            # profiler=PyTorchProfiler(
            #     os.path.join(self.checkpoint_dir, "pytorch_profiler"),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.checkpoint_dir, "pytorch_profiler"))
            # ),
            **trainer_options
        )
        self.resume_from_checkpoint = resume_from_checkpoint
        self.load_best_checkpoint = load_best_checkpoint

        # if manual_load_weights:
        #     self._load_weights()

        # TODO: find a better place to call NNTrainableModule.load_checkpoint 
        # maybe after model is initialized but before training?)
        if isinstance(self.model, _LightningBasicModelWrapper):
            if isinstance(self.model.model, NNTrainableModule):
                self.model.model.load_checkpoint()

    def _load_weights(self):
        checkpoint_path = self.resume_from_checkpoint # os.path.join(self.checkpoint_dir, "last.ckpt")
        if not checkpoint_path is None and os.path.exists(checkpoint_path):
            # default to load checkpoint on CPU
            pl_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            if self.load_best_checkpoint:
                for cb, info in pl_checkpoint['callbacks'].items():
                    if cb == pl.callbacks.ModelCheckpoint:
                        best_checkpoint_path = info['best_model_path']
                        pl_checkpoint = torch.load(best_checkpoint_path)
                        self.logger.info("Best checkpoint {} found and loaded!".format(best_checkpoint_path))
            state_dict = pl_checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
    
    def _train(self, *args, **kwargs):
        if not self.train_loader is None:
            # NOTE: this is an extra check of the model if training is needed!
            # Could be better to check this in trainer.fit(**) process
            if self.model.configure_optimizers() is not None:
                self.trainer.fit(self.model, self.train_loader, self.val_loader,
                    ckpt_path=self.resume_from_checkpoint, # pl >= v1.5 compability, load all trainer state
                )
            # if self.trainer.global_rank != 0:
            #     return

    def _validate(self, *args, **kwargs) -> Any:
        if not self.val_loader is None:
            results = self.trainer.validate(self.model, self.val_loader,
                ckpt_path=self.resume_from_checkpoint, # pl >= v1.5 compability, load all trainer state
            )
            # only return main process
            if self.trainer.global_rank != 0:
                return
            if results is not None:
                # TODO: reduce result dict!
                return results[0] 
                # TODO: there is only 1 dataloader, so we could use index 0 
  
    def _test(self, *args, **kwargs):
        if not self.test_loader is None:
            results = self.trainer.test(self.model, self.test_loader, 
                ckpt_path=self.resume_from_checkpoint, # pl >= v1.5 compability, load all trainer state
            )
            # only return main process
            if self.trainer.global_rank != 0:
                return
            if results is not None:
                return results[0] # there is only 1 dataloader, so we could use index 0 
    
    # NOTE: we still lacks a way to end the DDP process, and they will try to exec out code (such as writing metric files)!
    # def _deinitialize(self, *args, **kwargs):
        # self.trainer.should_stop = True
        # if self.trainer.global_rank != 0:
        #     # TODO: can we get a soft exit?
        #     exit(0)
        # self.trainer.accelerator.training_type_plugin.barrier()
        # self.trainer.accelerator.training_type_plugin.teardown()
        # delattr(self, "trainer")