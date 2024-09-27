import csv
import pickle
import random
import time
import os
import json
import hashlib
import logging
import multiprocessing
import traceback
from tqdm import tqdm
from typing import Iterable, Sequence, Union, List, Tuple, Callable, Any, Optional
import numpy as np

# TODO: fix distributed overrun during testing to remove dependence on torch
import torch.distributed

from cbench.modules.base import TrainableModuleInterface

from .base import BaseBenchmark, BaseEngine
from .trainer import BasicTrainer
from .metrics.base import BaseMetric
from .metrics.bj_delta import BJDeltaMetric

from cbench.codecs.base import BaseCodec, CodecInterface, NNTrainableCodec, VariableRateCodecInterface, VariableComplexityCodecInterface, VariableTaskCodecInterface
from cbench.data.dataloaders.basic import DataLoaderInterface
from cbench.utils.logging_utils import MetricLogger, SmoothedValue
from cbench.nn.base import NNTrainableModule, SelfTrainableInterface

class BenchmarkTestingWorker(BaseEngine):
    def __init__(self, 
                 codec: Optional[CodecInterface] = None, 
                 dataloader: Optional[DataLoaderInterface] = None, 
                #  metric_logger=None,
                 distortion_metric : Optional[BaseMetric] = None,
                 data_input_key=None,
                 data_target_key=None,
                 variable_rate_level=None,
                 variable_complex_level=None,
                 variable_task=None,
                 ignore_invalid_variable_task=False,
                 cache_dir=None,
                 cache_compressed_data=False,
                 cache_checksum=True,
                 skip_decompress=False,
                 save_decompressed_data=False,
                 save_dir=None,
                 save_format="image",
                 save_metric_in_filename=True,
                 nn_codec_use_forward_pass=False,
                 nn_codec_forward_pass_skip_compression=True,
                 **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self._codec = codec
        self._dataloader = dataloader
        self._distortion_metric = distortion_metric

        self.data_input_key = data_input_key
        self.data_target_key = data_target_key
        self.variable_rate_level = variable_rate_level
        self.variable_complex_level = variable_complex_level
        self.variable_task = variable_task
        self.ignore_invalid_variable_task = ignore_invalid_variable_task

        # self.metric_logger = metric_logger
        self.cache_dir = cache_dir
        self.cache_compressed_data = cache_compressed_data
        self.cache_checksum = cache_checksum
        self.skip_decompress = skip_decompress

        self.save_decompressed_data = save_decompressed_data
        self.save_dir = save_dir
        self.save_format = save_format
        self.save_metric_in_filename = save_metric_in_filename
        
        # nn codec benchmark
        self.nn_codec_use_forward_pass = nn_codec_use_forward_pass
        self.nn_codec_forward_pass_skip_compression = nn_codec_forward_pass_skip_compression

    @property
    def codec(self):
        return self._codec

    @codec.setter
    def codec(self, codec):
        self._codec = codec

    @property
    def dataloader(self):
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        self._dataloader = dataloader

    @property
    def distortion_metric(self):
        return self._distortion_metric

    @distortion_metric.setter
    def distortion_metric(self, distortion_metric):
        self._distortion_metric = distortion_metric

    def _estimate_byte_length(self, data):
        import torch
        # a lazy solution for evaluating total bytes...
        # pickle.dumps add some useless information such as class definitions!
        # return len(pickle.dumps(data))
        if isinstance(data, bytes):
            return len(data) # TODO: length of string?
        elif isinstance(data, str):
            return len(data.encode('utf-8')) # TODO: length of string?
        elif isinstance(data, torch.Tensor):
            return int(np.prod(data.shape)) * data.element_size()
        elif isinstance(data, np.ndarray):
            return int(data.size * data.itemsize)
        else:
            raise ValueError("Bitstream of data {} in type {} cannot be estimated!".format(data, type(data)))

    def _run_step(self, step, data, metric_logger, save_prefix):
        # Transform coding with different target
        # if isinstance(data, (list, tuple)) and len(data) == 2:
        #     data, target = data
        # else:
        #     target = None
        
        data_input, data_target = data, data
        if self.data_input_key is not None:
            if isinstance(self.data_input_key, (list, tuple)):
                data_input = [data[key] for key in self.data_input_key]
            else:
                data_input = data[self.data_input_key]

        if self.data_target_key is not None:
            if isinstance(self.data_target_key, (list, tuple)):
                data_target = [data[key] for key in self.data_target_key]
            else:
                data_target = data[self.data_target_key]

        # TODO: the data format should be defined!
        try:
            original_length = self._estimate_byte_length(data_input) 
        except ValueError:
            self.logger.warning("Cannot estimate original data length! Using pickle as an alt way.")
            original_string = pickle.dumps(data_input)
            original_length = len(original_string)
            # original_bits = original_length * 8

        metric_logger.update(
            original_length=original_length,
        )

        # distortion
        distortion_metric = self.distortion_metric

        if isinstance(self.codec, NNTrainableCodec) and self.nn_codec_use_forward_pass:
            
            decompressed_data, compressed_length = self.codec.forward_estimate_bitlen(data_input)
            self.codec.reset_all_cache()

            compression_ratio = compressed_length / original_length
            metric_logger.update(
                compression_ratio_nn_forward=compression_ratio,
                compressed_length_nn_forward=compressed_length,
            )

            # check decompress correctness for lossless
            # assert(data == decompressed_data)

            if distortion_metric is not None:
                # NOTE: results will be collected after all steps are done!
                results = distortion_metric(decompressed_data, data_target)

            if self.nn_codec_forward_pass_skip_compression:
                return
            else:
                if distortion_metric is not None:
                    self.logger.warning("Using both nn forward pass and normal comp/decomp pass may got mixed distortion_metric!")

        # search cache
        compressed_data = None
        if self.cache_dir:
            if self.cache_compressed_data:
                # NOTE: maybe using hash values and filenames? dataloader may be shuffled!
                cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        compressed_data = pickle.load(f)
                        if self.cache_checksum:
                            compressed_data, checksum = compressed_data
                            original_data_checksum = hashlib.md5(data_input).digest()
                            # checksum fail! rerun codec!
                            if checksum != original_data_checksum:
                                self.logger.warning(f"Checksum fails for data iteration {step}. Check if the dataloader is deterministic!")
                                compressed_data = None

        # run codec compress
        if compressed_data is None:
            time_start = time.time()
            compressed_data = self.codec.compress(data_input)
            time_compress = time.time() - time_start

        try:
            compressed_length = self._estimate_byte_length(compressed_data) 
        except ValueError:
            self.logger.warning("Cannot estimate compressed data length! Using pickle as an alt way.")
            compressed_string = pickle.dumps(compressed_data)
            compressed_length = len(compressed_string)
        # compressed_bits = compressed_length * 8

        compression_ratio = compressed_length / original_length

        metric_logger.update(
            compression_ratio=compression_ratio,
            compressed_length=compressed_length,
            time_compress=time_compress*1000,
            speed_compress=(original_length/time_compress/1024/1024),
        )

        if not self.skip_decompress:
            time_start = time.time()
            decompressed_data = self.codec.decompress(compressed_data)
            time_decompress = time.time() - time_start

            # NOTE: use original_length or compressed_length when calculating speed_decompress?
            metric_logger.update(
                time_decompress=time_decompress*1000,
                speed_decompress=(original_length/time_decompress/1024/1024),
                time_total=(time_compress+time_decompress)*1000,
                speed_total=(original_length/(time_compress+time_decompress)/1024/1024),
            )
            # check decompress correctness for lossless
            # assert(data == decompressed_data)

            # distortion
            metric_string = None
            if distortion_metric is not None:
                results = distortion_metric(decompressed_data, data_target)
                # NOTE: results will be collected after all steps are done!
                if results is not None:
                    # self.metric_logger.update(**results)
                    if self.save_decompressed_data and self.save_metric_in_filename:
                        metric = results.get(distortion_metric.name, None)
                        if metric is not None:
                            metric_string = f"{distortion_metric.name}={metric:.2f}"
                
            # save result
            if self.save_decompressed_data: # and os.path.exists(self.save_dir):
                save_filename = f"{save_prefix}{step}"
                if metric_string is not None:
                    save_filename += f"_{metric_string}"
                # TODO: move saving code to sublib to avoid force dependency!
                if self.save_format == "image":
                    from torchvision.utils import save_image
                    save_image(decompressed_data, os.path.join(self.save_dir, "{}.png".format(save_filename)))

        # cache compressed data
        # TODO: update to support variable rate/complex/task benchmarks
        if self.cache_dir:
            if self.cache_compressed_data:
                cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
                if not os.path.exists(cache_file):
                    with open(cache_file, 'wb') as f:
                        if self.cache_checksum:
                            original_data_checksum = hashlib.md5(data).digest()
                            cached_data = (compressed_data, original_data_checksum)
                        else:
                            cached_data = compressed_data
                        pickle.dump(cached_data, f)            
        
        # reset cache to free memory?
        if isinstance(self.codec, NNTrainableModule):
            self.codec.reset_all_cache()


    # TODO: update this function
    def __call__(self, idxs: Iterable[int] = None):

        metric_logger = MetricLogger()
        save_prefix = ""

        # set codec variables
        if isinstance(self.codec, VariableRateCodecInterface) and self.variable_rate_level is not None:
            try:
                self.codec.set_rate_level(self.variable_rate_level)
            except:
                self.logger.warning(f"set_rate_level {self.variable_rate_level} failed! Will skip this worker!")
                return dict()
            save_prefix += f"vrlevel{self.variable_rate_level}_"
        if isinstance(self.codec, VariableComplexityCodecInterface) and self.variable_complex_level is not None:
            try:
                self.codec.set_complex_level(self.variable_complex_level)
            except:
                self.logger.warning(f"set_complex_level {self.variable_complex_level} failed! Will skip this worker!")
                return dict()
            save_prefix += f"sclevel{self.variable_complex_level}_"
        if isinstance(self.codec, VariableTaskCodecInterface) and self.variable_task is not None:
            try:
                assert self.codec.set_task(self.variable_task)
            except:
                if self.ignore_invalid_variable_task:
                    self.logger.warning(f"set_task {self.variable_task} failed! Ignoring and continuing benchmark...")
                else:
                    self.logger.warning(f"set_task {self.variable_task} failed! Will skip this worker!")
                    return dict()
            save_prefix += f"task{self.variable_task}_"

        if self.distortion_metric is not None:
            self.distortion_metric.reset()

        # for step, data in dataloader:
        if idxs is None:
            idxs = range(len(self.dataloader))
            for idx, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"pid={os.getpid()}"):
                self._run_step(idx, data, metric_logger, save_prefix)
        else:
            # TODO: some dataloader may not perform the same with indexing and iterating!
            for idx in tqdm(idxs, desc=f"pid={os.getpid()}"):
                step = idx
                if idx >= len(self.dataloader): continue
                data = self.dataloader[idx]

                self._run_step(idx, data, metric_logger, save_prefix)

        all_metrics = metric_logger.get_global_average()
        if self.distortion_metric is not None:
            all_metrics.update(**self.distortion_metric.collect_metrics())
        return all_metrics


class BasicLosslessCompressionBenchmark(BaseBenchmark):
    def __init__(self, codec: CodecInterface, dataloader: DataLoaderInterface, *args,
                 metric_logger=None,  # TODO: an interface for metric_logger
                 add_intermediate_to_metric=False,
                 # training
                 need_training=False,
                 training_dataloader=None,
                 trainer=None,
                 training_config=dict(),
                 load_checkpoint=True,
                 save_checkpoint=True,
                 checkpoint_file="params.pkl",
                #  max_training_samples=-1,
                 # testing
                 cache_compressed_data=False,
                 cache_checksum=True,
                 cache_subdir="cache",
                 skip_decompress=False,
                 save_decompressed_data=False,
                 save_subdir="decompressed",
                 save_format="image",
                 nn_codec_use_forward_pass=False,
                 nn_codec_forward_pass_skip_compression=True,
                 distortion_metric : Optional[BaseMetric] = None,
                 testing_variable_rate_levels : Optional[List[int]] = None,
                 testing_variable_rate_bj_delta_metric : Optional[BJDeltaMetric] = None,
                 testing_variable_rate_bj_delta_metric_keys : Tuple[str, str] = ("compressed_length", "psnr"), # deprecated
                 testing_complexity_levels : Optional[List[int]] = None,
                 testing_tasks : Optional[List[str]] = None,
                 testing_task_workers : Optional[List[BenchmarkTestingWorker]] = None,
                 testing_task_metrics : Optional[List[BaseMetric]] = None,
                 testing_task_variable_rate_bj_delta_metrics : Optional[List[BJDeltaMetric]] = None,
                 num_repeats=1,
                 num_testing_workers=0,
                 skip_trainer_testing=False,
                 force_basic_testing=False,
                 force_testing_device=None,
                 **kwargs):

        # output_dir for saving results
        self.cache_compressed_data = cache_compressed_data
        self.cache_checksum = cache_checksum
        self.cache_subdir = cache_subdir
        # moved to setup_engine
        # self.cache_dir = os.path.join(self.output_dir, cache_subdir)
        # if self.cache_compressed_data and not os.path.exists(self.cache_dir):
        #     os.makedirs(self.cache_dir)

        self.save_decompressed_data = save_decompressed_data
        self.save_subdir = save_subdir
        self.save_format = save_format
        
        # nn codec benchmark
        self.nn_codec_use_forward_pass = nn_codec_use_forward_pass
        self.nn_codec_forward_pass_skip_compression = nn_codec_forward_pass_skip_compression

        # benchmark flow
        self.skip_decompress = skip_decompress
        self.distortion_metric = distortion_metric
        self.testing_variable_rate_levels = testing_variable_rate_levels
        self.testing_variable_rate_bj_delta_metric = testing_variable_rate_bj_delta_metric
        # self.testing_variable_rate_bj_delta_metric_keys = testing_variable_rate_bj_delta_metric_keys
        self.current_testing_variable_rate_level_idx = 0
        self.testing_complexity_levels = testing_complexity_levels
        self.current_testing_complexity_level_idx = 0
        self.testing_tasks = testing_tasks
        self.testing_task_workers = testing_task_workers
        self.testing_task_metrics = testing_task_metrics
        self.testing_task_variable_rate_bj_delta_metrics = testing_task_variable_rate_bj_delta_metrics
        self.current_testing_task_idx = 0

        self.num_repeats = num_repeats
        self.num_testing_workers = num_testing_workers
        if num_testing_workers == -1:
            self.num_testing_workers = multiprocessing.cpu_count()
        self.skip_trainer_testing = skip_trainer_testing
        self.force_basic_testing = force_basic_testing
        self.force_testing_device = force_testing_device

        # if self.num_testing_workers > 0:
        #     self.testing_pool = multiprocessing.Pool(self.num_testing_workers)
        # else:
        #     self.testing_pool = None

        # metric logger
        self.metric_logger = MetricLogger() if metric_logger is None else metric_logger
        self.metric_logger.add_meter("compression_ratio", SmoothedValue(fmt="{global_avg:.4f}"))
        self.metric_logger.add_meter("time_compress", SmoothedValue(fmt="{global_avg:.2f} ms"))
        self.metric_logger.add_meter("speed_compress", SmoothedValue(fmt="{global_avg:.2f} MB/s"))
        self.metric_logger.add_meter("time_decompress", SmoothedValue(fmt="{global_avg:.2f} ms"))
        self.metric_logger.add_meter("speed_decompress", SmoothedValue(fmt="{global_avg:.2f} MB/s"))

        # an intermediate logger for finding bottlenecks
        self.itmd_logger = MetricLogger()
        self.itmd_logger.add_meter("time_dataloader", SmoothedValue(fmt="{median:.2f} ({global_avg:.2f}) ms"))
        self.itmd_logger.add_meter("time_iter", SmoothedValue(fmt="{median:.2f} ({global_avg:.2f}) ms"))
        
        self.add_intermediate_to_metric = add_intermediate_to_metric
        
        self.need_training = need_training
        self.training_dataloader = dataloader if training_dataloader is None else training_dataloader
        
        # setup trainer
        if trainer is None:
            trainer = BasicTrainer(self.training_dataloader, **training_config)
        # elif isinstance(trainer, BaseEngine):
        #     trainer.setup_engine_from_copy(self)
        # else:
        #     raise ValueError("Invalid trainer!")
        self.trainer = trainer
        self.training_config = training_config
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.checkpoint_file = checkpoint_file
        # self.max_training_samples = max_training_samples

        # initialize benchmark engine
        super().__init__(codec, dataloader, *args, **kwargs)


    def setup_engine(self, *args, output_dir=None, **kwargs):
        super().setup_engine(*args, output_dir=output_dir, **kwargs)
        # setup cache output
        if self.output_dir is not None:
            self.cache_dir = os.path.join(self.output_dir, self.cache_subdir)
            self.save_dir = os.path.join(self.output_dir, self.save_subdir)
            if self.cache_compressed_data and not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            if self.save_decompressed_data and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        else:
            self.logger.warning("Cache dir not properly setup!")
        # setup trainer output
        if isinstance(self.trainer, BaseEngine):
            self.trainer.setup_engine(*args, output_dir=output_dir, **kwargs)


    def set_codec(self, codec) -> None:
        super().set_codec(codec)
        # inject profiler (seems it is not needed as all codecs are BaseModule which has it own profiler)
        # if isinstance(self.codec, BaseCodec):
        #     self.itmd_logger.reset()
        #     self.codec.profiler = self.itmd_logger

        # training
        if not isinstance(self.codec, TrainableModuleInterface) and self.need_training:
            self.logger.warning("Codec is not trainable! Skip training!")
            self.need_training = False

        # TODO: maybe could be removed? Let self.trainer handle this?
        if isinstance(self.codec, SelfTrainableInterface) and self.codec.is_trainer_valid() and not self.codec.is_trainer_setup():
            self.codec.setup_trainer_engine(
                output_dir=self.output_dir, 
            )
            self.codec.do_train()

    def _estimate_byte_length(self, data):
        import torch
        # a lazy solution for evaluating total bytes...
        # pickle.dumps add some useless information such as class definitions!
        # return len(pickle.dumps(data))
        if isinstance(data, bytes):
            return len(data) # TODO: length of string?
        elif isinstance(data, str):
            return len(data.encode('utf-8')) # TODO: length of string?
        elif isinstance(data, torch.Tensor):
            return int(np.prod(data.shape)) * data.element_size()
        elif isinstance(data, np.ndarray):
            return int(data.size * data.itemsize)
        else:
            raise ValueError("Bitstream of data {} in type {} cannot be estimated!".format(data, type(data)))

    # def _run_step(self, step: int, data: Any, target: Optional[Any] = None):
    #     # search cache
    #     # TODO: update to support variable rate/complex/task benchmarks
    #     compressed_data = None
    #     if self.output_dir:
    #         if self.cache_compressed_data:
    #             # NOTE: maybe using hash values and filenames? dataloader may be shuffled!
    #             cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
    #             if os.path.exists(cache_file):
    #                 with open(cache_file, 'rb') as f:
    #                     compressed_data = pickle.load(f)
    #                     if self.cache_checksum:
    #                         compressed_data, checksum = compressed_data
    #                         original_data_checksum = hashlib.md5(data).digest()
    #                         # checksum fail! rerun codec!
    #                         if checksum != original_data_checksum:
    #                             self.logger.warning(f"Checksum fails for data iteration {step}. Check if the dataloader is deterministic!")
    #                             compressed_data = None

    #     # run codec compress
    #     if compressed_data is None:
    #         time_start = time.time()
    #         compressed_data = self.codec.compress(data)
    #         time_compress = time.time() - time_start

    #     try:
    #         compressed_length = self._estimate_byte_length(compressed_data) 
    #     except ValueError:
    #         self.logger.warning("Cannot estimate compressed data length! Using pickle as an alt way.")
    #         compressed_string = pickle.dumps(compressed_data)
    #         compressed_length = len(compressed_string)
    #     # compressed_bits = compressed_length * 8

    #     # TODO: the data format should be defined!
    #     try:
    #         original_length = self._estimate_byte_length(data) 
    #     except ValueError:
    #         self.logger.warning("Cannot estimate original data length! Using pickle as an alt way.")
    #         original_string = pickle.dumps(data)
    #         original_length = len(original_string)
    #         # original_bits = original_length * 8

    #     compression_ratio = compressed_length / original_length

    #     self.metric_logger.update(
    #         compression_ratio=compression_ratio,
    #         compressed_length=compressed_length,
    #         original_length=original_length,
    #         time_compress=time_compress*1000,
    #         speed_compress=(original_length/time_compress/1024/1024),
    #     )

    #     if not self.skip_decompress:
    #         time_start = time.time()
    #         decompressed_data = self.codec.decompress(compressed_data)
    #         time_decompress = time.time() - time_start

    #         # NOTE: use original_length or compressed_length when calculating speed_decompress?
    #         self.metric_logger.update(
    #             time_decompress=time_decompress*1000,
    #             speed_decompress=(original_length/time_decompress/1024/1024),
    #             time_total=(time_compress+time_decompress)*1000,
    #             speed_total=(original_length/(time_compress+time_decompress)/1024/1024),
    #         )
    #         # check decompress correctness for lossless
    #         # assert(data == decompressed_data)

    #         # distortion
    #         distortion_metric = self.distortion_metric
    #         if len(self.testing_task_metrics) > 1:
    #             distortion_metric = self.testing_task_metrics[self.current_testing_task_idx]
    #         if distortion_metric is not None:
    #             # if isinstance(distortion_metric, dict):
    #             #     for name, method in distortion_metric.items():
    #             #         self.metric_logger[f'distortion_{name}'] = method(data, decompressed_data)
    #             # else:
    #             if target is None:
    #                 target = data
    #             results = distortion_metric(decompressed_data, target)
    #             # NOTE: results will be collected after all steps are done!
    #             # if results is not None:
    #             #     self.metric_logger.update(**results)


    #     # cache compressed data
    #     # TODO: update to support variable rate/complex/task benchmarks
    #     if self.output_dir:
    #         if self.cache_compressed_data:
    #             cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
    #             if not os.path.exists(cache_file):
    #                 with open(cache_file, 'wb') as f:
    #                     if self.cache_checksum:
    #                         original_data_checksum = hashlib.md5(data).digest()
    #                         cached_data = (compressed_data, original_data_checksum)
    #                     else:
    #                         cached_data = compressed_data
    #                     pickle.dump(cached_data, f)            
        
    #     # reset cache to free memory?
    #     if isinstance(self.codec, NNTrainableModule):
    #         self.codec.reset_all_cache()

    def run_training(self, *args, **kwargs):
        # training step
        if self.trainer is not None:
            training_config = dict(
                load_checkpoint=self.load_checkpoint,
                save_checkpoint=self.save_checkpoint,
                checkpoint_file=self.checkpoint_file
            )
            training_config.update(**self.training_config)
            self.trainer.train_module(self.codec, *args,
                **training_config, **kwargs
            )
            # checkpoint_path = os.path.join(self.output_dir, self.checkpoint_file)
            # # load checkpoint
            # if self.load_checkpoint and os.path.exists(checkpoint_path):
            #     self.logger.info("Loading checkpoint from {} ...".format(checkpoint_path))
            #     with open(checkpoint_path, 'rb') as f:
            #         params = pickle.load(f)
            #         self.codec.load_parameters(params)
            # else:
            #     self.logger.info("Start Training!")
            #     if self.trainer == "fulldata":
            #         # TODO: use a special training dataloader
            #         training_samples = list(self.dataloader)
            #         if self.max_training_samples:
            #             random.shuffle(training_samples)
            #             training_samples = training_samples[:self.max_training_samples]
            #         self.codec.train_full(training_samples)
            #     else:
            #         raise ValueError("Trainer {} not supported!".format(self.trainer))

            #     # save checkpoint
            #     if self.save_checkpoint:
            #         self.logger.info("Saving checkpoint to {} ...".format(checkpoint_path))
            #         params = self.codec.get_parameters()
            #         with open(checkpoint_path, 'wb') as f:
            #             pickle.dump(params, f)

    def run_testing(self, *args, **kwargs):
        if self.codec is None:
            raise ValueError("No codec to benchmark!")

        self.logger.info("Starting benchmark testing!")
        metrics = dict()
        metrics_2d = dict() # for vr/sc models
        try:
            # raise NotImplementedError()
            # custom trainer testing
            # TODO: metric logger?
            if self.trainer is not None:
                metrics_trainer = dict()
                if not self.skip_trainer_testing:
                    metrics_trainer = self.trainer.test_module(self.codec,
                        **self.training_config # TODO: may use independent config for testing!
                    )
                else:
                    # just load checkpoint
                    self.trainer.load_checkpoint(self.codec, checkpoint_file=self.checkpoint_file)
                
                # NOTE: when using ddp distributed training in self.trainer, subprocess should skip testing and return here.
                if self.trainer.should_end_process():
                    return

                if self.force_basic_testing:
                    if isinstance(metrics_trainer, dict):
                        metrics.update(**metrics_trainer)
                    # else:
                    #     # NOTE: when using ddp distributed training in self.trainer, subprocess should skip testing and return here.
                    #     self.logger.warning("No metric collected from trainer testing procedure! Exiting...")
                    #     self.logger.warning("When using ddp distributed training in self.trainer, subprocess should skip testing and return here. This warning may could be ignored.")
                    #     return
                else:
                    return metrics_trainer
                
        except NotImplementedError:
            self.logger.info("Using default benchmark testing!")
            
        benchmark_testing_params = dict(
            cache_dir=self.cache_dir,
            cache_checksum=self.cache_checksum,
            cache_compressed_data=self.cache_compressed_data,
            skip_decompress=self.skip_decompress,
            save_decompressed_data=self.save_decompressed_data,
            save_dir=self.save_dir,
            save_format=self.save_format,
            nn_codec_use_forward_pass=self.nn_codec_use_forward_pass,
            nn_codec_forward_pass_skip_compression=self.nn_codec_forward_pass_skip_compression,
        )

        current_worker = BenchmarkTestingWorker(self.codec, self.dataloader,
            # metric_logger=self.metric_logger,
            distortion_metric=self.distortion_metric,
            **benchmark_testing_params
        )

        # TODO: create all possible workers and process them in parallel if possible
        testing_variable_rate_levels = self.testing_variable_rate_levels
        if testing_variable_rate_levels is not None and isinstance(self.codec, VariableRateCodecInterface):
            # assert isinstance(self.codec, VariableRateCodecInterface)
            if len(testing_variable_rate_levels) == 0:
                testing_variable_rate_levels = list(range(self.codec.num_rate_levels))
            self.current_testing_variable_rate_level_idx = 0
            # if self.testing_variable_rate_bj_delta_metric is not None:
            # for calculating bj_delta
            rate_pts, distortion_pts = [], []
        else:
            testing_variable_rate_levels = []

        testing_complexity_levels = self.testing_complexity_levels
        if testing_complexity_levels is not None and isinstance(self.codec, VariableComplexityCodecInterface):
            # assert isinstance(self.codec, VariableComplexityCodecInterface)
            if len(testing_complexity_levels) == 0:
                testing_complexity_levels = list(range(self.codec.num_complex_levels))
            self.current_testing_complexity_level_idx = 0
        else:
            testing_complexity_levels = []

        # NOTE: VariableTaskCodecInterface is not required for multi-task testing!
        testing_tasks = self.testing_tasks
        testing_task_workers = self.testing_task_workers
        testing_task_metrics = self.testing_task_metrics
        testing_task_variable_rate_bj_delta_metrics = self.testing_task_variable_rate_bj_delta_metrics
        if (testing_task_workers is not None or testing_task_metrics is not None): # and isinstance(self.codec, VariableTaskCodecInterface):
            self.current_testing_task_idx = 0
            if testing_task_workers is None:
                testing_task_workers = [
                    BenchmarkTestingWorker(
                        variable_task=testing_tasks[i],
                        distortion_metric=distortion_metric,
                        **benchmark_testing_params,
                    ) for i, distortion_metric in enumerate(testing_task_metrics)
                ]
        else:
            # testing_tasks = [0]
            # testing_task_workers = [current_worker]
            # testing_task_metrics = [self.distortion_metric]
            # testing_task_variable_rate_bj_delta_metrics = [self.testing_variable_rate_bj_delta_metric]
            testing_tasks = []
            testing_task_workers = []
            testing_task_metrics = []
            testing_task_variable_rate_bj_delta_metrics = []
        assert len(testing_task_workers) == len(testing_task_variable_rate_bj_delta_metrics)

        for _ in range(self.num_repeats):

            # variable rate/scalable testing loop
            while True:

                distortion_metric = self.distortion_metric

                # stop benchmark if vr/vc/vt exceeds
                if len(testing_task_workers) > 0:
                    if self.current_testing_task_idx >= len(testing_task_workers):
                        break
                    # override worker
                    current_worker = testing_task_workers[self.current_testing_task_idx]
                    # NOTE: task is set within worker!
                    # if isinstance(self.codec, VariableTaskCodecInterface):
                    #     self.codec.set_task(self.current_testing_task_idx)
                if len(testing_variable_rate_levels) > 0:
                    if self.current_testing_variable_rate_level_idx >= len(testing_variable_rate_levels):
                        break
                    current_worker.variable_rate_level = testing_variable_rate_levels[self.current_testing_variable_rate_level_idx]
                    # self.codec.set_rate_level(testing_variable_rate_levels[self.current_testing_variable_rate_level_idx])
                if len(testing_complexity_levels) > 0:
                    if self.current_testing_complexity_level_idx >= len(testing_complexity_levels):
                        break
                    current_worker.variable_complex_level = testing_complexity_levels[self.current_testing_complexity_level_idx]
                    # self.codec.set_complex_level(testing_complexity_levels[self.current_testing_complexity_level_idx])

                # TODO: when using distributed training, the distributed backend could not end propoerly
                # which causes testing to run on multiple processes.
                # this is just a temporary fix!
                # NOTE: it seems this causes deadlock...
                # if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0: break

                # for trainable codecs
                if isinstance(self.codec, TrainableModuleInterface):
                    if isinstance(self.codec, NNTrainableModule):
                        self.codec.eval()
                        if self.force_testing_device is not None:
                            try:
                                self.codec.to(device=self.force_testing_device)
                            except:
                                self.logger.warning(f"Device {self.force_testing_device} not applicable on this machine! Testing on default CPU!")
                    # run post_training_process for safety (if training is interrupted)
                    self.codec.post_training_process()
                self.codec.update_state()

                # Use default benchmark testing
                # current_worker = BenchmarkTestingWorker(self.codec, self.dataloader,
                #     # metric_logger=self.metric_logger,
                #     variable_rate_level=self.current_testing_variable_rate_level_idx,
                #     variable_complex_level=self.current_testing_complexity_level_idx,
                #     variable_task=self.current_testing_task_idx,
                #     distortion_metric=distortion_metric,
                #     cache_dir=self.cache_dir,
                #     cache_checksum=self.cache_checksum,
                #     cache_compressed_data=self.cache_compressed_data,
                #     skip_decompress=self.skip_decompress,
                # )

                # set objects for worker
                current_worker.codec = self.codec
                if current_worker.dataloader is None:
                    current_worker.dataloader = self.dataloader
                if current_worker.distortion_metric is None and distortion_metric is not None:
                    distortion_metric.reset()
                    current_worker.distortion_metric = distortion_metric
                    
                # override caching/saving params
                if self.cache_compressed_data:
                    current_worker.cache_compressed_data = self.cache_compressed_data
                    current_worker.cache_dir = self.cache_dir
                    current_worker.cache_checksum = self.cache_checksum
                if self.save_decompressed_data:
                    current_worker.save_decompressed_data = self.save_decompressed_data
                    current_worker.save_dir = self.save_dir
                    current_worker.save_format = self.save_format

                time_start = time.time()
                # data_cache = []

                can_pickle = True
                try:
                    pickle.dumps(current_worker)
                except:
                    self.logger.warning("Cannot pickle worker! Multiprocessing disabled!")
                    can_pickle = False

                # disable multiprocessing if we need to test on non-cpu device
                if self.num_testing_workers > 0 and can_pickle and self.force_testing_device is None:
                    with multiprocessing.Pool(self.num_testing_workers) as testing_pool:
                        # dataloader_segment = len(self.dataloader) // self.num_testing_workers
                        # dataloader_split = [enumerate(self.dataloader[(i*dataloader_segment):((i+1)*dataloader_segment)], start=i*dataloader_segment) for i in range(self.num_testing_workers)]
                        # metric_data = self.testing_pool.starmap(worker, dataloader_split)
                        # for metric_dict in metric_data:
                        #     self.metric_logger.update(**metric_dict)
                        # data_cache = []
                        # async_results = []
                        # for i, data in enumerate(self.dataloader):
                        #     data_cache.append((i, data))
                        #     if len(data_cache) > 100:
                        #         async_results.append(self.testing_pool.starmap_async(worker, [data_cache]))
                        #         data_cache = []
                        num_segments = self.num_testing_workers # * 10
                        segment_length = len(self.dataloader) // num_segments
                        dataloader_split = [range(i*segment_length, min((i+1)*segment_length, len(self.dataloader))) for i in range(num_segments)]
                        # for split in tqdm(dataloader_split):
                        #     result = worker(split)
                        results = testing_pool.map(current_worker, dataloader_split)
                        for result in results:
                            self.metric_logger.update(**result)
                        # results = [self.testing_pool.apply_async((split, )) for split in dataloader_split]
                        # for result in tqdm(results):
                        #     try:
                        #         metric_dict = result.get()
                        #         self.metric_logger.update(**metric_dict)
                        #     except:
                        #         traceback.print_exc()
                else:
                    # try:
                        self.metric_logger.update(**current_worker())
                    # except Exception as e:
                    #     self.logger.error("Basic Testing found an error! Aborting...")
                    #     self.logger.error(traceback.format_exc())
                    #     break
                        # compressed_data = worker(i, data)

                    # for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                    #     # Transform coding with different target
                    #     if isinstance(data, tuple) and len(data) == 2:
                    #         data, target = data
                    #     else:
                    #         target = None

                    #     # if not i in [38, 47, 82, 86, 92, 97]: continue
                    #     # if i != 92: continue
                    #     # print(i)
                    #     time_dataloader = time.time() - time_start
                    #     self.itmd_logger.update(time_dataloader=time_dataloader*1000)

                    #     with self.itmd_logger.start_time_profile("time_iter"):
                    #         # if self.testing_pool is not None:
                    #         #     data_cache.append((i, data))
                    #         #     if len(data_cache) == self.num_testing_workers or i == len(self.dataloader)-1:
                    #         #         metric_data = self.testing_pool.starmap(worker, data_cache)
                    #         #         for metric_dict in metric_data:
                    #         #             self.metric_logger.update(**metric_dict)
                    #         #         data_cache = []
                    #         # else:
                    #         try:
                    #             compressed_data = self._run_step(i, data, target=target)
                    #         except Exception as e:
                    #             self.logger.error("Basic Testing found an error! Aborting...")
                    #             self.logger.error(traceback.format_exc())
                    #             break
                    #             # compressed_data = worker(i, data)

                    #     if i % 1000 == 0:
                    #         self.logger.info(self.itmd_logger)
                        
                    #     time_start = time.time()

                # NOTE: we dont want external multi-processing (such as pytorch ddp) to function in this benchmark!
                # self.metric_logger.synchronize_between_processes()

                # update metrics
                current_level_metrics = self.metric_logger.get_global_average()
                if distortion_metric is not None:
                    current_level_metrics.update(**distortion_metric.collect_metrics())

                if self.add_intermediate_to_metric:
                    if isinstance(self.codec, BaseCodec):
                        intermediate_metrics = self.codec.collect_profiler_results(recursive=True, clear=True)
                        self.logger.info("Running Intermediate Logger: {}".format(
                                json.dumps(
                                    intermediate_metrics, 
                                    indent=4
                                )
                            )
                        )
                        current_level_metrics.update(**intermediate_metrics)

                metric_prefixes = []
                if len(testing_task_workers) > 0:
                    # add metrics from testing_task_metrics
                    # current_level_metrics.update(**testing_task_metrics[self.current_testing_task_idx].collect_metrics())
                    metric_prefixes.append(f"task{current_worker.variable_task}")
                if len(testing_complexity_levels) > 0:
                    metric_prefixes.append(f"sclevel{current_worker.variable_complex_level}")
                if len(testing_variable_rate_levels) > 0:
                    testing_variable_rate_bj_delta_metric = self.testing_variable_rate_bj_delta_metric
                    if len(testing_task_workers) > 0:
                        testing_variable_rate_bj_delta_metric = testing_task_variable_rate_bj_delta_metrics[self.current_testing_task_idx]
                    if testing_variable_rate_bj_delta_metric is not None:
                        # bj_delta_rate_metric_name = testing_variable_rate_bj_delta_metric_keys[0]
                        # if bj_delta_distortion_metric_name is None:
                        #     bj_delta_distortion_metric_name = testing_variable_rate_bj_delta_metric_keys[1]
                        bj_delta_rate_metric_name, bj_delta_distortion_metric_name = testing_variable_rate_bj_delta_metric.collect_metric_names
                        if bj_delta_rate_metric_name in current_level_metrics and bj_delta_distortion_metric_name in current_level_metrics:
                            rate_pts.append(current_level_metrics[bj_delta_rate_metric_name])
                            distortion_pts.append(current_level_metrics[bj_delta_distortion_metric_name])
                    metric_prefixes.append(f"vrlevel{current_worker.variable_rate_level}")
                
                metric_prefix = ""
                if len(metric_prefixes) > 0:
                    metric_prefix = "_".join(metric_prefixes) # + "_"
                    self.logger.info(f"Using {metric_prefix} as metric prefix")
                    self.logger.info(current_level_metrics)
                    if not metric_prefix in metrics_2d:
                        metrics_2d[metric_prefix] = dict()
                for key, value in current_level_metrics.items():
                    metrics[f"{metric_prefix}_{key}"] = value
                    metrics_2d[metric_prefix][key] = value

                if len(testing_complexity_levels) > 1:
                    # add complexity metrics
                    complexity_metrics = self.codec.get_current_complex_metrics()
                    for key, value in complexity_metrics.items():
                        metrics[f"{metric_prefix}_{key}"] = value
                        if len(metric_prefix) > 0:
                            metrics_2d[metric_prefix][key] = value

                # update next level (sequence is vr-vc-vt)
                if len(testing_variable_rate_levels) > 0:
                    self.metric_logger.clear()
                    self.current_testing_variable_rate_level_idx += 1
                    # meshgrid levels
                    if self.current_testing_variable_rate_level_idx >= len(testing_variable_rate_levels):
                        testing_variable_rate_bj_delta_metric = self.testing_variable_rate_bj_delta_metric

                        # bj_delta requires at least 4 points
                        if len(testing_variable_rate_levels) > 3:
                            bj_delta_prefix = ""
                            if len(testing_complexity_levels) > 0:
                                bj_delta_prefix = f"sclevel{self.current_testing_complexity_level_idx}_" + bj_delta_prefix
                            if len(testing_task_workers) > 0:
                                bj_delta_prefix = f"taskidx{self.current_testing_task_idx}_" + bj_delta_prefix
                                testing_variable_rate_bj_delta_metric = testing_task_variable_rate_bj_delta_metrics[self.current_testing_task_idx]
                            if testing_variable_rate_bj_delta_metric is not None:
                                bj_delta = testing_variable_rate_bj_delta_metric((rate_pts, distortion_pts))[testing_variable_rate_bj_delta_metric.name]
                                metrics[bj_delta_prefix + testing_variable_rate_bj_delta_metric.name] = bj_delta
                                metrics_2d[metric_prefix][testing_variable_rate_bj_delta_metric.name] = bj_delta
                                rate_pts, distortion_pts = [], []
                        
                        if len(testing_complexity_levels) > 0:
                            self.current_testing_variable_rate_level_idx = 0
                            self.current_testing_complexity_level_idx += 1
                            if self.current_testing_complexity_level_idx >= len(testing_complexity_levels):
                                if len(testing_task_workers) > 0:
                                    self.current_testing_complexity_level_idx = 0
                                    self.current_testing_task_idx += 1
                        elif len(testing_task_workers) > 0:
                            self.current_testing_variable_rate_level_idx = 0
                            self.current_testing_task_idx += 1
                        else:
                            pass

                elif len(testing_complexity_levels) > 0:
                    self.metric_logger.clear()
                    self.current_testing_complexity_level_idx += 1
                    if self.current_testing_complexity_level_idx >= len(testing_complexity_levels):
                        self.current_testing_complexity_level_idx = 0
                        if len(testing_task_workers) > 0:
                            self.current_testing_task_idx += 1
                elif len(testing_task_workers) > 0:
                    self.metric_logger.clear()
                    self.current_testing_task_idx += 1
                else:
                    break


                # if hasattr(self.codec, "profiler"):
                #     self.logger.info("Running Intermediate Logger: {}".format(self.codec.profiler))

        
        # save metrics_2d as csv
        if len(metrics_2d) > 0:
            self.save_metrics(metric_file=os.path.join(self.output_dir, "metrics_2d.csv"), 
                              metric_data=list(metrics_2d.values()), names=list(metrics_2d.keys()))

        return metrics


    def run_benchmark(self, *args,
        run_training=True,
        run_testing=True,
        ignore_exist_metrics=False,
        **kwargs):
        if self.codec is None:
            raise ValueError("No codec to benchmark!")

        # check if the benchmark has been run by checking metric file
        if not ignore_exist_metrics and os.path.exists(self.metric_raw_file):
            self.logger.warning("Metric file {} already exists! Skipping benchmark...".format(self.metric_raw_file))
            self.logger.warning("Specify ignore_exist_metrics=True if you want to restart the benchmark.")
            # read metric file as output
            metric_data = None
            with open(self.metric_raw_file, 'rb') as f:
                # read the first line
                # metric_data = next(csv.DictReader(f))
                metric_data = pickle.load(f)
            
            return metric_data
        
        self.metric_logger.reset()

        if run_training:
            self.run_training(*args, **kwargs)

        if run_testing:
            metric_dict = self.run_testing(*args, **kwargs)
            self.save_metrics(metric_dict)
            return metric_dict

    def collect_metrics(self, *args, **kwargs):
        # TODO: this may be invalid when using custom test_module function!
        return self.metric_logger.get_global_average()


class GroupedLosslessCompressionBenchmark(BasicLosslessCompressionBenchmark):
    def __init__(self, codec_group: Union[CodecInterface, List[CodecInterface]],
                 dataloader: DataLoaderInterface,
                 *args,
                 **kwargs):
        if isinstance(codec_group, CodecInterface):
            codec_group = [codec_group]
        self.codec_group = codec_group
        super().__init__(codec_group[0], dataloader, *args, **kwargs)

        self.cached_metrics = []

    def run_benchmark(self, *args, **kwargs):
        self.cached_metrics = []
        for idx, codec in enumerate(self.codec_group):
            if self.trainer is not None:
                codec_output_dir = os.path.join(self.output_dir, "codec{}".format(idx))
                self.trainer.setup_engine(output_dir=codec_output_dir)
            self.codec = codec
            metric_dict = super().run_benchmark(*args, **kwargs)
            self.cached_metrics.append(metric_dict)

        return self.collect_metrics()

    def collect_metrics(self, *args, **kwargs):
        # TODO: draw a plot?
        return self.cached_metrics