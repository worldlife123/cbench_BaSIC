from typing import List, Union
import os
import csv
import pickle
import copy
import hashlib

from configs.class_builder import ClassBuilder, ClassBuilderBase, ClassBuilderList, NamedParamBase

from cbench.benchmark.basic_benchmark import BasicLosslessCompressionBenchmark
from cbench.benchmark.base import BaseBenchmark
from cbench.codecs.base import CodecInterface
from cbench.data.base import DataLoaderInterface


class GroupedCodecBenchmarkBuilder(BaseBenchmark, ClassBuilderBase):
    def __init__(self, codec_group_builder: ClassBuilderList,
                #  dataloader: DataLoaderInterface,
                 benchmark_builder: ClassBuilder,
                 *args,
                 codec_name_length_limit=256,
                 codec_name_hash_length=8,
                 group_name="",
                 **kwargs):
        self.codec_group_builder = codec_group_builder
        self.benchmark_builder = benchmark_builder
        self.codec_name_length_limit = codec_name_length_limit
        self.codec_name_hash_length = codec_name_hash_length
        self.group_name = group_name

        super().__init__(None, None, *args, **kwargs)

        self.cached_metrics = []

    @property
    def name(self) -> str:
        return self.group_name + self.benchmark_builder.name

    @property
    def param(self):
        return self.benchmark_builder.param

    def build_class(self, *args, sync_url=None, **kwargs) -> object:
        # NOTE: we first build benchmark so that codecs could use obj ref of benchmark for extra params such as output_dir
        # NOTE: we leave sync parameters to subdirs for codecs so that we do not need to sync the whole experiment dir!
        self.benchmark = self.benchmark_builder.build_class(*args, **kwargs)
        self.codec_names = [cb.name for cb in self.codec_group_builder]
        # NOTE: build codecs later when running benchmark?
        # self.codec_group = [cb.build_class() for cb in self.codec_group_builder]
        assert(isinstance(self.benchmark, BaseBenchmark))
        self.setup_engine_from_copy(self.benchmark)
        self._cached_sync_url = sync_url
        return self

    def run_benchmark(self, *args, 
        ignore_exist_metrics=False,
        codecs_ignore_exist_metrics=False,
        **kwargs):

        # check if the benchmark has been run by checking metric file
        if not ignore_exist_metrics and os.path.exists(self.metric_raw_file):
            self.logger.warning("Metric file {} already exists! Skipping benchmark...".format(self.metric_raw_file))
            self.logger.warning("Specify ignore_exist_metrics=True if you want to restart the benchmark.")
            return

        self.cached_metrics = []
        metric_data_all = []
        hparams_all = []
        names_all = []
        for idx, codec_builder in enumerate(self.codec_group_builder):
            codec = codec_builder.build_class()
            codec_build_name = self.codec_group_builder[idx].build_name()
            codec_name_full = self.codec_group_builder[idx].name
            codec_name = self.codec_group_builder[idx].get_name_under_limit(
                name_length_limit=self.codec_name_length_limit, 
                hash_length=self.codec_name_hash_length,
            )
            codec_hashtag = self.codec_group_builder[idx].get_hashtag(hash_length=self.codec_name_hash_length)
            # avoid filename too long 
            # if len(codec_name_full) > self.codec_name_length_limit:
            #     config_hash = hashlib.sha256(codec_name_full.encode()).hexdigest()[:self.codec_name_hash_length]
            #     codec_name = f"{config_hash}:{codec_name_full[:(self.codec_name_length_limit-len(config_hash))]}..."
            # else:
            #     codec_name = codec_name_full

            # setup benchmark
            codec_output_dir = os.path.join(self.output_dir, codec_name)
            assert(isinstance(self.benchmark, BaseBenchmark))
            self.benchmark.setup_engine_from_copy(self,
                output_dir=codec_output_dir,
                sync_url=self._cached_sync_url,
            )
            
            # TODO: check if we should overwrite existing files or configs?

            # save full codec name
            with open(os.path.join(codec_output_dir, "build_name.txt"), 'w') as f:
                f.write(codec_build_name)
            with open(os.path.join(codec_output_dir, "config_name.txt"), 'w') as f:
                f.write(codec_name_full)
            # TODO: use ClassBuilderDict to pass in a custom exp name for the codec
            with open(os.path.join(codec_output_dir, "exp_name.txt"), 'w') as f:
                f.write(codec_name_full)

            # save a copy of class builder in codec_output_dir for reproduction
            # TODO: maybe give a warning of using objref which may rely on other codec configs?
            benchmark_builder_copy = self.benchmark_builder.clone(copy_slot_data=True, copy_obj_ref=False).update_args(
                codec=self.codec_group_builder[idx].clone(copy_slot_data=True, copy_obj_ref=False),
            )
            with open(os.path.join(codec_output_dir, 'config.pkl'), 'wb') as f: 
                # TODO: We should clone the codec class builder to avoid saving cached obj ref
                # pickle.dump(
                #     benchmark_builder_copy.update_args(
                #         self.codec_group_builder[idx]).clone(copy_slot_data=True),
                #     f
                # )
                pickle.dump(benchmark_builder_copy, f)

            self.benchmark.set_codec(codec)
            self.logger.info(f"Running benchmark with codec: {codec_name_full}")
            self.logger.info(f"Output dir: {codec_output_dir}")
            # run benchmark and save metrics
            metric_dict = self.benchmark.run_benchmark(*args, 
                ignore_exist_metrics=codecs_ignore_exist_metrics, **kwargs)
            self.cached_metrics.append(metric_dict)
            config_dict = {name:param for name, param in self.codec_group_builder[idx].iter_slot_data()}
            # config_and_metric_dict = dict()
            # if isinstance(metric_dict, dict):
            #     config_and_metric_dict.update(**metric_dict)
            #     # config_and_metric_dict.update(codec_name=self.codec_group_builder[idx].name)
            # config_and_metric_dict.update(**config_dict)
            metric_data_all.append(metric_dict)
            hparams_all.append(config_dict)
            names_all.append(codec_name)
            
            self.benchmark.stop_engine()

        self.save_metrics(metric_data=metric_data_all, hparams=hparams_all, names=names_all)
        
        # Finally we should upload everything if sync_url is enabled!
        if self._cached_sync_url is not None:
            self.setup_engine_from_copy(self, sync_url=self._cached_sync_url, sync_start_action=None)
            self.logger.info(f"File sync : final upload start!")
            self.sync_utils.upload_directory(self.remote_dir, self.output_dir)
            self.logger.info(f"File sync : final upload complete!")
            self.stop_engine()

        return self.collect_metrics()

    def collect_metrics(self, *args, **kwargs):
        # TODO: draw a plot?
        return self.cached_metrics