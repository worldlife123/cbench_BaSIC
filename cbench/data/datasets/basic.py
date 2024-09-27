# TODO: this seems to be similar to torch.utils.data, maybe use pytorch dataset instead
import bisect
import numpy as np
from typing import Callable, List
from collections import OrderedDict
import os
from torch.utils.data.dataset import Dataset

from cbench.utils.engine import BaseEngine

class BasicDataset(object):
    # TODO: check dataset iterable and indexable
    def __init__(self, *args, transform : Callable = None, **kwargs):
        self.transform = transform

    def do_transform(self, data):
        if self.transform is not None:
            data = self.transform(data)
        return data

class MappingDataset(BasicDataset):
    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class IterableDataset(BasicDataset):
    def __iter__(self):
        raise NotImplementedError()


class CachedFileMappingDataset(BaseEngine, MappingDataset):
    def __init__(self, file_list, *args, root=None, max_cache_size=0, cache_strategy="persistent", **kwargs):
        kwargs["output_dir"] = root # overwrite output_dir for BaseEngine
        super().__init__(*args, **kwargs)
        self.file_list = file_list
        self.root = root
        
        # inner cache
        self.use_cache = (max_cache_size > 0)
        if self.use_cache:
            self.max_cache_size = max_cache_size
            self.cache_strategy = cache_strategy
            self.current_cache_size = 0
            if self.cache_strategy == "persistent":
                self.cache = dict()
            elif self.cache_strategy == "rr":
                self.cache = dict()
            elif self.cache_strategy == "fifo":
                self.cache = OrderedDict()
            else:
                raise NotImplementedError(f"Unknown cache_strategy {self.cache_strategy}")

        # TODO: check if file exists

    def _fetch_or_add_to_cache(self, index : int):
        if self.use_cache and index in self.cache:
            # read from cache
            byte_string = self.cache[index]
        else:
            # add to cache
            file_path = self.file_list[index]
            if self.root is not None:
                file_path = os.path.join(self.root, file_path)
            with open(file_path, 'rb') as f:
                byte_string = f.read()
                if self.use_cache:
                    file_bytes = len(byte_string)
                    if file_bytes + self.current_cache_size > self.max_cache_size:
                        if self.cache_strategy == "persistent":
                            # refuse caching
                            return byte_string
                        elif self.cache_strategy == "rr":
                            while file_bytes + self.current_cache_size > self.max_cache_size:
                                self.current_cache_size -= len(self.cache.pop(next(self.cache.keys())))
                        elif self.cache_strategy == "fifo":
                            while file_bytes + self.current_cache_size > self.max_cache_size:
                                self.current_cache_size -= len(self.cache.pop(next(self.cache.keys())))
                    self.cache[index] = byte_string
                    self.current_cache_size += file_bytes
        return byte_string

    def _load_file(self, index) -> bytes:
        raise NotImplementedError()

    def __getitem__(self, index):
        return self._fetch_or_add_to_cache(index)

    def __len__(self) -> int:
        return len(self.file_list)


class ConcatMappingDataset(MappingDataset):
    def __init__(self, datasets : List[MappingDataset], *args, transform : Callable = None, **kwargs):
        super().__init__(*args, transform=transform, **kwargs)
        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.cumulative_sizes = np.cumsum(np.array(self.dataset_lengths))

    # from torch.utils.data.dataset.Dataset
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data_sample = self.datasets[dataset_idx][sample_idx]
        return self.do_transform(data_sample)

    def __len__(self):
        return self.cumulative_sizes[-1]
