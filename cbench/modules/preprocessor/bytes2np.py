
from collections import namedtuple
import functools
from typing import Dict, List, Tuple
import numpy as np
import zstandard

from .base import Preprocessor
from ..base import TrainableModuleInterface

from cbench.zstd_wrapper import ZSTD_cParameter, zstd_lz77_forward, zstd_lz77_reverse

class Bytes2NumpyPreprocessor(Preprocessor):
    def __init__(self, *args, 
                 dtype=np.uint8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def preprocess(self, data : bytes, *args, prior=None, **kwargs):
        return np.frombuffer(data, dtype=self.dtype)

    def postprocess(self, data : np.ndarray, *args, prior=None, **kwargs):
        return data.tobytes()