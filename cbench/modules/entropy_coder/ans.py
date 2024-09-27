import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from typing import Tuple, Optional, List, Union

from .torch_base import TorchQuantizedEntropyCoder
from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder


class ANSEntropyCoder(TorchQuantizedEntropyCoder):
    # FREQ_PRECISION = 16

    def __init__(self, *args, 
        coder_type="rans64",
        use_bypass_coding=False,
        freq_precision=16,
        default_freqs : Optional[Union[List[float], List[List[float]]]] = None,
        default_indexes : Optional[List[int]] = None,
        default_data_shape : Optional[List[int]] = None, 
        **kwargs):
        super().__init__(*args, **kwargs)
        
        self.coder_type = coder_type
        self.use_bypass_coding = use_bypass_coding
        self.freq_precision = freq_precision
        # self.update_state()

        self.default_freqs = default_freqs
        self.default_indexes = default_indexes
        self.default_data_shape = default_data_shape # TODO

        if default_freqs is not None:
            self.default_freqs = np.array(default_freqs)
            if self.default_freqs.ndim == 1:
                self.default_freqs = self.default_freqs[None]
                # self.default_indexes = np.zeros(1)
            else:
                assert self.default_indexes is not None

        if default_indexes is not None:
            self.default_indexes = np.array(default_indexes)

    def _select_best_indexes(self, prior) -> torch.LongTensor:
        """ quantize prior distributions to indices

        Args:
            prior (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.LongTensor: _description_
        """        
        raise NotImplementedError()
    
    def _get_ar_params(self, prior) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return None
    
    def _get_ans_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def _data_preprocess_with_prior(self, data, prior, **kwargs):
        data = self._data_preprocess(data, **kwargs) 
        data = data % self.data_precision # use mod to control range
        # NOTE: by default prior is not used
        return data

    def _data_postprocess_with_prior(self, data, prior, **kwargs):
        data = self._data_postprocess(data, **kwargs)
        # NOTE: by default prior is not used
        return data

    def encode(self, data, *args, prior=None, **kwargs):
        with self.profiler.start_time_profile("time_prior_preprocess_encode"):
            if prior is not None:
                indexes = self._select_best_indexes(prior)
                # quant_prior = self._init_dist_params()[indexes.reshape(-1)]
                indexes = indexes.detach().cpu().contiguous().numpy()
            else:
                assert self.default_indexes is not None
                indexes = self.default_indexes
                # raise ValueError("prior should not be None!")
            
            ar_indexes, ar_offsets = None, None
            ar_params = self._get_ar_params(prior)
            if ar_params is not None:
                ar_indexes, ar_offsets = ar_params
                # ar_indexes = ar_indexes.detach().cpu().numpy()
                # ar_offsets = ar_offsets.detach().cpu().numpy()

        with self.profiler.start_time_profile("time_data_preprocess_encode"):
            data = self._data_preprocess_with_prior(data.contiguous(), prior)
            # data = data.reshape(-1)

        with self.profiler.start_time_profile("time_ans_encode"):
            byte_string = self.encoder.encode_with_indexes(data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
            # peek = self.encoder.peek_cache()
            # byte_string = self.encoder.flush()

        return byte_string


    def decode(self, byte_string: bytes, *args, prior=None, data_length=None, **kwargs):
        # assert(prior is not None) # TODO: default prior?
        with self.profiler.start_time_profile("time_prior_preprocess_decode"):
            if prior is not None:
                # data_shape = prior.shape[:-1]
                indexes = self._select_best_indexes(prior)
                indexes = indexes.detach().cpu().contiguous().numpy()
            else:
                assert self.default_indexes is not None
                indexes = self.default_indexes
                # raise ValueError("prior should not be None!")

            # TODO: data_shape from bytes
            data_shape = indexes.shape

            ar_indexes, ar_offsets = None, None
            ar_params = self._get_ar_params(prior)
            if ar_params is not None:
                ar_indexes, ar_offsets = ar_params
                # ar_indexes = ar_indexes.detach().cpu().numpy()
                # ar_offsets = ar_offsets.detach().cpu().numpy()

        with self.profiler.start_time_profile("time_ans_decode"):
            symbols = self.decoder.decode_with_indexes(byte_string, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)

        with self.profiler.start_time_profile("time_data_postprocess_decode"):
            data = np.array(symbols).reshape(*data_shape)
            data = self._data_postprocess_with_prior(data, prior)
        return data

    def update_state(self, *args, **kwargs) -> None:
        if self.coder_type == "rans" or self.coder_type == "rans64":
            encoder = Rans64Encoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding)
            decoder = Rans64Decoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding)
        elif self.coder_type == "tans":
            encoder = TansEncoder(table_log=self.freq_precision, max_symbol_value=self.data_precision-1, bypass_coding=self.use_bypass_coding)
            decoder = TansDecoder(table_log=self.freq_precision, max_symbol_value=self.data_precision-1, bypass_coding=self.use_bypass_coding)
        else:
            raise NotImplementedError(f"Unknown coder type {self.coder_type}")

        if self.default_freqs is not None:
            freqs = self.default_freqs
            nfreqs = np.array([len(freqs) for freqs in self.default_freqs])
            offsets = np.zeros(len(self.default_freqs), dtype=np.int32) # [0] * len(indexes)
        else:
            freqs, nfreqs, offsets = self._get_ans_params()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)

        self.encoder = encoder
        self.decoder = decoder
