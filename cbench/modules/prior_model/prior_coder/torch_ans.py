import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Tuple, Optional

from .base import PriorCoder
from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder
from cbench.nn.base import NNTrainableModule


class TorchANSPriorCoder(PriorCoder, NNTrainableModule):
    def __init__(self, *args, 
                 quantizer_type="uniform",
                 quantizer_params=None,
                 training_quantizer_type="universal",
                #  data_range=(0, 1),
                #  data_step=None,
                 data_precision=8,
                 coder_type="rans64",
                 freq_precision=16,
                 use_bypass_coding=False,
                 bypass_precision=4,
        **kwargs):
        # super().__init__(*args, **kwargs) # NOTE: may cause unexpected kwargs error! dunno why...
        super().__init__() 
        NNTrainableModule.__init__(self)
    
        self.quantizer_type = quantizer_type
        self.training_quantizer_type = training_quantizer_type
        # self.data_range = data_range
        # self.data_step = data_step
        # if self.data_step is None:
        #     self.data_step = (self.data_range[1] - self.data_range[0]) / (1<<self.data_precision - 1) 
        if self.quantizer_type == "uniform":
            if quantizer_params is None:
                quantizer_params = [0.0, 1 << data_precision - 1, 1.0]
            assert len(quantizer_params) == 3
        elif self.quantizer_type == "uniform_scale":
            if quantizer_params is None:
                quantizer_params = [1.0]
            assert len(quantizer_params) == 1
        elif self.quantizer_type == "nonuniform":
            if quantizer_params is None:
                quantizer_params = [i + 0.5 for i in range(1 << data_precision)]
            assert len(quantizer_params) == (1 << data_precision)
        elif self.quantizer_type == "vector":
            if quantizer_params is None:
                quantizer_params = torch.rand(1 << data_precision, 64) # TODO: param for codebook size
            assert len(quantizer_params) == (1 << data_precision)
        # TODO: trainable params?
        self.register_buffer("quantizer_params", torch.as_tensor(quantizer_params), persistent=False)

        self.data_precision = data_precision

        self.coder_type = coder_type
        self.freq_precision = freq_precision
        self.data_precision = data_precision
        self.use_bypass_coding = use_bypass_coding
        self.bypass_precision = bypass_precision
        # self.update_state()

    def _build_coding_params(self, prior : torch.Tensor) -> Tuple[torch.IntTensor, torch.IntTensor]:
        return self._build_coding_indexes(prior), self._build_coding_offsets(prior)

    def _build_coding_indexes(self, prior : torch.Tensor) -> torch.IntTensor:
        """ quantize prior distributions to indices

        Args:
            prior (_type_): _description_

        Returns:
            Tuple[np.ndarray, np.ndarrray]: (indices, data offsets)
        """        
        raise NotImplementedError()

    def _build_coding_offsets(self, prior : torch.Tensor) -> torch.IntTensor:
        """ quantize prior distributions to data offsets

        Args:
            prior (_type_): _description_

        Returns:
            Tuple[np.ndarray, np.ndarrray]: (indices, data offsets)
        """        
        raise NotImplementedError()

    def _get_ar_params(self, prior) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return None
    
    def _get_ans_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    # @property
    # def data_step(self):
    #     return (self.data_range[1] - self.data_range[0]) / (1<<self.data_precision - 1)

    # @property
    # def data_mid(self):
    #     return (self.data_range[1] + self.data_range[0]) / 2

    def _data_preprocess(self, data, quantizer_params=None, transform=True, quantize=True, differentiable=False, dequantize=False, to_numpy=False, **kwargs):
        quantizer_params = self.quantizer_params if quantizer_params is None else quantizer_params
        
        # Transform
        if transform:
            if self.quantizer_type == "uniform":
                data = (data - quantizer_params[0]) / quantizer_params[2]
            elif self.quantizer_type == "uniform_scale":
                data = data / quantizer_params
            # TODO:
            elif self.quantizer_type == "nonuniform":
                raise NotImplementedError()
            elif self.quantizer_type == "vector":
                raise NotImplementedError()
            else:
                raise NotImplementedError(f"Unknown quantizer_type {self.quantizer_type}")

        # Quantize
        if quantize:
            # Differentiable quantize
            quantized_data = data.round()
            if differentiable:
                if self.training_quantizer_type == "none":
                    pass
                elif self.training_quantizer_type == "universal":
                    # TODO: universal quantizer for non-uniform?
                    noise = torch.empty_like(data).uniform_(-0.5, 0.5)
                    if self.quantizer_type == "uniform":
                        quantized_data = data + noise # * quantizer_params[2]
                    elif self.quantizer_type == "uniform_scale":
                        quantized_data = data + noise # * quantizer_params
                    else:
                        raise NotImplementedError("Universal quantization do not support non-uniform quantization!")
                elif self.training_quantizer_type == "st":
                    quantized_data = data + (quantized_data - data.detach())
                else:
                    raise NotImplementedError(f"Unknown training_quantizer_type {self.training_quantizer_type}")                

            if dequantize:
                if self.quantizer_type == "uniform":
                    dequantized_data = quantized_data * quantizer_params[2] + quantizer_params[0]
                elif self.quantizer_type == "uniform_scale":
                    dequantized_data = quantized_data * quantizer_params
                # TODO:
                elif self.quantizer_type == "nonuniform":
                    raise NotImplementedError()
                elif self.quantizer_type == "vector":
                    raise NotImplementedError()
                else:
                    raise NotImplementedError(f"Unknown quantizer_type {self.quantizer_type}")
                return dequantized_data

            if to_numpy:
                quantized_data = quantized_data.detach().cpu().numpy().astype(np.int32)
            return quantized_data
        
        return data
    
    def _data_postprocess(self, data, quantizer_params=None, **kwargs):
        quantizer_params = self.quantizer_params if quantizer_params is None else quantizer_params

        if isinstance(data, np.ndarray):
            data = torch.as_tensor(data.astype(np.float32), device=self.device)
        if self.quantizer_type == "uniform":
            data = data * quantizer_params[2] + quantizer_params[0]
        elif self.quantizer_type == "uniform_scale":
            data = data * quantizer_params
        # TODO:
        elif self.quantizer_type == "nonuniform":
            raise NotImplementedError()
        elif self.quantizer_type == "vector":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Unknown quantizer_type {self.quantizer_type}")
        
        return data
    
    def forward(self, data, *args, prior=None, **kwargs):
        raise NotImplementedError()

    def encode(self, data, *args, prior=None, **kwargs):
        assert hasattr(self, "ans_encoder"), "Not Initialized! Should call self.update_state() before coding!"
        with self.profiler.start_time_profile("time_prior_preprocess_encode"):
            if prior is not None:
                indexes, data_offsets = self._build_coding_params(prior)
            else:
                raise ValueError("prior should not be None!")
            
            ar_indexes, ar_offsets = None, None
            ar_params = self._get_ar_params(prior)
            if ar_params is not None:
                ar_indexes, ar_offsets = ar_params
                # ar_indexes = ar_indexes.detach().cpu().numpy()
                # ar_offsets = ar_offsets.detach().cpu().numpy()

        with self.profiler.start_time_profile("time_data_preprocess_encode"):
            data = self._data_preprocess(data, to_numpy=True)
            data = data - data_offsets
            # data = data.reshape(-1)

        with self.profiler.start_time_profile("time_ans_encode"):
            byte_string = self.ans_encoder.encode_with_indexes(data, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)
            # peek = self.ans_encoder.peek_cache()
            # byte_string = self.ans_encoder.flush()

        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, data_length=None, **kwargs):
        assert hasattr(self, "ans_decoder"), "Not Initialized! Should call self.update_state() before coding!"
        assert prior is not None # TODO: default prior?
        data_shape = prior.shape[:-1]
        with self.profiler.start_time_profile("time_prior_preprocess_decode"):
            if prior is not None:
                indexes, data_offsets = self._build_coding_params(prior)
            else:
                raise ValueError("prior should not be None!")

            ar_indexes, ar_offsets = None, None
            ar_params = self._get_ar_params(prior)
            if ar_params is not None:
                ar_indexes, ar_offsets = ar_params
                # ar_indexes = ar_indexes.detach().cpu().numpy()
                # ar_offsets = ar_offsets.detach().cpu().numpy()

        with self.profiler.start_time_profile("time_ans_decode"):
            symbols = self.ans_decoder.decode_with_indexes(byte_string, indexes, ar_indexes=ar_indexes, ar_offsets=ar_offsets)

        with self.profiler.start_time_profile("time_data_postprocess_decode"):
            symbols = symbols + data_offsets
            data = self._data_postprocess(data).reshape(*data_shape)
        return data

    def update_state(self, *args, **kwargs) -> None:
        if self.coder_type == "rans" or self.coder_type == "rans64":
            encoder = Rans64Encoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding, bypass_precision=self.bypass_precision)
            decoder = Rans64Decoder(freq_precision=self.freq_precision, bypass_coding=self.use_bypass_coding, bypass_precision=self.bypass_precision)
        elif self.coder_type == "tans":
            encoder = TansEncoder(table_log=self.freq_precision, max_symbol_value=(1<<self.data_precision)-1, bypass_coding=self.use_bypass_coding, bypass_precision=self.bypass_precision)
            decoder = TansDecoder(table_log=self.freq_precision, max_symbol_value=(1<<self.data_precision)-1, bypass_coding=self.use_bypass_coding, bypass_precision=self.bypass_precision)
        else:
            raise NotImplementedError(f"Unknown coder type {self.coder_type}")

        freqs, nfreqs, offsets = self._get_ans_params()
        encoder.init_params(freqs, nfreqs, offsets)
        decoder.init_params(freqs, nfreqs, offsets)
        self.ans_encoder = encoder
        self.ans_decoder = decoder


class ContinuousDistributionANSPriorCoder(TorchANSPriorCoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.dist_params = self._init_dist_params()

    def _init_dist_params(self) -> torch.Tensor:
        raise NotImplementedError()

    def _select_best_indexes(self, params) -> torch.LongTensor:
        raise NotImplementedError()

    def _params_to_dist(self, params : torch.Tensor) -> D.Distribution:
        # prior_mean, prior_logvar = params.chunk(2, dim=1)
        # return D.Normal(prior_mean, torch.exp(prior_logvar))
        raise NotImplementedError()

    def _params_to_dist_and_offset(self, params : torch.Tensor) -> Tuple[D.Distribution, torch.Tensor]:
        # prior_mean, prior_logvar = params.chunk(2, dim=1)
        # return D.Normal(prior_mean, torch.exp(prior_logvar))
        raise NotImplementedError()

    def _build_coding_indexes(self, prior : torch.Tensor) -> torch.IntTensor:
        indexes = self._select_best_indexes(prior)
        return indexes

    def _build_coding_offsets(self, prior : torch.Tensor) -> torch.IntTensor:
        dist = self._params_to_dist(prior)
        data_offset = dist.mean # .int()
        return data_offset

    def _get_ans_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        freq_cnt = 1 << self.freq_precision
        tail_mass = torch.tensor([0.5 / freq_cnt], device=self.device)
        prior_cnts, num_symbols, offsets = [], [], []
        for dist_param in self._init_dist_params():
            if dist_param.ndim == 0: dist_param = dist_param.unsqueeze(0)
            dist = self._params_to_dist(dist_param.unsqueeze(0)) # for batch dim

            dist_min = int(dist.icdf(tail_mass).floor().item())
            dist_max = int(dist.icdf(1 - tail_mass).ceil().item())
            offsets.append(dist_min)
            num_symbols.append(dist_max - dist_min + 1)

            pts = torch.arange(dist_min - 1, dist_max + 1).type_as(dist.mean) + 0.5
            prior_logprob = (dist.cdf(pts[1:]) - dist.cdf(pts[:-1])).log()[0]  # remove batch dim
            prior_pmfs = torch.softmax(prior_logprob, dim=-1) # .clamp_min(np.log(1./self.freq_precision_total))

            prior_cnt = (prior_pmfs * freq_cnt).clamp_min(1)
            prior_cnt = prior_cnt.detach().cpu().contiguous().numpy().astype(np.int32)
            prior_cnts.append(prior_cnt)

        prior_cnts_np = np.zeros((len(prior_cnts), max([len(prior_cnt) for prior_cnt in prior_cnts])), dtype=np.int32)
        for i, prior_cnt in enumerate(prior_cnts):
            prior_cnts_np[i, :len(prior_cnt)] = prior_cnt
        num_symbols = np.array(num_symbols, dtype=np.int32)
        offsets = np.array(offsets, dtype=np.int32)
        return prior_cnts_np, num_symbols, offsets
