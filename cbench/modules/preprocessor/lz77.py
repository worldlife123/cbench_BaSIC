
from collections import namedtuple
import functools
from typing import Dict, List, Tuple
import numpy as np
import zstandard

from cbench.modules.preprocessor.lz77_dict_training import dict_training_fastcover, dict_training_fastcover_tryparameters
from cbench.zstd_wrapper import zdict_train_from_buffer, zdict_finalize_dictionary, ZDICT_fastCover_params_t

from .base import Preprocessor
from ..base import TrainableModuleInterface

from cbench.zstd_wrapper import ZSTD_cParameter, zstd_lz77_forward, zstd_lz77_reverse

# TODO: initial state of cache
def to_relative_offset(offsets, codes=3, mode=None, **kwargs):
    if mode is None:
        mode = "delta"

    if mode == "delta":
        relative_offsets = offsets[1:] - offsets[:-1]
        offsets_new = offsets + codes + 1
        ro_mask = np.abs(relative_offsets) <= codes//2
        offsets_new[1:][ro_mask] = relative_offsets[ro_mask] + codes//2 + 1
    elif mode == "cache":
        # NOTE: this cache mode is still different from zstd, but in many cases should be close!
        offsets_new = offsets + codes
        # make sure that smaller relative idxs overwrites bigger ones
        for i in range(codes, 0, -1):
            if i >= len(offsets): continue
            relative_offsets_i = offsets[i:] - offsets[:-i]
            offsets_new[i:] = np.where(relative_offsets_i == 0, np.zeros(len(offsets)-i)+i, offsets_new[i:])
    else:
        raise NotImplementedError(f"Unknown mode {mode}")

    return offsets_new

def from_relative_offset(offsets, codes=3, mode=None, **kwargs):
    if mode is None:
        mode = "delta"

    if mode == "delta":
        ro_mask = offsets[1:] <= codes
        offsets_new = offsets - codes - 1
        # offsets_new[1:][ro_mask] = offsets_new[:-1][ro_mask] + offsets[1:][ro_mask] - codes//2

        # TODO: implement the serial process in c++ to obtain better speed
        for idx, is_rel in enumerate(ro_mask):
            if is_rel:
                offsets_new[idx+1] = offsets_new[idx] + offsets[idx+1] - codes//2 - 1
    elif mode == "cache":
        ro_mask = offsets[1:] <= codes
        offsets_new = offsets - codes
        for idx, code_val in enumerate(offsets):
            if code_val <= codes:
                offsets_new[idx] = offsets_new[idx-code_val]
    else:
        raise NotImplementedError(f"Unknown mode {mode}")

    return offsets_new

class LZ77Preprocessor(Preprocessor):
    def __init__(self, *args, 
                 compressor_config : Dict[ZSTD_cParameter, int] = dict(), 
                 level=0,
                 relative_offset_codes=None,
                 relative_offset_mode=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor_config = compressor_config
        if level > 0:
            self.compressor_config[ZSTD_cParameter.ZSTD_c_compressionLevel] = level
        
        if relative_offset_codes is None or relative_offset_codes < 0:
            relative_offset_codes = None
        if relative_offset_codes is not None:
            assert relative_offset_codes % 2 == 1, "relative_offset_codes should be odd!"
        self.relative_offset_codes = relative_offset_codes
        self.relative_offset_mode = relative_offset_mode
        assert relative_offset_mode in [None, "delta", "cache"]
        # self.lz77_encoder = functools.partial(zstd_lz77_forward, config=self.compressor_config)
        # self.lz77_decoder = functools.partial(zstd_lz77_reverse)
    
    def lz77_encoder(self, x):
        return zstd_lz77_forward(x, config=self.compressor_config)

    def lz77_decoder(self, x):
        return zstd_lz77_reverse(x)

    def _modify_seq(self, literals, offsets, lit_lengths, match_lengths):
        if self.relative_offset_codes is not None:
            offsets = to_relative_offset(offsets, codes=self.relative_offset_codes, mode=self.relative_offset_mode)
        return literals, offsets, lit_lengths, match_lengths

    def _demodify_seq(self, literals, offsets, lit_lengths, match_lengths):
        if self.relative_offset_codes is not None:
            offsets = from_relative_offset(offsets, codes=self.relative_offset_codes, mode=self.relative_offset_mode)
        return literals, offsets, lit_lengths, match_lengths

    def preprocess(self, data, *args, prior=None, **kwargs):
        literals, seqs = self.lz77_encoder(data)
        literals = np.frombuffer(literals, dtype=np.uint8)
        seqs_np = np.array(seqs)
        # print(data, literals, seqs_np)
        assert(seqs_np.shape[1] == 4)
        # seqs_np = seqs_np.astype(np.uint8)
        # return dict(
        #     literals=
        #     litLengths=
        #     matchLengths=
        #     offsets=
        # )
        # print(zstd_lz77_reverse((literals, [tuple(seq) for seq in seqs_np.tolist()])))
        offsets, lit_lengths, match_lengths = seqs_np[:,0], seqs_np[:,1], seqs_np[:,2]
        self._cache = (literals, offsets, lit_lengths, match_lengths)
        literals, offsets, lit_lengths, match_lengths = self._modify_seq(
            literals, offsets, lit_lengths, match_lengths
        )
        return (literals, offsets, lit_lengths, match_lengths)

    def postprocess(self, data, *args, prior=None, **kwargs):
        literals, offsets, lit_lengths, match_lengths = data
        # offsets = np.frombuffer(offsets, dtype=np.uint8)
        # lit_lengths = np.frombuffer(lit_lengths, dtype=np.uint8)
        # match_lengths = np.frombuffer(match_lengths, dtype=np.uint8)
        literals, offsets, lit_lengths, match_lengths = self._demodify_seq(
            literals, offsets, lit_lengths, match_lengths
        )
        for d1, d2 in zip((literals, offsets, lit_lengths, match_lengths), self._cache):
            assert (d1==d2).all()

        literals = literals.astype(np.uint8).tobytes()
        seqs_np = np.stack([
            offsets,
            lit_lengths,
            match_lengths,
            np.zeros(len(offsets), dtype=offsets.dtype), # leave rep as 0
        ], axis=-1)
        return self.lz77_decoder((literals, [tuple(seq) for seq in seqs_np.tolist()]))


class LZ77DictPreprocessor(LZ77Preprocessor, TrainableModuleInterface):
    def __init__(self, *args, 
                 level=3,
                 dict_size=1000,
                 dict_initialize: bytes = b"",
                 dict_training_method : str = None,
                 dict_config : dict = dict(),
                 dict_relative_offset=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_dict = zstandard.ZstdCompressionDict(dict_initialize, **dict_config).as_bytes()
        self.level = level
        self.dict_size = dict_size
        self.dict_training_method = "default" if dict_training_method is None else dict_training_method
        self.dict_config = dict_config
        self.dict_relative_offset = dict_relative_offset

        # self.lz77_encoder = lambda x : zstd_lz77_forward(x, self.compression_dict.as_bytes(), config=self.compressor_config)
        # self.lz77_decoder = lambda x : zstd_lz77_reverse(x, self.compression_dict.as_bytes())
        # self.lz77_encoder = functools.partial(zstd_lz77_forward, dict_string=self.compression_dict.as_bytes(), config=self.compressor_config)
        # self.lz77_decoder = functools.partial(zstd_lz77_reverse, dict_string=self.compression_dict.as_bytes())
    
    def lz77_encoder(self, x):
        return zstd_lz77_forward(x, self.compression_dict, config=self.compressor_config)

    def lz77_decoder(self, x):
        return zstd_lz77_reverse(x, self.compression_dict)

    # def preprocess(self, data, *args, prior=None, **kwargs):
    #     literals, offsets, lit_lengths, match_lengths = super().preprocess(data, *args, prior=prior, **kwargs)
    #     if self.dict_relative_offset:
    #         relative_offset_threshold = np.cumsum(lit_lengths + match_lengths)
    #         offsets = self.dict_size + relative_offset_threshold - offsets
    #     return literals, offsets, lit_lengths, match_lengths


    # def postprocess(self, data, *args, prior=None, **kwargs):
    #     literals, offsets, lit_lengths, match_lengths = data
    #     if self.dict_relative_offset:
    #         relative_offset_threshold = np.cumsum(lit_lengths + match_lengths)
    #         offsets = self.dict_size + relative_offset_threshold - offsets
    #     return super().postprocess((literals, offsets, lit_lengths, match_lengths), *args, prior=prior, **kwargs)

    def _modify_seq(self, literals, offsets, lit_lengths, match_lengths):
        if self.dict_relative_offset:
            relative_offset_threshold = np.cumsum(lit_lengths + match_lengths)
            offsets = self.dict_size + relative_offset_threshold - offsets
        return super()._modify_seq(literals, offsets, lit_lengths, match_lengths)

    def _demodify_seq(self, literals, offsets, lit_lengths, match_lengths):
        literals, offsets, lit_lengths, match_lengths = \
            super()._demodify_seq(literals, offsets, lit_lengths, match_lengths)
        if self.dict_relative_offset:
            relative_offset_threshold = np.cumsum(lit_lengths + match_lengths)
            offsets = self.dict_size + relative_offset_threshold - offsets
        return (literals, offsets, lit_lengths, match_lengths)

    def train_full(self, dataloader: List[bytes], *args, **kwargs):
        if self.dict_training_method == "default":
            self.compression_dict = zstandard.train_dictionary(
                dict_size=self.dict_size,
                samples=dataloader,
                **self.dict_config
            ).as_bytes()
        elif self.dict_training_method == "custom_fastcover":
            params = ZDICT_fastCover_params_t()
            params.d = 8
            params.steps = 4
            # params.countUniqueFreq = 1
            # params.scoreFreqMean = 1
            self.dict_buffer = zdict_train_from_buffer(
                self.dict_size, dataloader, params
                # **self.dict_config
            )
        elif self.dict_training_method == "pyfastcover":
            # dict_training_fastcover_tryparameters too slow!
            raw_dict = dict_training_fastcover(
                dataloader, 
                dict_length=self.dict_size,
                **self.dict_config
            )
            self.compression_dict = zdict_finalize_dictionary(raw_dict, dataloader)
        else:
            raise NotImplementedError(f"Unknown self.dict_training_method {self.dict_training_method}")

    def train_iter(self, data, *args, **kwargs) -> None:
        raise NotImplementedError("Dictionary training does not support iterable training!")

    def get_parameters(self, *args, **kwargs) -> bytes:
        return self.compression_dict

    def load_parameters(self, parameters: bytes, *args, **kwargs) -> None:
        self.compression_dict = zstandard.ZstdCompressionDict(parameters, *args, **kwargs).as_bytes()