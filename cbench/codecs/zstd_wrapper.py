import struct
from typing import Any, Dict, List, Tuple
import numpy as np
import zstandard

from cbench.modules.base import TrainableModuleInterface
from cbench.modules.preprocessor import LZ77DictPreprocessor
from cbench.modules.entropy_coder import GroupedEntropyCoder, TrainablePredCntTANSEntropyCoder
from cbench.modules.preprocessor.lz77_dict_training import dict_training_fastcover

from .base import BaseCodec
from .general_codec import GeneralCodec

from cbench.zstd_wrapper import ZSTD_cParameter, \
    zstd_extract_lz77_sequences, zstd_compress_with_lz77_sequences, \
    zstd_extract_and_compress_with_lz77_sequences, \
    zdict_train_from_buffer, zdict_finalize_dictionary, ZDICT_fastCover_params_t
from cbench.zstd_wrapper import ZstdWrapper, zstd_compress, zstd_decompress

class ZstdWrapperCodec(BaseCodec):
    def __init__(self, *args, 
                 compressor_config : Dict[ZSTD_cParameter, int] = dict(), 
                 max_length=1<<23,
                 use_sequences=False,
                 sequence_exec="py",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor_config = compressor_config
        self.max_length = max_length
        self.use_sequences = use_sequences
        self.sequence_exec = sequence_exec
        
        self.zstd_wrapper = ZstdWrapper()

    def compress(self, data, *args, **kwargs) -> bytes:
        assert(len(data) < self.max_length), "Data of length {} exceeds max length".format(len(data))
        # TODO: compression level
        if self.use_sequences:
            if self.sequence_exec == "py":
                with self.profiler.start_time_profile("time_compress_extract_lz77_sequences"):
                    lz77_seqs = zstd_extract_lz77_sequences(data)
                with self.profiler.start_time_profile("time_compress_with_lz77_sequences"):
                    comp_data = zstd_compress_with_lz77_sequences(data, lz77_seqs)
            else:
                with self.profiler.start_time_profile("time_compress_extract_and_compress_with_lz77_sequences"):
                    comp_data = zstd_extract_and_compress_with_lz77_sequences(data)
        else:
            with self.profiler.start_time_profile("time_compress_wo_sequences"):
                # comp_data = zstd_compress(data)
                comp_data = self.zstd_wrapper.compress_once(data)
        return comp_data

    def decompress(self, data: bytes, *args, **kwargs):
        # return zstd_decompress(data, self.max_length)
        return self.zstd_wrapper.decompress_once(data, self.max_length)


class ZstdDictTrainer(TrainableModuleInterface):
    def __init__(self, *args, 
                 dict_size=1000,
                 dict_initialize: bytes = b"",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_dict = zstandard.ZstdCompressionDict(dict_initialize)
        self.dict_size = dict_size

    def train_full(self, dataloader: List[bytes], *args, **kwargs):
        self.compression_dict = zstandard.train_dictionary(
            dict_size=self.dict_size,
            samples=dataloader,
            **kwargs
        )

    def train_iter(self, data, *args, **kwargs) -> None:
        raise NotImplementedError("Dictionary training does not support iterable training!")

    def get_parameters(self, *args, **kwargs) -> bytes:
        return self.compression_dict.as_bytes()

    def load_parameters(self, parameters: bytes, *args, **kwargs) -> None:
        self.compression_dict = zstandard.ZstdCompressionDict(parameters, *args, **kwargs)


class ZDictWrapperTrainer(TrainableModuleInterface):
    def __init__(self, *args, 
                 dict_size=1000,
                 dict_initialize: bytes = b"",
                 dict_training_method = None,
                 dict_config = dict(),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_size = dict_size
        self.dict_buffer = dict_initialize
        self.dict_training_method = "default" if dict_training_method is None else dict_training_method
        self.dict_config = dict_config

    def train_full(self, dataloader: List[bytes], *args, **kwargs):
        if self.dict_training_method == "default":
            # self.dict_buffer = zdict_train_from_buffer(
            #     self.dict_size, dataloader
            #     # **self.dict_config
            # )
            self.dict_buffer = zstandard.train_dictionary(
                dict_size=self.dict_size,
                samples=dataloader,
                **self.dict_config
            ).as_bytes()
        elif self.dict_training_method == "custom_fastcover":
            params = ZDICT_fastCover_params_t()
            params.d = 8
            # params.k = 250
            # params.splitPoint = 1.0
            params.steps = 4
            # params.countUniqueFreq = 1
            # params.scoreFreqMean = 1
            params.zParams.notificationLevel = 4
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
            self.dict_buffer = zdict_finalize_dictionary(raw_dict, dataloader)
        else:
            raise NotImplementedError(f"Unknown self.dict_training_method {self.dict_training_method}")
        # print(self.dict_buffer)

    def train_iter(self, data, *args, **kwargs) -> None:
        raise NotImplementedError("Dictionary training does not support iterable training!")

    def get_parameters(self, *args, **kwargs) -> bytes:
        return self.dict_buffer

    def load_parameters(self, parameters: bytes, *args, **kwargs) -> None:
        self.dict_buffer = parameters


class ZstdDictWrapperCodec(ZstdWrapperCodec, ZDictWrapperTrainer):
    def __init__(self, *args, 
                 dict_size=1000,
                 dict_initialize: bytes = b"",
                 dict_training_method = None,
                 dict_config = dict(),
                 **kwargs):
        ZDictWrapperTrainer.__init__(self, 
            dict_size=dict_size, 
            dict_initialize=dict_initialize,
            dict_training_method=dict_training_method,
            dict_config=dict_config,
        )
        super().__init__(*args, **kwargs)
        # self.compression_dict = zstandard.ZstdCompressionDict(dict_initialize)
        # self.dict_size = dict_size

    def compress(self, data, *args, **kwargs) -> bytes:
        dict_string = self.get_parameters()
        assert(len(data) < self.max_length), "Data of length {} exceeds max length".format(len(data))
        if self.use_sequences:
            if self.sequence_exec == "py":
                with self.profiler.start_time_profile("time_compress_extract_lz77_sequences"):
                    lz77_seqs = zstd_extract_lz77_sequences(data, dict_string)
                with self.profiler.start_time_profile("time_compress_with_lz77_sequences"):
                    comp_data = zstd_compress_with_lz77_sequences(data, lz77_seqs, dict_string)
            else:
                with self.profiler.start_time_profile("time_compress_extract_and_compress_with_lz77_sequences"):
                    comp_data = zstd_extract_and_compress_with_lz77_sequences(data, dict_string)
        else:
            with self.profiler.start_time_profile("time_compress_wo_sequences"):
                # comp_data = zstd_compress(data, dict_string)
                comp_data = self.zstd_wrapper.compress_once(data)
        return comp_data

    # def decompress(self, data: bytes, *args, **kwargs):
    #     dict_string = self.get_parameters()
    #     return zstd_decompress(data, self.max_length, dict_string)

    def update_state(self, *args, **kwargs) -> None:
        self.zstd_wrapper.load_dictionary(self.get_parameters())


class ZstdGeneralCodec(GeneralCodec):
    def __init__(self, *args, 
        lz77_dict_preprocessor_config=dict(), 
        literals_entropy_coder_config=dict(), 
        offset_entropy_coder_config=dict(), 
        litlen_entropy_coder_config=dict(), 
        matchlen_entropy_coder_config=dict(), 
        max_length=1<<23,
        **kwargs):
        lz77_dict_preprocessor_config_default = dict(
            relative_offset_codes=3,
            relative_offset_mode="cache",
            dict_size=32*1024,  
        )
        lz77_dict_preprocessor_config_default.update(**lz77_dict_preprocessor_config)
        
        literals_entropy_coder_config_default = dict(
            coding_table=[i for i in range(256)],
            coding_extra_symbols=[1] * 256,
            max_bits=8,
            table_log=11,
            predcnt_table_log=11,
        )
        literals_entropy_coder_config_default.update(**literals_entropy_coder_config)
        
        offset_entropy_coder_config_default = dict(
            coding_table=[0, 0],
            coding_extra_symbols=[1 << bits for bits in range(31)],
            max_bits=31,
            # decoding_table_min=1,
            table_log=10,
            update_coding_table=True,
            update_coding_table_method="recursive_split",
            target_max_symbol=100,
            auto_adjust_max_symbol=False,
            force_log2_extra_code=True,
            predcnt_table_log=10,
            # num_predcnts=64,
        )
        offset_entropy_coder_config_default.update(**offset_entropy_coder_config)

        ll_coding_table = [ 0,  1,  2,  3,  4,  5,  6,  7,
                            8,  9, 10, 11, 12, 13, 14, 15,
                            16, 16, 17, 17, 18, 18, 19, 19,
                            20, 20, 20, 20, 21, 21, 21, 21,
                            22, 22, 22, 22, 22, 22, 22, 22,
                            23, 23, 23, 23, 23, 23, 23, 23,
                            24, 24, 24, 24, 24, 24, 24, 24,
                            24, 24, 24, 24, 24, 24, 24, 24]
        ll_coding_extra_bits = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 2, 2, 3, 3,
            4, 6, 7, 8, 9,10,11,12,
            13,14,15,16
        ]
        ll_coding_extra_symbols = [1 << bits for bits in ll_coding_extra_bits]

        litlen_entropy_coder_config_default = dict(
            coding_table=ll_coding_table,
            coding_extra_symbols=ll_coding_extra_symbols,
            max_bits=16,
            table_log=9,
        )
        litlen_entropy_coder_config_default.update(**litlen_entropy_coder_config)
        
        ml_coding_table = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                            32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
                            38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
                            40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                            41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 ]
        ml_coding_extra_bits = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 2, 2, 3, 3,
            4, 4, 5, 7, 8, 9,10,11,
            12,13,14,15,16
        ]
        ml_coding_extra_symbols = [1 << bits for bits in ml_coding_extra_bits]
        matchlen_entropy_coder_config_default = dict(
            coding_table=ml_coding_table,
            coding_extra_symbols=ml_coding_extra_symbols,
            max_bits=16,
            table_log=9,
            min_value=3,
            decoding_table_min=3,
        )
        matchlen_entropy_coder_config_default.update(**matchlen_entropy_coder_config)
        
        preprocessor = LZ77DictPreprocessor(**lz77_dict_preprocessor_config_default)
        entropy_coder = GroupedEntropyCoder(
            [
                TrainablePredCntTANSEntropyCoder(**literals_entropy_coder_config_default),
                TrainablePredCntTANSEntropyCoder(**offset_entropy_coder_config_default),
                TrainablePredCntTANSEntropyCoder(**litlen_entropy_coder_config_default),
                TrainablePredCntTANSEntropyCoder(**matchlen_entropy_coder_config_default),
            ]
        )
        
        super().__init__(*args, 
            preprocessor=preprocessor, 
            entropy_coder=entropy_coder, 
        **kwargs)

        self.max_length = max_length

        self.zstd_wrapper = ZstdWrapper()
        self.zstd_custom_dict = None

    def generate_zstd_custom_dict(self, dict_id=0):
        dict_bytes_literals = self.entropy_coder.entropy_coders[0].export_zstd_custom_dict(is_huf=True)
        dict_bytes_offset = self.entropy_coder.entropy_coders[1].export_zstd_custom_dict(is_huf=False)
        dict_bytes_litlen = self.entropy_coder.entropy_coders[2].export_zstd_custom_dict(is_huf=False)
        dict_bytes_matchlen = self.entropy_coder.entropy_coders[3].export_zstd_custom_dict(is_huf=False)
        byte_strings = [
            # magic number
            struct.pack('<L', 0xED30A437),
            # dict ID
            struct.pack('<L', dict_id),
            dict_bytes_literals,
            dict_bytes_offset,
            dict_bytes_matchlen,
            dict_bytes_litlen,
            # struct.pack('B', 0),
            # struct.pack('B', 0),
            # struct.pack('B', 0),
            # rep codes (TODO: obtain from lz77 dict)
            # struct.pack('<L', 1),
            # struct.pack('<L', 4),
            # struct.pack('<L', 8),
            # zstd dict (omit 8 bytes header)
            self.preprocessor.compression_dict[8:]
        ]

        self.zstd_custom_dict = b''.join(byte_strings)
        self.zstd_wrapper.load_dictionary(self.zstd_custom_dict)

    def train_full(self, dataloader, *args, **kwargs) -> None:
        super().train_full(dataloader, *args, **kwargs)
        self.generate_zstd_custom_dict()

    def load_parameters(self, parameters: Dict[str, Any], *args, **kwargs) -> None:
        super().load_parameters(parameters, *args, **kwargs)
        self.generate_zstd_custom_dict()

    def update_state(self, *args, **kwargs) -> None:
        super().update_state(*args, **kwargs)
        self.generate_zstd_custom_dict()

    def compress(self, data, *args, **kwargs) -> bytes:
        # if self.zstd_custom_dict is None:
        #     comp_data = super().compress(data, *args, **kwargs)
        # else:
        #     with self.profiler.start_time_profile("time_compress"):
        #         comp_data = zstd_compress(data, self.zstd_custom_dict)
        comp_data = self.zstd_wrapper.compress_once(data)
        return comp_data

    def decompress(self, data: bytes, *args, **kwargs):
        # if self.zstd_custom_dict is None:
        #     return super().decompress(data, *args, **kwargs)
        # else:
        #     return zstd_decompress(data, self.max_length, self.zstd_custom_dict)
        return self.zstd_wrapper.decompress_once(data, self.max_length)