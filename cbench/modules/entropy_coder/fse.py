from locale import normalize
from typing import Any, Dict, List, Union
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans, whiten
from scipy.spatial.distance import pdist

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EntropyCoder
from .utils import BinaryHeadConstructor, merge_bytes, split_merged_bytes, estimate_entropy_coding_length, estimate_entropy_coding_length_by_cnt
from .tans_utils import estimate_coding_table_total_entropy, export_zstd_custom_dict, generate_tans_coding_table, tans_code_to_data, tans_data_to_code

from cbench.modules.base import TrainableModuleInterface
from cbench.utils.logging_utils import MetricLogger
from cbench.zstd_wrapper import fse_compress, fse_decompress, huf_compress, huf_decompress, fse_tans_compress, fse_tans_decompress, fse_tans_compress_advanced, fse_tans_decompress_advanced


class FSEEntropyCoder(EntropyCoder):
    CODER_FSE = "fse"
    CODER_HUF = "huffman"
    STRING_LIMIT = 65535
    
    def __init__(self, *args, coder="fse", max_symbol=255, table_log=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.coder = coder
        if coder == self.CODER_FSE:
            self.encoder = fse_compress
            self.decoder = fse_decompress
        elif coder == self.CODER_HUF:
            self.encoder = huf_compress
            self.decoder = huf_decompress
        else:
            raise ValueError()
        self.max_symbol = max_symbol
        self.table_log = table_log
        # encode as 16bit
        self.use_fse_16bit = max_symbol > 255
        self.max_symbol_lowbit = max_symbol & 0x00ff
        self.max_symbol_highbit = max_symbol & 0xff00
        assert(max_symbol < 65536) # TODO: higher bit?

    def encode(self, data: Union[bytes, list, np.ndarray] , *args, prior=None, **kwargs):
        # TODO: warn about data loss if input data is float!
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            if data.dtype != np.long or len(data.shape) != 1:
                raise ValueError("Cannot encode non-long-vector array!")
            if self.use_fse_16bit:
                data_lowbit = np.bitwise_and(data, 0x00ff).astype(np.uint8).tobytes()
                data_highbit = np.bitwise_and(data, 0xff00).astype(np.uint8).tobytes()
                data = (data_lowbit, data_highbit)
        assert(len(data) < self.STRING_LIMIT), "exceed string limit" # limit of data
        
        if self.use_fse_16bit:
            # orig_len = len(data[0]) + len(data[1])
            byte_string_lowbit = self.encoder(data[0], maxSymbolValue=self.max_symbol_lowbit, tableLog=self.table_log)
            byte_string_highbit = self.encoder(data[1], maxSymbolValue=self.max_symbol_highbit, tableLog=self.table_log)
            byte_string = merge_bytes([byte_string_lowbit, byte_string_highbit], num_segments=2)
        else:
            # orig_len = len(data)
            byte_string = self.encoder(data, maxSymbolValue=self.max_symbol, tableLog=self.table_log)
        # TODO: deal with overflow!
        # compress error
        if len(byte_string) == 0: # or len(byte_string) >= orig_len: 
            byte_string = b'\x00' + data
        else:
            byte_string = b'\xff' + byte_string
        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        if byte_string[0] == 0:
            return byte_string[1:]
        else:
            return self.decoder(byte_string[1:], self.STRING_LIMIT)


class TANSEntropyCoder(EntropyCoder):
    STRING_LIMIT = 65535

    def __init__(self, *args, 
                 table_distribution : Union[np.ndarray, None] = None, 
                 coding_table=None, 
                 coding_extra_symbols=None, 
                 max_bits=31,
                 max_symbol=255, # deprecated! will be overwritten!
                 table_log=12,
                 min_value=0, # a workaround for zstd matchLength
                 decoding_table_min=0, 
                 **kwargs
        ):
        super().__init__(*args, **kwargs)
        # convert table_distribution
        if table_distribution is not None:
            coding_table, coding_extra_symbols, max_symbol = \
                generate_tans_coding_table(table_distribution, max_symbol=max_symbol, max_bits=max_bits)
        
        if coding_table is not None and coding_extra_symbols is not None:
            # TODO: maybe check if coding_table is valid
            self.coding_table = np.array(coding_table)
            self.coding_extra_symbols = np.array(coding_extra_symbols)
            max_symbol = len(self.coding_extra_symbols) - 1
        else:
            # use default coding table
            self.coding_table = np.array([0])
            self.coding_extra_symbols = np.array([1 << bits for bits in range(max_bits+1)])
            max_symbol = len(self.coding_extra_symbols) - 1
            # raise ValueError("Either table_distribution or both [coding_table, coding_extra_symbols] should be provided!")
        
        assert(max_symbol < 256)
        self.decoding_table = np.zeros(len(coding_table), dtype=np.int32)
        self.decoding_table[self.coding_table] = np.arange(len(coding_table)) + decoding_table_min
        self.max_symbol = max_symbol
        self.max_bits = max_bits
        self.table_log = table_log
        self.min_value = min_value
        self.decoding_table_min = decoding_table_min

        self.profiler = MetricLogger()
        self.encoder = self._fse_tans_compress
        self.decoder = self._fse_tans_decompress
        # self.encoder = lambda table_code, extra_code: fse_tans_compress(table_code, extra_code, 
        #     self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log,
        # )
        # self.decoder = lambda byte_string: fse_tans_decompress(byte_string, 
        #     self.coding_extra_symbols, self.STRING_LIMIT, maxSymbolValue=self.max_symbol, tableLog=self.table_log,
        # ) # current limit is 64KB!
        
        self._total_estimated_entropy = 0
    
    def _fse_tans_compress(self, table_code, extra_code):
        byte_string = fse_tans_compress_advanced(table_code, extra_code, 
            self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log)
        if len(byte_string) == 0: # direct store mode
            byte_string = b'\x00' + \
                fse_tans_compress_advanced(table_code, extra_code, 
                self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log, predefined_count=np.ones(self.max_symbol+1))
        else:
            byte_string = b'\xff' + byte_string
        return byte_string

    def _fse_tans_decompress(self, byte_string):
        if byte_string[0] == 0:
            return fse_tans_decompress_advanced(byte_string[1:], 
                self.coding_extra_symbols, self.STRING_LIMIT, 
                maxSymbolValue=self.max_symbol, tableLog=self.table_log, 
                predefined_count=np.ones(self.max_symbol+1))
        else:
            return fse_tans_decompress_advanced(byte_string[1:], 
                self.coding_extra_symbols, self.STRING_LIMIT, 
                maxSymbolValue=self.max_symbol, tableLog=self.table_log)

    def encode(self, data : np.ndarray, *args, prior=None, **kwargs):
        assert(data.size < self.STRING_LIMIT / (self.max_bits+1) * 8), "exceed string limit" # limit of data
        # data_cnt = np.ones(max(data.max()+1, len(self.coding_table)))
        # np.add.at(data_cnt, data, 1)
        # normalized_pdf = data_cnt / data_cnt.sum(-1, keepdims=True)
        # self._total_estimated_entropy += estimate_coding_table_total_entropy(normalized_pdf, self.coding_table)
        # print(f"Estimated coding entropy: current:{estimate_coding_table_total_entropy(normalized_pdf, self.coding_table)} total:{self._total_estimated_entropy}")
        # with self.profiler.start_time_profile("time_tans_data_to_code"):
        table_code, extra_code = tans_data_to_code(data - self.min_value, self.coding_table, decoding_table=self.decoding_table)
        # convert table_code to bytes to process by fse (maybe check table_code range 0-255?)
        table_code = table_code.astype(np.uint8).tobytes()
        # with self.profiler.start_time_profile("time_fse_tans_encoder"):
        byte_string = self.encoder(table_code, extra_code)
        # print(self.profiler)
        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        table_code, extra_code = self.decoder(byte_string) 
        # convert table_code from bytes
        table_code = np.frombuffer(table_code, dtype=np.uint8).astype(np.int32)
        data = tans_code_to_data(table_code, extra_code, self.coding_table, decoding_table=self.decoding_table)
        return data


class TrainablePredCntTANSEntropyCoder(TANSEntropyCoder, TrainableModuleInterface):
    def __init__(self, *args, 
        initial_distribution=None, 
        max_dist=1<<20, 
        update_coding_table=False,
        update_coding_table_method="recursive_split",
        auto_adjust_max_symbol=True,
        target_max_symbol=255,
        force_log2_extra_code=False,
        num_predcnts=1,
        predcnt_table_log=8,
        **kwargs):
        super().__init__(*args, 
            **kwargs
        )
        # limit max distribution
        max_dist = min(1 << self.max_bits, max_dist)
        self.max_dist = max_dist
        self.update_coding_table = update_coding_table
        self.update_coding_table_method = update_coding_table_method
        self.auto_adjust_max_symbol = auto_adjust_max_symbol
        self.target_max_symbol = target_max_symbol
        self.force_log2_extra_code = force_log2_extra_code

        self.num_predcnts = num_predcnts
        self.predcnt_table_log = predcnt_table_log

        # predefined_counts: used to reduce overhead when distribution of data is similar to predefined_counts
        self.predefined_counts = np.ones((self.num_predcnts, self.max_symbol+1), dtype=np.int32) # [ [1]*(self.max_symbol+1) ] * self.num_predcnts # start from 1 to avoid error

        self.predefined_distribution = np.zeros(max_dist - self.min_value)
        self.samples = []

        # copy predefined distribution
        assert (isinstance(initial_distribution, np.ndarray) and initial_distribution.ndim == 1) or initial_distribution is None
        if initial_distribution is not None:
            self.predefined_distribution[:initial_distribution.shape[0]] = initial_distribution

        # self.encoder = lambda table_code, extra_code: fse_tans_compress(table_code, extra_code, 
        #     self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log,
        #     predefined_counts=self.predefined_counts
        # )
        # self.decoder = lambda byte_string: fse_tans_decompress(byte_string, 
        #     self.coding_extra_symbols, self.STRING_LIMIT, maxSymbolValue=self.max_symbol, tableLog=self.table_log,
        #     predefined_counts=self.predefined_counts
        # ) # current limit is 64KB!

        self._total_estimated_length = 0
        self._total_fact_length = 0
        self._total_symbols = 0
        self._total_batch = 0
    
    def _fse_tans_compress(self, table_code, extra_code):
        table_code_cnt = np.zeros(self.max_symbol+1)
        np.add.at(table_code_cnt, np.frombuffer(table_code, dtype=np.uint8), 1)
        distributions = np.concatenate([table_code_cnt[None], np.array(self.predefined_counts)], axis=0)
        distributions_pdf = distributions / distributions.sum(1, keepdims=True)
        # estimate coding length with cross entropy
        estimated_coding_length = (table_code_cnt * -np.log2(distributions_pdf + 1e-8)).sum(1) / 8
        estimated_coding_length[0] += (self.max_symbol+1) * (1 << self.table_log) / 8 # add fse table length
        # estimate the best coding mode
        best_mode = estimated_coding_length.argmin()

        # byte_string = fse_tans_compress_advanced(table_code, extra_code, 
        #     self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log)
        
        if best_mode == 0:
            byte_string = fse_tans_compress_advanced(table_code, extra_code, 
                self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log)
        else:
            byte_string = fse_tans_compress_advanced(table_code, extra_code, 
                self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log, predefined_count=self.predefined_counts[best_mode-1])

        if len(byte_string) == 0:
            best_mode = len(self.predefined_counts) + 1
            byte_string = fse_tans_compress_advanced(table_code, extra_code, 
                self.coding_extra_symbols, maxSymbolValue=self.max_symbol, tableLog=self.table_log, predefined_count=np.ones(self.max_symbol+1))

        # estimated_extra_code_length = np.ceil(np.log2(self.coding_extra_symbols[np.frombuffer(table_code, dtype=np.uint8)])).sum() / 8
        # self._total_estimated_length += estimated_coding_length[best_mode]+estimated_extra_code_length
        # self._total_fact_length += len(byte_string)
        # # print(f"Estimated table code length: current:{estimated_coding_length[best_mode]}")
        # print(f"Estimated coding length: current:{estimated_coding_length[best_mode]+estimated_extra_code_length} total:{self._total_estimated_length}")
        # print(f"True coding length: current:{len(byte_string)} total:{self._total_fact_length}")

        header = BinaryHeadConstructor(coding_mode=(len(self.predefined_counts) + 2))
        header.set_data(coding_mode=best_mode)
        byte_string = header.get_bytes() + byte_string
        
        return byte_string

    def _fse_tans_decompress(self, byte_string):
        header = BinaryHeadConstructor(coding_mode=(len(self.predefined_counts) + 2))
        header_length = header.get_max_bytes()
        header_data = header.get_data_from_bytes(byte_string[:header_length])
        coding_mode = header_data['coding_mode']
        if coding_mode == len(self.predefined_counts) + 1:
            return fse_tans_decompress_advanced(byte_string[header_length:], 
                self.coding_extra_symbols, self.STRING_LIMIT, 
                maxSymbolValue=self.max_symbol, tableLog=self.table_log, 
                predefined_count=np.ones(self.max_symbol+1))
        elif coding_mode > 0:
            return fse_tans_decompress_advanced(byte_string[header_length:], 
                self.coding_extra_symbols, self.STRING_LIMIT, 
                maxSymbolValue=self.max_symbol, tableLog=self.table_log, 
                predefined_count=self.predefined_counts[coding_mode-1])
        else:
            return fse_tans_decompress_advanced(byte_string[header_length:], 
                self.coding_extra_symbols, self.STRING_LIMIT, 
                maxSymbolValue=self.max_symbol, tableLog=self.table_log)

    @property
    def parameter_names(self) -> List[str]:
        return [
            "coding_table",
            "coding_extra_symbols",
            "max_symbol",
            "decoding_table",
            "predefined_counts",
        ]

    def get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
        # return self.predefined_distribution
        # return dict(
        #     coding_table=self.coding_table,
        #     coding_extra_symbols=self.coding_extra_symbols,
        #     max_symbol=self.max_symbol,
        #     decoding_table=self.decoding_table,
        #     predefined_counts=self.predefined_counts,
        # )
        return {name : getattr(self, name) for name in self.parameter_names}

    def load_parameters(self, parameters: Dict[str, Any], *args, **kwargs) -> None:
        # self.predefined_distribution = parameters
        # self.update_state(*args, **kwargs)
        for name, value in parameters.items():
            setattr(self, name, value)

    def train_full(self, dataloader, *args, **kwargs) -> None:
        for data in dataloader:
            self.train_iter(data, *args, **kwargs)
        # print(self.predefined_counts)

    def train_iter(self, data, *args, **kwargs) -> None:
        self.samples.append(data)
        # indexed add
        np.add.at(self.predefined_distribution, data - self.min_value, 1)
        # for symbol in data:
        #     self.predefined_distribution[symbol] += 1
        # TODO: limit maximum of predefined_counts
        # table_code, extra_code = tans_data_to_code(data - self.min_value, self.coding_table)
        # for code in table_code:
        #     self.predefined_counts[0][code] += 1

    def update_state(self, *args, **kwargs) -> None:
        # trim predefined_distribution at first non-zero from last one and max_dist
        last_nonzero = len(self.predefined_distribution) - np.argmax(np.flip(self.predefined_distribution) > 0)
        self.max_dist = last_nonzero
        self.predefined_distribution = self.predefined_distribution[:last_nonzero]
        
        # update coding table with predefined_distribution
        if self.update_coding_table:
            # gaussian_kernel = np.exp(-0.5 * np.square(np.linspace(-6, 6, 1)) / np.square(1))
            # coding_distribution = np.convolve(
            #     self.predefined_distribution, 
            #     # gaussian kernel
            #     gaussian_kernel,
            #     mode='same',
            # ).clip(1)
            coding_distribution = self.predefined_distribution #.clip(1)
            coding_table, coding_extra_symbols, max_symbol = \
                generate_tans_coding_table(
                    coding_distribution, # minimum freq 1
                    auto_adjust_max_symbol=self.auto_adjust_max_symbol,
                    max_symbol=self.target_max_symbol, 
                    max_bits=self.max_bits,
                    method=self.update_coding_table_method,
                    # table_log=self.predcnt_table_log,
                    force_log2_extra_code=self.force_log2_extra_code,
                )
            
            # estimate improvement
            normalized_pdf = coding_distribution.astype(np.float) / coding_distribution.sum()
            estimated_entropy_old = estimate_coding_table_total_entropy(self.predefined_distribution, self.coding_table, 
                normalized_pdf=normalized_pdf, 
                table_log=self.predcnt_table_log, 
                force_log2_extra_code=self.force_log2_extra_code
            )
            estimated_entropy_new = estimate_coding_table_total_entropy(self.predefined_distribution, coding_table, 
                normalized_pdf=normalized_pdf, 
                table_log=self.predcnt_table_log, 
                force_log2_extra_code=self.force_log2_extra_code
            )
            print("Improved Entropy: {} -> {}".format(estimated_entropy_old, estimated_entropy_new))

            self.coding_table = np.array(coding_table)
            self.coding_extra_symbols = np.array(coding_extra_symbols)
            self.max_symbol = len(coding_extra_symbols) - 1
            self.decoding_table = np.zeros(len(coding_table), dtype=np.int32)
            self.decoding_table[self.coding_table] = np.arange(len(coding_table)) + self.decoding_table_min

        # update predefined_counts
        predefined_counts = np.zeros((self.num_predcnts, self.max_symbol+1)) # start from 1 to avoid error
        # estimate entropy

        if self.num_predcnts > 1:
            total_entropy_min = 0
            total_entropy_predcnt = 0

            # total_entropy_whole
            whole_data_cnts = np.zeros(self.max_symbol+1)
            table_code, _ = tans_data_to_code(np.arange(len(self.predefined_distribution)), self.coding_table, decoding_table=self.decoding_table)
            np.add.at(whole_data_cnts, table_code, self.predefined_distribution)
            total_entropy_whole = estimate_entropy_coding_length_by_cnt(whole_data_cnts)

            # samples to dists
            sample_dists = []
            for data in self.samples:
                if len(data) == 0: continue
                sample_cnts = np.zeros(self.max_symbol+1)
                table_code, _ = tans_data_to_code(data - self.min_value, self.coding_table, decoding_table=self.decoding_table)
                np.add.at(sample_cnts, table_code, 1)
                sample_dists.append(sample_cnts)
                # estimate entropy for evalutation
                total_entropy_min += estimate_entropy_coding_length_by_cnt(sample_cnts)

            if len(sample_dists) == 0:
                return
                
            # k-means clustering
            # obs = whiten(sample_dists)
            # centroids, _ = kmeans(obs, self.num_predcnts)
            # self.predefined_counts = (centroids / centroids.sum(-1, keepdims=True) * (1 << self.predcnt_table_log)+1).astype(np.int32).tolist()

            # Agglomerative Clustering
            # dist_matrix = pdist(sample_dists, metric="jensenshannon") 
            # NOTE: is jensenshannon suitable for entropy distance?
            Z = linkage(sample_dists, 
                method='average', 
                metric="jensenshannon"
            )
            cluster = fcluster(Z, t=self.num_predcnts, criterion='maxclust')

            for predcnt_idx in range(self.num_predcnts):
                sample_idxs = np.where(cluster==(predcnt_idx+1))[0]
                print("{} samples clustered for predcnt {}".format(len(sample_idxs), predcnt_idx))
                for si in sample_idxs:
                    table_code, _ = tans_data_to_code(self.samples[si] - self.min_value, self.coding_table, decoding_table=self.decoding_table)
                    np.add.at(predefined_counts[predcnt_idx], table_code, 1)

            # Adam based clustering
            # param = torch.nn.Parameter(
            #     torch.rand(self.num_predcnts, self.max_symbol+1)
            # )
            # nn.init.normal_(param)
            # optimizer = torch.optim.Adam([param], lr=2e-3)
            # epoches = 50
            # loss_batch_size = 64
            # for epoch_idx in range(epoches):
            #     loss_buffer = []
            #     for data in self.samples:
            #         table_code, _ = tans_data_to_code(data - self.min_value, self.coding_table)
            #         optimizer.zero_grad()
            #         prior_dist = param.repeat(len(data), 1)
            #         target = torch.as_tensor(table_code).repeat(self.num_predcnts)
            #         loss = F.cross_entropy(prior_dist, target, reduction="none").view(self.num_predcnts, len(data)).sum(-1)
            #         loss_min, _ = torch.min(loss, 0)
            #         loss_buffer.append(loss_min)
            #         if len(loss_buffer) >= loss_batch_size:
            #             sum(loss_buffer).backward()
            #             loss_buffer = []
            #         optimizer.step()
            
            #     predefined_counts = torch.softmax(param, -1).detach().numpy()

            #     # estimate entropy with predcnt
            #     total_entropy_predcnt = 0
            #     for data in self.samples:
            #         table_code, _ = tans_data_to_code(data - self.min_value, self.coding_table)
            #         total_entropy_predcnt += estimate_entropy_coding_length(table_code, distribution=predefined_counts)

            #     print("Epoch {} end, current entropy is {}".format(epoch_idx, total_entropy_predcnt))
                
            total_entropy_predcnt = 0
            for data in self.samples:
                table_code, _ = tans_data_to_code(data - self.min_value, self.coding_table, decoding_table=self.decoding_table)
                total_entropy_predcnt += estimate_entropy_coding_length(table_code, distribution=predefined_counts)

            print("Predcnt Entropy {} -> {}, minimum is {}".format(total_entropy_whole, total_entropy_predcnt, total_entropy_min))

        # faster when num_predcnt==1
        else:
            table_code, _ = tans_data_to_code(np.arange(len(self.predefined_distribution)), self.coding_table)
            np.add.at(predefined_counts[0], table_code, self.predefined_distribution)
            # for code, cnt in zip(table_code, self.predefined_distribution):
            #     self.predefined_counts[0][code] += int(cnt)
            # for i, cnt in enumerate(self.predefined_distribution):
            #     code = self.coding_table[i]
            #     self.predefined_counts[0][code] += int(cnt)
        
        # post process with table log
        predefined_counts_norm = (predefined_counts / predefined_counts.sum(-1, keepdims=True) * (1 << self.predcnt_table_log))
        self.predefined_counts = predefined_counts_norm.clip(1).astype(np.int32) #.clip(1)? could use lowprob?
        # TODO: discard same counts
        # self.predefined_counts = (predefined_counts+1).astype(np.int32).tolist()
        # print(self.coding_table, self.coding_extra_symbols, self.predefined_counts)
        
        self._total_estimated_length = 0
        self._total_fact_length = 0

    def export_zstd_custom_dict(self, is_huf=False):
        # TODO: export all predefined_counts
        return export_zstd_custom_dict(
            self.coding_table, 
            self.coding_extra_symbols, 
            table_log=self.table_log,
            # decoding_table=self.decoding_table, # this decoding_table is different, just regenerate!
            decoding_table_min=self.decoding_table_min,
            num_cnts=self.predefined_counts,
            cnt_table_log=self.predcnt_table_log,
            is_huf=is_huf,
        )