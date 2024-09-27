import math
import struct
import numpy as np
from scipy.interpolate import UnivariateSpline

def _highbits(val, eps=1e-6):
    # return (np.ceil(np.log2(np.maximum(val, 1) + eps))).astype(np.uint32)
    return (np.ceil(np.log2(np.maximum(val + eps, 1)))).astype(np.uint32)

def _equalize_cdf(
        distribution: np.ndarray, 
        max_symbol=255, 
        # allow_symbol_reordering=False,
        force_log2_extra_code=False, # TODO:
        **kwargs,
    ):
    # 1. normalize distribution and get cdf
    # if allow_symbol_reordering:
    #     coding_order = (-distribution).argsort() # descending order
    #     distribution = distribution[coding_order]
    pdf = distribution.astype(np.float) / distribution.sum()
    cdf = np.cumsum(np.concatenate([np.array([0]), pdf]))
    
    # 2. we may adjust max_symbol to fit the maximum of pdf (recommended!)
    max_symbol_adjust = min(max_symbol, int(np.ceil(1 / pdf.max())))
    # max_symbol_normal = max_symbol - overflow_symbols
    # if auto_adjust_max_symbol:
    #     max_symbol_adjust = int(np.ceil(1 / pdf.max()))
    #     if max_symbol_adjust < max_symbol_normal:
    #         max_symbol_normal = max_symbol_adjust
    #         max_symbol = max_symbol_adjust + overflow_symbols

    # 3. equalize each symbols with cdf
    if force_log2_extra_code:
        # TODO: coding_extra_symbols should be monotically increase and with all 2^N values
        raise NotImplementedError()
    else:
        cdf_split_values = np.arange(1, max_symbol_adjust+1) / (max_symbol_adjust + 1)
        coding_table = np.digitize(cdf, cdf_split_values)
        # coding_extra_symbols = np.bincount(coding_table)
        # coding_extra_table = np.zeros(len(coding_table), dtype=np.int32)
    
    # if allow_symbol_reordering:
    #     coding_table = coding_table[coding_order]
    return coding_table

def _estimate_coding_group_entropy(distribution_group, occurance_group=None, table_log=None, force_log2_extra_code=False):
    if occurance_group is None:
        occurance_group = distribution_group
    occurance_sum = occurance_group.sum()
    # avoid math domain error
    if occurance_sum == 0:
        return 0
    extra_code_entropy = math.log2(len(distribution_group))
    if force_log2_extra_code:
        extra_code_entropy = math.ceil(extra_code_entropy)
    distribution_sum = distribution_group.sum()
    if table_log is not None:
        table_log_norm = 1<<table_log
        distribution_sum = max(1, round(distribution_sum * table_log_norm)) / table_log_norm
    table_code_entropy = math.log2(distribution_sum)

    return occurance_sum * \
        (extra_code_entropy - table_code_entropy)

def _recursive_split(
        distribution: np.ndarray, 
        max_symbol=255, 
        table_log=None,
        force_log2_extra_code=False,
        **kwargs,
    ):
    normalized_pdf = distribution.astype(np.float) / distribution.sum()

    # def _recursive_split_iter(
    #         normalized_pdf: np.ndarray, 
    #         max_symbol=255, 
    #         current_split=None,
    #         **kwargs,
    #     ):

    #     if current_split is None:
    #         current_split = [0, len(normalized_pdf)-1]

    #     all_splits = []
    #     for i in range(len(current_split) - 1):
    #         try_split_idx = (current_split[i] + current_split[i+1]) // 2
    #         if try_split_idx == current_split[i] or try_split_idx == current_split[i+1]:
    #             continue
    #         s1, s2 = current_split[i], try_split_idx
    #         e1, e2 = try_split_idx, current_split[i+1]
    #         ce_merge = _estimate_coding_group_entropy(normalized_pdf[s1:e2])
    #         ce_split = _estimate_coding_group_entropy(normalized_pdf[s1:e1]) + _estimate_coding_group_entropy(normalized_pdf[s2:e2])
    #         if ce_merge > ce_split:
    #             all_splits.extend(_recursive_split_iter(normalized_pdf, max_symbol, [s1, e1]))
    #             all_splits.extend(_recursive_split_iter(normalized_pdf, max_symbol, [s2, e2]))
    #         else:
    #             all_splits = current_split

    #     return all_splits

    # distribution_splits = _recursive_split_iter(normalized_pdf, max_symbol)
    
    # initialize the whole range as split, and then split this range iteratively
    distribution_splits = [0, len(normalized_pdf)-1]
    while (len(distribution_splits) <= max_symbol):
        current_split = distribution_splits
        new_splits = []
        # split_scores = []
        split_idx_score = dict()
        for i in range(len(current_split) - 1):
            current_split_length = current_split[i+1] - current_split[i]
            if force_log2_extra_code:
                # for log2 segments, only try its half split
                if math.floor(math.log2(current_split_length)) == math.log2(current_split_length):
                    try_split_idxs = [current_split[i] + current_split_length // 2]
                # for arbitrary segments (commonly only 1 such segment), try all possible log2 splits
                else:
                    log2_idxs = 1 << np.arange(math.floor(round(math.log2(current_split_length))))
                    split_idxs_l = current_split[i] + log2_idxs
                    split_idxs_r = current_split[i+1] - log2_idxs
                    try_split_idxs = np.unique(split_idxs_l.tolist() + split_idxs_r.tolist())
            else:
                # try all nonzero splits
                # try_split_idxs = current_split[i] + 1 + np.array(np.nonzero(distribution[(current_split[i]+1):current_split[i+1]] != 0))[0] # np.arange(current_split[i]+1, current_split[i+1])
                # try only mid point
                try_split_idxs = [(current_split[i] + current_split[i+1]) // 2]
            for try_split_idx in try_split_idxs:
                if try_split_idx == current_split[i] or try_split_idx == current_split[i+1]:
                    continue
                s1, s2 = current_split[i], try_split_idx
                e1, e2 = try_split_idx, current_split[i+1]
                ce_merge = _estimate_coding_group_entropy(normalized_pdf[s1:e2], distribution[s1:e2], table_log=table_log, force_log2_extra_code=force_log2_extra_code)
                ce_split = _estimate_coding_group_entropy(normalized_pdf[s1:e1], distribution[s1:e1], table_log=table_log, force_log2_extra_code=force_log2_extra_code) + \
                    _estimate_coding_group_entropy(normalized_pdf[s2:e2], distribution[s2:e2], table_log=table_log, force_log2_extra_code=force_log2_extra_code)
                # if ce_merge > ce_split:
                #     new_splits.extend([s1, s2])
                # else:
                #     new_splits.append(s1)
                score = ce_merge - ce_split
                assert(try_split_idx not in split_idx_score)
                split_idx_score[try_split_idx] = score
            # split_scores.append(ce_merge - ce_split)
        best_split_idx = max(split_idx_score, key=split_idx_score.get)
        new_splits = sorted(current_split + [best_split_idx])
        # new_splits.extend(current_split[:(best_split_idx+1)])
        # new_splits.append((current_split[best_split_idx] + current_split[best_split_idx+1]) // 2)
        # new_splits.extend(current_split[(best_split_idx+1):])
        # new_splits.append(current_split[-1])
        if len(new_splits) <= max_symbol:
            distribution_splits = new_splits
        else:
            break
        
        print("Current splits: {}, Entropy: {}".format(
            len(distribution_splits),
            estimate_distribution_split_entropy(normalized_pdf, distribution_splits, table_log=table_log, force_log2_extra_code=force_log2_extra_code)
        ))

    # build coding table
    max_symbol_adjust = len(distribution_splits) - 1
    assert(max_symbol_adjust < 256)
    coding_table = np.zeros(len(distribution), dtype=np.int32)
    for i in range(len(distribution_splits) - 1):
        coding_table[distribution_splits[i]:distribution_splits[i+1]] = i
    coding_table[distribution_splits[-1]:] = len(distribution_splits) - 1
    return coding_table

def _recursive_merge(
        distribution: np.ndarray, 
        max_symbol=255, 
        merge_threshold=0.0,
        merge_threshold_adjust_step=0.1,
        table_log=None,
        force_log2_extra_code=False,
        **kwargs,
    ):
    normalized_pdf = distribution.astype(np.float) / distribution.sum()
    distribution_splits = np.arange(len(distribution)+1)
    merge_threshold_adjust = merge_threshold
    while (len(distribution_splits) >= min(max_symbol, 256)):
        print("Remaining splits: {}, Entropy: {}".format(
            len(distribution_splits),
            estimate_distribution_split_entropy(normalized_pdf, distribution_splits, table_log=table_log, force_log2_extra_code=force_log2_extra_code)
        ))

        merge_groups = []
        merge_scores = []
        split_scores = []
        
        # 2 division iteration
        for i in range(len(distribution_splits)-2):
            s1, s2 = distribution_splits[i], distribution_splits[i+1]
            e1, e2 = distribution_splits[i+1], distribution_splits[i+2]
            ce_merge = _estimate_coding_group_entropy(normalized_pdf[s1:e2], distribution[s1:e2], table_log=table_log, force_log2_extra_code=force_log2_extra_code)
            ce_split1 = _estimate_coding_group_entropy(normalized_pdf[s1:e1], distribution[s1:e1], table_log=table_log, force_log2_extra_code=force_log2_extra_code)
            ce_split2 = _estimate_coding_group_entropy(normalized_pdf[s2:e2], distribution[s2:e2], table_log=table_log, force_log2_extra_code=force_log2_extra_code)
            # a lossless merge
            if ce_merge - ce_split1 - ce_split2 <= merge_threshold_adjust:
                merge_groups.append(i+1)
            merge_scores.append(ce_split1 + ce_split2 - ce_merge)
            # merge_scores.append(ce_merge)
            # if i==0:
            #     split_scores.append(ce_split1)
            # split_scores.append(ce_split2)
        # faster optimization: nothing to merge, update merge_threshold_adjust
        if len(merge_groups) == 0:
            merge_threshold_adjust += merge_threshold_adjust_step
        
        # TODO: faster optimization: allow topk merging
        best_merge_idx = np.argmax(merge_scores)
        merge_groups.append(best_merge_idx+1)
        distribution_splits = np.delete(distribution_splits, merge_groups)
        # print(distribution_splits)

    # build coding table
    max_symbol_adjust = len(distribution_splits) - 1
    assert(max_symbol_adjust < 256)
    coding_table = np.zeros(len(distribution), dtype=np.int32)
    for i in range(len(distribution_splits) - 1):
        coding_table[distribution_splits[i]:distribution_splits[i+1]] = i
    # coding_table[distribution_splits[-1]:] = len(distribution_splits) - 1
    return coding_table

def _recursive_smoothing(
        distribution: np.ndarray, 
        max_symbol=255, 
        initial_smoothing_factor=10,
        iterate_smoothing_factor=5,
        table_log=None,
        force_log2_extra_code=False,
        **kwargs,
    ):
    normalized_pdf = distribution.astype(np.float) / distribution.sum()
    distribution_cdf = np.cumsum(np.concatenate([np.array([0]), distribution]))
    smoothing_factor = len(distribution)*initial_smoothing_factor
    smooth_cdf_spline = UnivariateSpline(
        np.arange(len(distribution)+1), distribution_cdf,
        k=1,
        s=smoothing_factor,
    )
    distribution_splits = smooth_cdf_spline.get_knots()

    while len(distribution_splits) > max_symbol:
        smoothing_factor += len(distribution) * iterate_smoothing_factor
        # smooth_cdf_spline.set_smoothing_factor(smoothing_factor)
        smooth_cdf_spline = UnivariateSpline(
            np.arange(len(distribution)+1), distribution_cdf,
            k=1,
            s=smoothing_factor,
        )

        distribution_splits = smooth_cdf_spline.get_knots().astype(np.int32)
        print("Current splits: {}, Entropy: {}".format(
            len(distribution_splits),
            estimate_distribution_split_entropy(normalized_pdf, distribution_splits, table_log=table_log, force_log2_extra_code=force_log2_extra_code)
        ))
        
    coding_table = np.zeros(len(distribution), dtype=np.int32)
    for i in range(len(distribution_splits) - 1):
        coding_table[distribution_splits[i]:distribution_splits[i+1]] = i
    return coding_table


def estimate_distribution_split_entropy(normalized_pdf, distribution_splits, table_log=None, force_log2_extra_code=False):
    total_entropy = 0
    for i in range(len(distribution_splits) - 1):
        total_entropy += _estimate_coding_group_entropy(normalized_pdf[distribution_splits[i]:distribution_splits[i+1]], table_log=table_log, force_log2_extra_code=force_log2_extra_code)
    if distribution_splits[-1] < len(normalized_pdf):
        overflow_idxs = np.arange(distribution_splits[-1], len(normalized_pdf))
        total_entropy += (_highbits(overflow_idxs) * normalized_pdf[overflow_idxs]).sum()
    return total_entropy

def estimate_coding_table_total_entropy(distribution, coding_table, normalized_pdf=None, table_log=None, force_log2_extra_code=False):
    total_entropy = 0
    if normalized_pdf is None:
        normalized_pdf = distribution.astype(np.float) / distribution.sum()
    # append overflow coding table
    if len(coding_table) < len(normalized_pdf):
        overflow_value = len(coding_table)
        overflow_data = np.arange(len(coding_table), len(normalized_pdf))
        overflow_mincode = np.max(coding_table) + 1
        overflow_coding_table = overflow_mincode + _highbits(overflow_data) - _highbits(overflow_value)
        coding_table = np.concatenate([coding_table, overflow_coding_table])
    for i in range(np.max(coding_table) + 1):
        total_entropy += _estimate_coding_group_entropy(normalized_pdf[coding_table==i], 
            occurance_group=distribution[coding_table==i], 
            table_log=table_log,
            force_log2_extra_code=force_log2_extra_code
        )
    return total_entropy

def estimate_code_entropy(code_cnt, distributions=None, table_log=None, eps=1e-8):
    if distributions is None:
        distributions = code_cnt # self entropy
    distributions_pdf = distributions / distributions.sum(-1, keepdims=True)
    if table_log is None:
        distributions_pdf += eps
    else:
        table_log_norm = 1<<table_log
        distributions_pdf = np.maximum(1, (distributions_pdf * table_log_norm).round()) / table_log_norm
    # estimate coding length with cross entropy
    return (code_cnt * -np.log2(distributions_pdf)).sum(-1)
     
def generate_tans_coding_table(
        distribution: np.ndarray, 
        max_symbol=255, 
        max_bits=31,
        smooth_distribution=False,
        method="equalize_cdf",
        auto_adjust_max_symbol=True, 
        # allow_symbol_reordering=False,
        # force_log2_extra_code=False, # TODO:
        **kwargs,
    ):
    # 1. determine neccessary parameters
    max_value = 1 << max_bits
    assert(len(distribution.shape) == 1)
    overflow_value = distribution.shape[0]
    assert(overflow_value <= max_value)
    overflow_symbols = _highbits(max_value // overflow_value)

    if auto_adjust_max_symbol:
        max_symbol = 255

    # 2. calculate coding table
    if smooth_distribution:
        distribution_cdf = np.cumsum(np.concatenate([np.array([0]), distribution]))
        smooth_cdf_spline = UnivariateSpline(
            np.arange(len(distribution)+1), distribution_cdf,
            # k=3,
            # s=len(distribution)*256,
        )
        smooth_distribution_cdf = smooth_cdf_spline(np.arange(len(distribution)+1))
        distribution = (smooth_distribution_cdf[1:] - smooth_distribution_cdf[:-1]).clip(0)

    if method == "equalize_cdf":
        coding_table = _equalize_cdf(distribution, 
            max_symbol=max_symbol-overflow_symbols,
            **kwargs
        )
    elif method == "recursive_merge":
        coding_table = _recursive_merge(distribution, 
            max_symbol=max_symbol-overflow_symbols,
            **kwargs
        )
    elif method == "recursive_split":
        coding_table = _recursive_split(distribution, 
            max_symbol=max_symbol-overflow_symbols,
            **kwargs
        )
    elif method == "recursive_smoothing":
        coding_table = _recursive_smoothing(distribution, 
            max_symbol=max_symbol-overflow_symbols,
            **kwargs
        )
    else:
        raise NotImplementedError()

    # 3. calculate num symbols (TODO: maybe trim unnecessary codes)
    coding_extra_symbols = np.bincount(coding_table)
    # build decoding table
    # decoding_table = np.zeros(len(coding_table), dtype=np.int32)
    # decoding_table[coding_table] = np.arange(overflow_value)
    # if allow_symbol_reordering:
    #     coding_table = coding_table[coding_order]
    
    # 4. finally add overflow information
    max_symbol = len(coding_extra_symbols) - 1 + overflow_symbols
    # overflow_start_bits = _highbits(max_symbol + 1)
    overflow_start_bits = _highbits(len(coding_table)) - 1 # _highbits(max_symbol + 1)
    coding_extra_symbols_overflow = 1 << np.arange(overflow_start_bits, overflow_start_bits + overflow_symbols)
    # the first overflow symbol may start from the last coding symbol
    coding_extra_symbols_overflow[0] = (1 << (overflow_start_bits+1)) - len(coding_table) 
    coding_extra_symbols = np.concatenate([coding_extra_symbols, coding_extra_symbols_overflow])
    return coding_table, coding_extra_symbols, max_symbol

def tans_data_to_code(data: np.ndarray, coding_table: np.ndarray, decoding_table: np.ndarray = None):
    # TODO: deal with negative data? or check data is unsigned?
    assert((data >= 0).all())
    overflow_value = len(coding_table)
    overflow_mincode = np.max(coding_table) + 1

    # build decoding table (may move to generator)
    if decoding_table is None:
        decoding_table = np.zeros(overflow_value, dtype=np.int32)
        decoding_table[coding_table] = np.arange(overflow_value)

    # base_code
    data_clip = data.clip(0, overflow_value-1)
    base_code = coding_table[data_clip] 
    # code residue
    data_base = decoding_table[base_code]
    base_extra_code = data_base - data 
    if (data == data_clip).all():
        return base_code, base_extra_code

    # if overflow exists, generate overflow code
    overflow_code = overflow_mincode + (_highbits(data) - _highbits(overflow_value)).clip(0)
    # TODO: solve inverse overflow_extra_code
    overflow_extra_code = 2 ** (_highbits(data)) - 1 - data

    # select extra code
    table_code = np.where(data < overflow_value, base_code, overflow_code)
    extra_code = np.where(data < overflow_value, base_extra_code, overflow_extra_code)

    # get extra bits from coding_extra_symbols (move to coder)
    # extra_symbols = coding_extra_symbols[table_code]

    return table_code, extra_code

def tans_code_to_data(table_code: np.ndarray, extra_code: np.ndarray, coding_table: np.ndarray, decoding_table : np.ndarray = None):
    overflow_value = len(coding_table)
    overflow_mincode = np.max(coding_table)
    table_code_clipped = (table_code - overflow_mincode).clip(0)
    
    # generate overflow data
    data_overflow = 2 ** (table_code_clipped + _highbits(overflow_value) - 1) - 1

    # build decoding table (may move to generator)
    if decoding_table is None:
        decoding_table = np.zeros(overflow_value, dtype=np.int32)
        decoding_table[coding_table] = np.arange(overflow_value)
    # decode base data
    data_base = decoding_table[table_code.clip(0, overflow_value-1)]

    # get extra bits from coding_extra_symbols (move to coder)
    # extra_symbols = coding_extra_symbols[table_code]

    # select by overflow flag
    data = np.where(table_code_clipped == 0, data_base, data_overflow) - extra_code
    return data


def export_zstd_custom_dict(
    coding_table, 
    coding_extra_symbols, 
    table_log=12,
    decoding_table=None,
    decoding_table_min=0,
    num_cnts=None,
    cnt_table_log=8,
    is_huf=False,
    # dict_id=0,
):
    if decoding_table is None:
        # decoding_table = np.zeros(len(coding_extra_symbols), dtype=np.int32)
        # decoding_table[coding_table] = np.arange(len(coding_table))
        decoding_table = np.concatenate((np.zeros(1), np.cumsum(coding_extra_symbols)[:(len(coding_extra_symbols)-1)])).astype(np.uint32)
        decoding_table += decoding_table_min
        
    if num_cnts is None:
        num_cnts = np.ones(1, len(coding_extra_symbols))
    assert(num_cnts.shape[1] == len(coding_extra_symbols))

    encode_table_length = len(coding_table)
    decode_table_length = len(coding_extra_symbols)
    
    byte_strings = [
        # magic number
        # struct.pack('<L', 0xED30A437),
        # dict ID
        # struct.pack('<L', dict_id),
        # table_log
        struct.pack('<L', table_log),
    ]

    if is_huf:
        byte_strings.extend([
            # custom dict mode (1 is huf and 2 is fse)
            struct.pack('B', 1),
        ])
    else:
        byte_strings.extend([
            # custom dict mode (1 is huf and 2 is fse)
            struct.pack('B', 2),
            # encode_table_length
            struct.pack('<L', encode_table_length),
            # encode_table
            b''.join([struct.pack('B', code) for code in coding_table]),
            # decode_table_length
            struct.pack('<L', decode_table_length),
            # decode_table
            b''.join([struct.pack('<L', code) for code in decoding_table]),
            # nsymbols_table
            b''.join([struct.pack('<L', x) for x in coding_extra_symbols]),
        ])
    
    byte_strings.extend([
        # predcnt_table_log
        struct.pack('<L', cnt_table_log),
        # num_cnt_tables
        struct.pack('<L', num_cnts.shape[0]),
        # num_cnts
        b''.join([struct.pack('<L', x) for x in num_cnts.reshape(-1)]),
    ])

    return b''.join(byte_strings)

if __name__ == "__main__":
    data = np.arange(0, 36000)
    coding_table, coding_extra_symbols, max_symbol = \
        generate_tans_coding_table((np.ones(3600)), max_bits=16)
    print(coding_table, coding_extra_symbols, max_symbol)
    table_code, extra_code = tans_data_to_code(data, coding_table)
    print("Encode:")
    print(table_code, extra_code)
    decoded_data = tans_code_to_data(table_code, extra_code, coding_table)
    print("Decode:")
    print(decoded_data)

    assert((data == decoded_data).all())
