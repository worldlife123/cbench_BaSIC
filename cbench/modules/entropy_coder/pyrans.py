__all__ = ['RANSByteCoder', 'RANS64Coder', 'RANSFlexCoder', 'PyRANSCoder']

import numpy as np
from .base import EntropyCoderInterface

# py ans coders
def float_to_int_probs(float_probs, prob_scale, min_prob=1):
    pdf = []
    cdf = [0]

    for prob in float_probs:
        next_prob = round(prob * prob_scale)
        if prob > 0 and next_prob < min_prob:
            next_prob = min_prob

        pdf.append(next_prob)
        cdf.append(cdf[-1] + next_prob)

    # Account for possible rounding error
    # Remove the correction from the largest element
    to_correct = prob_scale - cdf[-1]

    largest_index = np.argmax(np.array(pdf))
    pdf[largest_index] += to_correct
    for i in range(largest_index + 1, len(cdf)):
        cdf[i] += to_correct


    return (pdf, cdf)

class PyRANSEncoder(object):
    def __init__(self, state_bits=32, symbol_bits=8, freq_bits=16, **kwargs):
        assert(state_bits % symbol_bits == 0) # TODO: is this needed?
        assert(state_bits - symbol_bits > freq_bits)
        self.state_bits = state_bits
        self.symbol_bits = symbol_bits
        self.freq_bits = freq_bits

        # cached values
        self.freq_prec = 1 << freq_bits
        self.state_lower_bound = 1 << (self.state_bits - self.symbol_bits - 1)
        self.freq_base = (self.state_lower_bound >> self.freq_bits) << self.symbol_bits

        # initialize
        self.initialize()

    def _get_cdf_freq(self, symbol, pdf):
        assert(symbol < len(pdf))
        freq_pdf, freq_cdf = float_to_int_probs(pdf, self.freq_prec)
        return freq_cdf[symbol], freq_pdf[symbol]

    def initialize(self):
        self.state = self.state_lower_bound # RANS64_L
        self.encoded_data = []

    def encode_symbol(self, symbol, pdf=None, quantized_cdf=None):
        # TODO: bypass value
        max_symbol = len(pdf)-2 if not pdf is None else len(quantized_cdf)-2
        if symbol < 0:
            symbol = max_symbol
            bypass_value = -2 * symbol - 1
        elif symbol >= max_symbol:
            symbol = max_symbol
            bypass_value = 2 * (symbol - max_symbol)

        if not pdf is None:
            start, freq = self._get_cdf_freq(symbol, pdf)
        elif not quantized_cdf is None:
            # if max(quantized_cdf) != self.freq_prec: # check freq_prec equality
            #     # renormalize pdf (very slow!)
            #     cdf_norm = np.array(quantized_cdf) / max(quantized_cdf)
            #     pdf = cdf_norm[1:] - cdf_norm[:-1]
            #     start, freq = self._get_cdf_freq(symbol, pdf)
            # else:
            assert(symbol < len(quantized_cdf) - 1)
            assert(quantized_cdf[-1] <= self.freq_prec) # check freq_prec equality
            start, freq = quantized_cdf[symbol], (quantized_cdf[symbol+1] - quantized_cdf[symbol])
        else:
            raise ValueError("Either pdf or quantized_cdf should be provided!")

        assert(freq > 0)

        x = self.state
        # assert(x >= 0)
        # assert(x >= self.state_lower_bound)

        # Check if new symbols should be appended
        x_max = self.freq_base * freq
        while x >= x_max:
            self.encoded_data.append(x & ((1 << self.symbol_bits) - 1))
            x >>= self.symbol_bits

        self.state = ((x // freq) << self.freq_bits) + (x % freq) + start

    def flush(self):
        for _ in range(self.state_bits // self.symbol_bits - 1):
            self.encoded_data.append(self.state & ((1 << self.symbol_bits) - 1))
            self.state >>= self.symbol_bits
        self.encoded_data.append(self.state & ((1 << self.symbol_bits) - 1))
        return self.encoded_data

class PyRANSDecoder(object):
    def __init__(self, encoded_data=None, state_bits=32, symbol_bits=8, freq_bits=16, **kwargs):
        assert(state_bits % symbol_bits == 0) # TODO: is this needed?
        assert(state_bits - symbol_bits > freq_bits)
        self.state_bits = state_bits
        self.symbol_bits = symbol_bits
        self.freq_bits = freq_bits
        self.freq_prec = 1 << freq_bits
        self.state_lower_bound = 1 << (self.state_bits - self.symbol_bits - 1)

        # initialize
        if not encoded_data is None:
            self.initialize(encoded_data)

    def _get_cdf_freq(self, symbol, pdf):
        assert(symbol < len(pdf))
        freq_pdf, freq_cdf = float_to_int_probs(pdf, self.freq_prec)
        return freq_cdf[symbol], freq_pdf[symbol]

    def initialize(self, encoded_data):
        if len(encoded_data) > 0:
            initial_state = encoded_data.pop()
            for _ in range(self.state_bits // self.symbol_bits - 1):
                initial_state <<= self.symbol_bits
                initial_state |= encoded_data.pop()
        else:
            initial_state = self.state_lower_bound
        self.state = initial_state
        self.encoded_data = encoded_data

    def _get_symbol_cdf_freq(self, symbol_cdf, pdf=None, quantized_cdf=None):
        if not pdf is None:
            _, freq_cdf = float_to_int_probs(pdf, self.freq_prec)
        elif not quantized_cdf is None:
            # if max(quantized_cdf) != self.freq_prec: # check freq_prec equality
            #     # renormalize pdf (very slow!)
            #     cdf_norm = np.array(quantized_cdf) / max(quantized_cdf)
            #     pdf = cdf_norm[1:] - cdf_norm[:-1]
            #     _, freq_cdf = float_to_int_probs(pdf, self.freq_prec)
            # else:
            assert(quantized_cdf[-1] <= self.freq_prec) # check freq_prec equality
            freq_cdf = quantized_cdf
        else:
            raise ValueError("Either pdf or quantized_cdf should be provided!")

        symbol = None
        for i in range(len(freq_cdf) - 1):
            if freq_cdf[i] <= symbol_cdf and freq_cdf[i + 1] > symbol_cdf:
                symbol = i
                break
        assert(not symbol is None)
        
        start = freq_cdf[symbol]
        freq = freq_cdf[symbol+1] - freq_cdf[symbol]

        return symbol, start, freq

    # TODO: support for cdf
    def decode_symbol(self, pdf=None, quantized_cdf=None):
        
        # Decode symbol
        symbol_cdf = self.state & (self.freq_prec - 1)

        symbol, start, freq = self._get_symbol_cdf_freq(symbol_cdf, pdf=pdf, quantized_cdf=quantized_cdf)
        
        # Move state foward one step
        x = self.state
        x = freq * (x >> self.freq_bits) + (x & (self.freq_prec - 1)) - start

        # Enough of state has been read that we now need to get more out of encoded_data.
        while x < self.state_lower_bound:
            x = (x << self.symbol_bits)
            if len(self.encoded_data) > 0:
                x |= self.encoded_data.pop()

        self.state = x
        
        # TODO: bypass value
        # max_symbol = len(pdf)-2 if not pdf is None else len(quantized_cdf)-2
        # if symbol == max_symbol:
        #     bypass_value = -2 * symbol - 1

        return symbol

class PyRANSEntropyCoder(EntropyCoderInterface):
    def __init__(self, *args, symbol_bits=8, **kwargs):
        self.symbol_bits = symbol_bits
        super().__init__(PyRANSEncoder(*args, symbol_bits=symbol_bits, **kwargs), 
            PyRANSDecoder(*args, symbol_bits=symbol_bits, **kwargs), *args, **kwargs)
    
    def _unpackbits(self, x, num_bits):
        if np.issubdtype(x.dtype, np.floating):
            raise ValueError("numpy data type needs to be int-like")
        xshape = list(x.shape)
        x = x.reshape([-1, 1])
        mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
        return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

    def encode(self, data, *args, prior=None, **kwargs):
        self.encoder.initialize()
        # reversely encode symbols so that the decoder extracts symbols in sequence
        for symbol, pdf in zip(reversed(data), reversed(prior)):
            # max_value = cdf_lengths[index]
            self.encoder.encode_symbol(symbol, pdf=pdf)
        encoded = self.encoder.flush()
        # inverse bits so that the extra bits added when converting to byte string comes at the first
        bits = self._unpackbits(np.array(encoded), self.symbol_bits)[::-1]
        byte_string = np.packbits(bits.ravel()).tobytes()
        return byte_string

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        decoded_bits = np.unpackbits(np.frombuffer(byte_string, dtype=np.uint8)).reshape(-1, self.symbol_bits)[::-1]
        states = decoded_bits.dot(1 << np.arange(self.symbol_bits)).tolist()
        symbols = []
        self.decoder.initialize(states)
        for pdf in prior:
            # max_value = cdf_lengths[index]
            symbol = self.decoder.decode_symbol(pdf=pdf)
            symbols.append(symbol)
        return symbols
    
    def set_stream(self, string):
        decoded_bits = np.unpackbits(np.frombuffer(string, dtype=np.uint8)).reshape(-1, self.symbol_bits)[::-1]
        states = decoded_bits.dot(1 << np.arange(self.symbol_bits)).tolist()
        self.decoder.initialize(states)

    def decode_stream(self, indexes, cdfs, cdf_lengths, offsets):
        symbols = []
        for index in indexes:
            # max_value = cdf_lengths[index] # TODO: bypass coding
            symbol = self.decoder.decode_symbol(quantized_cdf=cdfs[index]) + offsets[index]
            symbols.append(symbol)
        return symbols

class PyBinaryRANSEncoder(PyRANSEncoder):
    def __init__(self, state_bits=16, symbol_bits=4, freq_bits=11):
        super().__init__(state_bits=state_bits, symbol_bits=symbol_bits, freq_bits=freq_bits)

    def _get_cdf_freq(self, symbol, prob):
        prob_int = min(max(round(prob * self.freq_prec), 1), self.freq_prec-1)
        if symbol > 0:
            freq = self.freq_prec - prob_int
            start = prob_int
        else:
            freq = prob_int
            start = 0

        return start, freq


class PyBinaryRANSDecoder(PyRANSDecoder):
    def __init__(self, encoded_data, state_bits=16, symbol_bits=4, freq_bits=11):
        super().__init__(encoded_data, state_bits=state_bits, symbol_bits=symbol_bits, freq_bits=freq_bits)

    def _get_symbol_cdf_freq(self, symbol_cdf, prob):
        prob_int = min(max(round(prob * self.freq_prec), 1), self.freq_prec-1)
        if symbol_cdf >= prob_int:
            symbol = 1
            freq = self.freq_prec - prob_int
            start = prob_int
        else:
            symbol = 0
            freq = prob_int
            start = 0

        return symbol, start, freq
