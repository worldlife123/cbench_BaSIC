__all__ = [
    'PyCodec',

    # string codecs
    'PyZlibCodec',
    'PyBz2Codec',
    'PyLzmaCodec',
    'PyZstdCodec',
    'PyBrotliCodec',

    # image codecs
    # 'ImagePNGCodec',

    # trainable
    "PyZstdDictCodec",

]

import io
from .base import BaseCodec, BaseTrainableCodec, VariableRateCodecInterface, VariableComplexityCodecInterface

import functools
import numpy as np
from typing import List, Any

# codecs

class PyCodec(BaseCodec):
    def __init__(self, compressor, decompressor, *args, compressor_config: dict = None, decompressor_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor = compressor
        self.decompressor = decompressor
        self.compressor_config = dict() if compressor_config is None else compressor_config
        self.decompressor_config = dict() if decompressor_config is None else decompressor_config

    def compress(self, data, *args, **kwargs):
        return self.compressor(data, *args, **self.compressor_config, **kwargs)

    def decompress(self, data, *args, **kwargs):
        return self.decompressor(data, *args, **self.decompressor_config, **kwargs)


class PyVariableRateCodec(PyCodec, VariableRateCodecInterface):
    def __init__(self, compressor, decompressor, *args, 
                 rate_level_config=dict(),
                 **kwargs):
        super().__init__(compressor, decompressor, *args, **kwargs)
        self.rate_level_config = rate_level_config
        self.current_rate_level = 0
    
    def set_rate_level(self, level, *args, **kwargs) -> bytes:
        self.current_rate_level = level
        self.compressor_config = self.rate_level_config[level]
    
    @property
    def num_rate_levels(self):
        return len(self.rate_level_config)


class PyVariableRateComplexityCodec(PyCodec, VariableRateCodecInterface, VariableComplexityCodecInterface):
    def __init__(self, compressor, decompressor, *args, 
                 rate_level_config=dict(),
                 default_rate_level=0,
                 complex_level_config=dict(),
                 default_complex_level=0,
                 **kwargs):
        super().__init__(compressor, decompressor, *args, **kwargs)
        self.rate_level_config = rate_level_config
        self.complex_level_config = complex_level_config
        self.current_rate_level = default_rate_level
        self.current_complex_level = default_complex_level
    
    def set_rate_level(self, level, *args, **kwargs) -> bytes:
        self.current_rate_level = level
    
    @property
    def num_rate_levels(self):
        return len(self.rate_level_config)
    
    def set_complex_level(self, level, *args, **kwargs) -> None:
        self.current_complex_level = level
    
    @property
    def num_complex_levels(self):
        return len(self.complex_level_config)
    
    def compress(self, data, *args, **kwargs):
        return self.compressor(data, *args, **self.compressor_config, 
                               **self.rate_level_config[self.current_rate_level],
                               **self.complex_level_config[self.current_complex_level],
                               **kwargs)

    # TODO: need decompressor config?
    def decompress(self, data, *args, **kwargs):
        return self.decompressor(data, *args, **self.decompressor_config, **kwargs)



# common codecs
import zlib, bz2, lzma # built-in codecs
# import zstd
import zstandard as zstd
import brotli

PyZlibCodec = functools.partial(PyCodec, zlib.compress, zlib.decompress)
PyBz2Codec = functools.partial(PyCodec, bz2.compress, bz2.decompress)
PyLzmaCodec = functools.partial(PyCodec, lzma.compress, lzma.decompress)
PyZstdCodec = functools.partial(PyCodec, zstd.compress, zstd.decompress)
PyBrotliCodec = functools.partial(PyCodec, brotli.compress, brotli.decompress)


class PyZstdDictCodec(BaseTrainableCodec):
    def __init__(self, *args, 
                 level=3,
                 dict_size=1000,
                 dict_initialize: bytes = b"",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_dict = zstd.ZstdCompressionDict(dict_initialize, *args, **kwargs)
        self.level = level
        self.dict_size = dict_size

    def compress(self, data, *args, **kwargs):
        compressor = zstd.ZstdCompressor(level=self.level, dict_data=self.compression_dict, **kwargs)
        return compressor.compress(data)

    def decompress(self, data, *args, **kwargs):
        decompressor = zstd.ZstdDecompressor(dict_data=self.compression_dict, **kwargs)
        return decompressor.decompress(data)

    def train_full(self, dataloader: List[bytes], *args, **kwargs):
        self.compression_dict = zstd.train_dictionary(
            dict_size=self.dict_size,
            samples=dataloader,
            **kwargs
        )

    def train_iter(self, data, *args, **kwargs) -> None:
        raise RuntimeError("Dictionary training does not support iterable training!")

    def get_parameters(self, *args, **kwargs) -> bytes:
        return self.compression_dict.as_bytes()

    def load_parameters(self, parameters: bytes, *args, **kwargs) -> None:
        self.compression_dict = zstd.ZstdCompressionDict(parameters, *args, **kwargs)


# image codecs
import imageio
try:
    import imageio_flif
except:
    print("imageio_flif not available! ImageFLIFCodec is disabled!")

def imageio_imwrite(data, **kwargs):
    with io.BytesIO() as bio:
        imageio.v2.imwrite(bio, data, **kwargs)
        return bio.getvalue()

def imageio_imread(data, **kwargs):
    return imageio.v2.imread(io.BytesIO(data), **kwargs)

ImageCodec = functools.partial(PyCodec, imageio_imwrite, imageio_imread)
ImagePNGCodec = functools.partial(ImageCodec, 
                                  compressor_config=dict(format="PNG"), 
                                  decompressor_config=dict(format="PNG"))
ImageWebPCodec = functools.partial(ImageCodec, 
                                   compressor_config=dict(format="WebP", lossless=True), 
                                   decompressor_config=dict(format="WebP"))
ImageFLIFCodec = functools.partial(ImageCodec, 
                                   compressor_config=dict(format="FLIF", disable_color_buckets=True), 
                                   decompressor_config=dict(format="FLIF"))

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


class PILPNGCodec(PyCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._compress
        decompressor = self._decompress
        super().__init__(compressor, decompressor, *args, **kwargs)

    def _compress(self, data, **kwargs):
        with io.BytesIO() as bio:
            if not isinstance(data, Image.Image):
                data = ToPILImage()(data)
            data.save(bio, format="png", **self.compressor_config)
            return bio.getvalue()

    def _decompress(self, data, *args, **kwargs):
        return Image.open(io.BytesIO(data))


class PILWebPLosslessCodec(PyCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._compress
        decompressor = self._decompress
        super().__init__(compressor, decompressor, *args, **kwargs)

    def _compress(self, data, **kwargs):
        with io.BytesIO() as bio:
            if not isinstance(data, Image.Image):
                data = ToPILImage()(data)
            data.save(bio, format="webp", lossless=True, **self.compressor_config)
            return bio.getvalue()

    def _decompress(self, data, *args, **kwargs):
        return Image.open(io.BytesIO(data))


class PILJPEGCodec(PyVariableRateComplexityCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._jpeg_compress
        decompressor = self._jpeg_decompress
        rate_level_config = {i:dict(quality=q) for i, q in enumerate(range(0, 100, 5))}
        complex_level_config = {0:dict()}
        super().__init__(compressor, decompressor, *args, 
                         rate_level_config=rate_level_config, 
                         default_rate_level=7,
                         complex_level_config=complex_level_config, 
                         default_complex_level=0,
                         **kwargs)

    def _jpeg_compress(self, data, quality=75, **kwargs):
        with io.BytesIO() as bio:
            if not isinstance(data, Image.Image):
                data = ToPILImage()(data)
            data.save(bio, format="jpeg", quality=quality)
            return bio.getvalue()

    def _jpeg_decompress(self, data, *args, **kwargs):
        return ToTensor()(Image.open(io.BytesIO(data)))


class PILWebPCodec(PyVariableRateComplexityCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._webp_compress
        decompressor = self._webp_decompress
        rate_level_config = {i:dict(quality=q) for i, q in enumerate(range(0, 110, 10))}
        complex_level_config = {i:dict(method=s) for i, s in enumerate(range(0, 7))}
        super().__init__(compressor, decompressor, *args, 
                         rate_level_config=rate_level_config, 
                         default_rate_level=8,
                         complex_level_config=complex_level_config, 
                         default_complex_level=4,
                         **kwargs)

    def _webp_compress(self, data, quality=80, method=4, **kwargs):
        with io.BytesIO() as bio:
            if not isinstance(data, Image.Image):
                data = ToPILImage()(data)
            data.save(bio, format="webp", lossless=False, quality=quality, method=method)
            return bio.getvalue()

    def _webp_decompress(self, data, *args, **kwargs):
        return ToTensor()(Image.open(io.BytesIO(data)))

