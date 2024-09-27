import abc
import numpy as np
import torch
import torch.distributions as dist

from ..base import BaseModule

class EntropyCoderInterface(abc.ABC):
    def encode(self, data, *args, prior=None, **kwargs) -> bytes:
        raise NotImplementedError()

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        raise NotImplementedError()

    def set_stream(self, byte_string: bytes, *args, **kwargs):
        raise NotImplementedError()

    def decode_from_stream(self, *args, prior=None, **kwargs):
        raise NotImplementedError()

    # optional method to cache some state for faster coding
    def update_state(self, *args, **kwargs) -> None:
        pass


class EntropyCoder(BaseModule, EntropyCoderInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
