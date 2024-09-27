from typing import Any
from ..base import BaseModule

# TODO: PriorCoder has similar interface with EntropyCoder! Consider merge them!
class PriorCoder(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, input, *args, **kwargs) -> bytes:
        raise NotImplementedError()

    def decode(self, byte_string, *args, **kwargs) -> Any:
        raise NotImplementedError()

    # optional method to cache some state for faster coding
    def update_state(self, *args, **kwargs) -> None:
        pass
