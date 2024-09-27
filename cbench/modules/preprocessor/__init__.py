try:
    from .lz77 import *
except (ImportError, ModuleNotFoundError):
    print("Warning! LZ77 cannot be imported because zstd wrapper is not complied properly!")