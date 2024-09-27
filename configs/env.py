'''
This file is for defining environment specific variables and its default value. 
Do not change this file directly!
Create a new env_config.py instead!
'''
from multiprocessing import cpu_count
import os

# hardware
# 0 means no multiprocessing; using too many cores may slow down data loading!
DEFAULT_CPU_CORES = 8 # cpu_count()
# could be like -1(all), 1, 4, [0, 1, 2, 3], "0, 1, 2, 3"
DEFAULT_GPU_DEVICES = -1 
# 0 mean no memory cache used. By default, linux has its own caching logic, this is not needed!
DEFAULT_MAX_MEMORY_CACHE = 0 

# paths
DEFAULT_PRETRAINED_PATH = "pretrained"
DEFAULT_EXPERIMENT_PATH = "experiments"
DEFAULT_DATA_PATH = "data"
## bpg paths
BPG_ENCODER_PATH="3rdparty/libbpg-0.9.8/bpgenc"
BPG_DECODER_PATH="3rdparty/libbpg-0.9.8/bpgdec"

# sync 
DEFAULT_SYNC_ENABLED = False
DEFAULT_SYNC_URL = ""
DEFAULT_SYNC_EXPERIMENT_PATH = "experiments"
DEFAULT_SYNC_DATA_ENABLED = False
DEFAULT_SYNC_DATA_PATH = "data"

# disk sync
# DEFAULT_SYNC_DISK_ROOT_PATH = ""

# oss
DEFAULT_OSS_KEYID_BASE64 = ""
DEFAULT_OSS_KEYSEC_BASE64 = ""
DEFAULT_OSS_ENDPOINT = ""
DEFAULT_OSS_BUCKET_NAME = ""
# DEFAULT_OSS_PERSONAL_ROOT = ""

try:
    from .env_config import *
except ImportError:
    pass

## detectron2 path
os.environ["DETECTRON2_DATASETS"] = DEFAULT_DATA_PATH
