from cbench.utils.sync_utils import *
import os

from configs.env import DEFAULT_SYNC_URL
from configs.env import DEFAULT_OSS_ENDPOINT, DEFAULT_OSS_BUCKET_NAME, DEFAULT_OSS_KEYID_BASE64, DEFAULT_OSS_KEYSEC_BASE64
from configs.env import DEFAULT_CPU_CORES


# add defaults from config
class SyncUtils(GeneralFileSyncUtils):
    def __init__(self, *args, 
                 url=DEFAULT_SYNC_URL, 
                 max_retry=-1,
                 num_process=DEFAULT_CPU_CORES,
                 # oss params
                 keyId_b64=DEFAULT_OSS_KEYID_BASE64, 
                 keySec_b64=DEFAULT_OSS_KEYSEC_BASE64, 
                 endpoint=DEFAULT_OSS_ENDPOINT, 
                 bucket_name=DEFAULT_OSS_BUCKET_NAME,
                 **kwargs) -> None:
        super().__init__(url, *args,
                         num_process=num_process,
                         max_retry=max_retry,
                         keyId_b64=keyId_b64,
                         keySec_b64=keySec_b64,
                         endpoint=endpoint,
                         bucket_name=bucket_name,
                         **kwargs)