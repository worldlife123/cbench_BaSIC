from configs.class_builder import ClassBuilder, ParamSlot
from configs.env import DEFAULT_MAX_MEMORY_CACHE, DEFAULT_SYNC_DATA_ENABLED, DEFAULT_SYNC_URL, DEFAULT_CPU_CORES
from cbench.data.datasets.torchvision_datasets import ImageDatasetWrapper


config = ClassBuilder(
    ImageDatasetWrapper,
    root=ParamSlot("root"),
    max_cache_size=DEFAULT_MAX_MEMORY_CACHE,
    sync_url=DEFAULT_SYNC_URL if DEFAULT_SYNC_DATA_ENABLED else None,
    sync_utils_params=dict(num_process=DEFAULT_CPU_CORES),
    sync_start_action="sync_directory",
    sync_loop_action=None,
    sync_end_action=None,
).add_all_kwargs_as_param_slot()