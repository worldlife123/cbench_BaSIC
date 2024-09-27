from typing import Dict, List, Optional
from cbench.utils.logging_utils import MetricLogger, SmoothedValue
from cbench.utils.engine import BaseEngine

class BaseMetric(BaseEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric_logger = MetricLogger()

    @property
    def name(self) -> str:
        return ""

    @property
    def metric_names(self) -> List[str]:
        return []

    def reset(self):
        self.metric_logger.reset()

    def collect_metrics(self):
        return self.metric_logger.get_global_average()

    def __call__(self, output, target, cache_metrics=True) -> Optional[Dict[str, float]]:
        raise NotImplementedError()