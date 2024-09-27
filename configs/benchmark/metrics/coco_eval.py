from cbench.benchmark.metrics.detectron2_metrics import COCOEvaluationMetric

from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    COCOEvaluationMetric,
    ParamSlot("detectron2_model"),
    ParamSlot("dataset_name"),
).add_all_kwargs_as_param_slot()

