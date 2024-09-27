from cbench.benchmark.metrics.image_classification_metric import ImageClassificationMetric

from configs.import_utils import import_config_from_module
from configs.class_builder import ClassBuilder, ParamSlot

config = ClassBuilder(
    ImageClassificationMetric,
    ParamSlot("classifier"),
).add_all_kwargs_as_param_slot()

