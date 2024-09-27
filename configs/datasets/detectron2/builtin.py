from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.data import DatasetCatalog

config = ClassBuilder(
    DatasetCatalog.get,
    ParamSlot("split", default="coco_2017_train")
)