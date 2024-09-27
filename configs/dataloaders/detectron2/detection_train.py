from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.data import build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper

config = ClassBuilder(
    build_detection_train_loader,
    ParamSlot("cfg"),
    ParamSlot("dataset_name", default="coco_2017_train"),
    # ParamSlot("dataset"),
    # ParamSlot("mapper", default=DatasetMapper(is_train=True)),
)#.add_all_kwargs_as_param_slot()

