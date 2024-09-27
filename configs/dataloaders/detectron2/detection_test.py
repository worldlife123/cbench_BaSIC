from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.data import build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper

config = ClassBuilder(
    build_detection_test_loader,
    ParamSlot("cfg"),
    ParamSlot("dataset_name", default="coco_2017_val"),
    # ParamSlot("dataset"),
    # ParamSlot("mapper", default=DatasetMapper(is_train=False)),
)#.add_all_kwargs_as_param_slot()

