from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.model_zoo import get
from detectron2.model_zoo.model_zoo import _ModelZooUrls

config = ClassBuilder(get,
    ParamSlot("config_path", 
              choices=list(_ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()),
              default="COCO-Detection/faster_rcnn_R_50_FPN_3x",
              ),
    trained=True,
)
