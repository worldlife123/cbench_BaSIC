from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.layers import ShapeSpec

config = ClassBuilder(build_resnet_fpn_backbone,
    ParamSlot("cfg"),
    input_shape=ShapeSpec(channels=3),
)
