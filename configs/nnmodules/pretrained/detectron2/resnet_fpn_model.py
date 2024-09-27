from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.layers import ShapeSpec
from detectron2.model_zoo import get_config
from detectron2.model_zoo.model_zoo import _ModelZooUrls

import torchvision.models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn


class ResNetFPNModel(nn.Module):
    def __init__(self, config_path, trained=True):
        super().__init__()
        self.cfg = get_config(config_path, trained=trained)
        self.transform = transforms.Normalize(mean=self.cfg.MODEL.PIXEL_MEAN, std=self.cfg.MODEL.PIXEL_STD)
        self.model = build_resnet_fpn_backbone(self.cfg, ShapeSpec(channels=3))

    def forward(self, x):
        return list(self.model(self.transform(x.clamp(0, 1).flip(1).mul(255))).values())


config = ClassBuilder(ResNetFPNModel,
    ParamSlot("config_path", 
            #   choices=[_ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()],
              default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
              ),
    trained=True,
)
