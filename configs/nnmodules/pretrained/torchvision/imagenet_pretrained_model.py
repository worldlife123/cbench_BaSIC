from configs.class_builder import ClassBuilder, ParamSlot
import torchvision.models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn

class ImageNetPretrainedModel(nn.Module):
    def __init__(self, model_or_name):
        super().__init__()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        model = getattr(models, model_or_name)(pretrained=True) if isinstance(model_or_name, str) else model_or_name
        assert isinstance(model, nn.Module)
        self.model = model

    def forward(self, x):
        return self.model(self.transform(x.clamp(0, 1)))


config = ClassBuilder(ImageNetPretrainedModel,
    ParamSlot("model_or_name", default="resnet50"),
)
