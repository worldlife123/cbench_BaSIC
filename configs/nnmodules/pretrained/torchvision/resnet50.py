from configs.class_builder import ClassBuilder, ParamSlot
from torchvision.models import resnet50

config = ClassBuilder(resnet50,
    pretrained=True,
)
