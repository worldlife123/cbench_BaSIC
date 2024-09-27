from configs.class_builder import ClassBuilder, ParamSlot
import torchvision

config = ClassBuilder(
    torchvision.datasets.ImageNet,
    root="data/ImageNet",
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256), 
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
    ]),
    split="val",
)