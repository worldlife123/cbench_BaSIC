from configs.class_builder import ClassBuilder, ParamSlot
import torchvision

config = ClassBuilder(
    torchvision.datasets.ImageNet,
    root="data/ImageNet",
    transform=ClassBuilder(
        torchvision.transforms.ToTensor,
    ),
    split=ParamSlot("split", default="train", choices=["train", "val"]),
)