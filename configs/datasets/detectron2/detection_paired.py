from typing import Iterator, Sequence
from torch.utils.data import Dataset, Sampler
from configs.class_builder import ClassBuilder, ParamSlot
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.structures import ImageList
from detectron2.data import DatasetCatalog
from detectron2.data.detection_utils import read_image

from torch.utils.data.dataloader import Dataset, IterableDataset
import torchvision.transforms as T
import torch 

class DetectionTestPairedLoader(Dataset):
    def __init__(self, cfg: Dataset, dataset_name, train=False, size_divisibility=32, to_float=True, to_rgb=True, transform=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.train = train
        self.to_float = to_float
        self.to_rgb = to_rgb
        self.size_divisibility = size_divisibility
        
        if train:
            self.detection_loader = build_detection_train_loader(cfg, dataset_name, **kwargs)
        else:
            self.detection_loader = build_detection_test_loader(cfg, dataset_name, **kwargs)
        self.transform = transform
        
    # def __getitem__(self, index):
    #     data_list = self.detection_loader.__getitem__(index)
    #     images = [data["image"] for data in data_list]
    #     if self.transform is not None:
    #         images = [self.transform(img) for img in images]
    #     images = ImageList.from_tensors(images, size_divisibility=self.size_divisibility).tensor
    #     return images, data_list
        
    def __iter__(self) -> Iterator:
        for data_list in self.detection_loader.__iter__():
            # images = [T.ToTensor()(read_image(data["file_name"])) for data in data_list]
            images = [data["image"] for data in data_list]
            # if self.to_float:
            #     images = [img.float().div_(255) for img in images]
            # if self.to_rgb:
            #     images = [img[::-1] for img in images]
            if self.transform is not None:
                images = [self.transform(img) for img in images]
            images = torch.stack(images) # ImageList.from_tensors(images, size_divisibility=self.size_divisibility).tensor
            if self.to_float:
                images = images.float().div_(255)
            # d2 loaders load images in BGR format, should be convert to RGB
            if self.to_rgb:
                images = images.flip(1)
            yield images, data_list

    def __len__(self):
        return len(self.detection_loader)

config = ClassBuilder(
    DetectionTestPairedLoader,
    ParamSlot("cfg"),
    ParamSlot("dataset_name", default="coco_2017_val", choices=list(DatasetCatalog.keys())),
).add_all_kwargs_as_param_slot()

