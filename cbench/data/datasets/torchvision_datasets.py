import os
import numpy as np

import torch
import torch.utils.data
import torchvision
from torchvision import transforms as T
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.io import read_image, decode_image

from typing import Iterable

from .basic import MappingDataset, IterableDataset, CachedFileMappingDataset
from ..transforms import RandomGamma, RandomHorizontalFlip, RandomVerticalFlip, RandomPlanckianJitter, RandomAutocontrast

# helps loading only image data from PIL image datasets such as torchvision cifar10
class ImageDatasetWrapper(CachedFileMappingDataset, VisionDataset): 
    def __init__(self, root: str, 
        num_repeats=1,
        max_num_images=0,
        enable_augmentation=True,
        random_crop_size=256,
        random_gamma=True,
        random_planckian_jitter=1.0,
        random_horizontal_flip=0.5,
        random_vertical_flip=0.5,
        random_auto_contrast=0.0,
        center_crop_size=None,
        resize_size=None,
        post_transforms=[],
        max_cache_size=0,
        **kwargs):
        transforms = [
            T.ConvertImageDtype(torch.float32),
            # T.ToTensor(),
        ]
        if enable_augmentation:
            if random_crop_size > 0:
                transforms.append(T.RandomCrop(random_crop_size, pad_if_needed=True))
            if random_gamma:
                transforms.append(RandomGamma())
            if random_auto_contrast > 0:
                transforms.append(RandomAutocontrast(p=random_auto_contrast))
            if random_planckian_jitter > 0:
                transforms.append(RandomPlanckianJitter(p=random_planckian_jitter))
            if random_horizontal_flip > 0:
                transforms.append(RandomHorizontalFlip(p=random_horizontal_flip))
            if random_vertical_flip > 0:
                transforms.append(RandomVerticalFlip(p=random_vertical_flip))
        # transforms.append(T.Normalize(0.5, 0.5))
        if center_crop_size is not None:
            transforms.append(T.CenterCrop(center_crop_size))
        if resize_size is not None:
            transforms.append(T.Resize(resize_size))
        transforms.extend(post_transforms)
        VisionDataset.__init__(self, root, transform=T.Compose(transforms))

        # build image file list
        file_list = []
        for root, _, fnames in sorted(os.walk(root, followlinks=True)):
            for fname in sorted(fnames):
                # path = os.path.join(root, fname)
                # if any([path.lower().endswith(ext) for ext in IMG_EXTENSIONS]):
                #     file_list.append(path)
                if any([fname.lower().endswith(ext) for ext in IMG_EXTENSIONS]):
                    file_list.append(fname)
        if num_repeats > 1:
            file_list = file_list * num_repeats
        if max_num_images > 0:
            file_list = file_list[:max_num_images]

        if len(file_list) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)
        
        CachedFileMappingDataset.__init__(self, file_list, root=root, max_cache_size=max_cache_size, **kwargs)

    def __getitem__(self, index: int) -> torch.Tensor:
        # path = file_list[index]
        # No need to force RGB. Transforms will handle it.
        # sample = read_image(path)
        byte_string = self._fetch_or_add_to_cache(index)
        sample = decode_image(torch.from_numpy(np.frombuffer(byte_string, dtype=np.uint8).copy()))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample




# class SequentialImageDatasetWrapper(torch.utils.data.IterableDataset):
