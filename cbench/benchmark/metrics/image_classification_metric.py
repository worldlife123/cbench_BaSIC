import torch
import torch.nn as nn

import numpy as np

from torchvision.transforms import Normalize
from .base import BaseMetric

from detectron2.evaluation import COCOEvaluator

class ImageClassificationMetric(BaseMetric):
    def __init__(self, classifier : nn.Module, topk=(1,), device=None, apply_imagenet_transform=False, **kwargs) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.classifier = classifier
        self.topk = topk
        self.apply_imagenet_transform = apply_imagenet_transform

        self.device = device
        self.current_device = None
        for k, p in self.classifier.named_parameters():
            p.requires_grad = False
        self.classifier.eval()

        # ImageNet mean/scale (should be considered in classifier)
        if self.apply_imagenet_transform:
            self.transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _accuracy(self, output, target):
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = dict()
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res[f"accuracy_top{k}"] = correct_k.mul_(100.0 / batch_size).item()
        return res

    @property
    def name(self):
        return "accuracy"

    @property
    def metric_names(self):
        return [f"accuracy_top{k}" for k in self.topk]

    def __call__(self, output, target, cache_metrics=True):
        # move model device
        if self.current_device is None:
            if self.device is not None:
                self.classifier = self.classifier.to(self.device)
            else:
                self.device = next(self.classifier.parameters()).device
            self.current_device = self.device

        with torch.no_grad():
            cls_input = output.to(device=self.device)
            if self.apply_imagenet_transform:
                cls_input = self.transform(cls_input)
            output_cls = self.classifier(cls_input)
            result = self._accuracy(output_cls, target.to(device=self.device))
            if cache_metrics:
                self.metric_logger.update(**result)
            return result

