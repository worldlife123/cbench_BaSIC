import torch
import torch.nn as nn

import numpy as np

from pytorch_msssim import ms_ssim

from .base import BaseMetric

from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T

class COCOEvaluationMetric(BaseMetric):
    def __init__(self, detectron2_model, dataset_name, device=None, task="bbox", **kwargs):
        self.detectron2_model = detectron2_model
        self.dataset_name = dataset_name 
        self.task = task

        self.device = device
        self.current_device = None

        for k, p in self.detectron2_model.named_parameters():
            p.requires_grad = False
        self.detectron2_model.eval()
        
        super().__init__(**kwargs)

        self.evaluator = COCOEvaluator(dataset_name, tasks=[task], distributed=False, output_dir=self.output_dir, **kwargs)
        self.evaluator.reset()

    def setup_engine(self, *args, output_dir=None, logger=None, **kwargs):
        super().setup_engine(*args, output_dir=output_dir, logger=logger, **kwargs)
        if self.output_dir:
            self.evaluator = COCOEvaluator(self.dataset_name, tasks=[self.task], distributed=False, output_dir=self.output_dir, **kwargs)
            self.evaluator.reset()

    @property
    def name(self):
        return self.task

    @property
    def metric_names(self):
        # TODO:
        return ["precision", "recall"]

    def reset(self):
        self.evaluator.reset()

    def collect_metrics(self):
        # TODO: support multi task eval
        metrics_per_task = self.evaluator.evaluate()
        return metrics_per_task[self.task]

    def __call__(self, output, target, cache_metrics=True):
        # move model device
        if self.current_device is None:
            if self.device is not None:
                self.detectron2_model = self.detectron2_model.to(self.device)
            else:
                self.device = next(self.detectron2_model.parameters()).device
            self.current_device = self.device

        with torch.no_grad():
            d2_input_images = output.to(device=self.device).clamp(0, 1).flip(1).mul(255)
            d2_input = []
            for idx, img in enumerate(d2_input_images):
                # height, width = img.shape[-2:]
                d2_input.append({"image": img, "height": target[idx]["height"], "width": target[idx]["width"]})
            detectron_results = self.detectron2_model(d2_input)
            self.evaluator.process(target, detectron_results)

