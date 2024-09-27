import math

import torch
import torch.nn as nn

import numpy as np

from pytorch_msssim import ms_ssim

from .base import BaseMetric

def _compute_psnr(a, b, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def _compute_ms_ssim(a, b, max_val: float = 1.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()


class PytorchBatchedDistortion(BaseMetric):
    def __init__(self, *args, metrics="psnr", max_val=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = metrics
        self.max_val = max_val

        if not isinstance(metrics, list):
            self._metrics = [metrics]

        self.metric_funcs = []
        for metric_name in self._metrics:
            if metric_name == "psnr":
                self.metric_funcs.append(_compute_psnr)
            elif metric_name == "ms-ssim":
                self.metric_funcs.append(_compute_ms_ssim)
            else:
                raise NotImplementedError(f"{metric_name} is not implemented!")

    @property
    def name(self):
        return "&".join(self._metrics)

    @property
    def metrics(self):
        return self._metrics

    def __call__(self, output, target, cache_metrics=True):
        output = output.type_as(target)[..., :target.shape[-2], :target.shape[-1]] # make spatial size equal
        # TODO: for psnr, maybe we should compute mse and collect psnr later?
        result = {
            name : func(output, target, max_val=self.max_val)
            for name, func in zip(self.metrics, self.metric_funcs)
        }
        if cache_metrics:
            self.metric_logger.update(**result)
        
        return result


