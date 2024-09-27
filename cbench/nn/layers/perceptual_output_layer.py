from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualOutputLayer(nn.Module):
    def __init__(self,
                 nn: nn.Module,
                 perceptual_layers : List[str] = [],
                 **kwargs) -> None:
        super().__init__()
        self.nn = nn
        # freeze nn
        for k, p in self.nn.named_parameters():
            p.requires_grad = False
        self.nn.eval()

        self.perceptual_layers = perceptual_layers

        self.perceptual_layers_cache = []
        modules = {k:v for k,v in self.nn.named_modules()}
        for layer_name in perceptual_layers:
            layer = modules.get(layer_name)
            layer.register_forward_hook(self._append_perceptual_layers_cache)

    def train(self, mode: bool = True):
        # eval only even during training
        self.nn.eval()
        return super().train(mode)

    def _append_perceptual_layers_cache(self, model, input, output):
        self.perceptual_layers_cache.append(output)

    def forward(self, input):
        self.nn(input)
        output = [out for out in self.perceptual_layers_cache]
        self.perceptual_layers_cache.clear()
        return output
