import abc
import numpy as np
import torch
import torch.distributions as dist

from .base import EntropyCoder
from cbench.nn.base import NNTrainableModule

class TorchQuantizedEntropyCoder(EntropyCoder, NNTrainableModule):
    def __init__(self, *args, 
        data_format="scalar", 
        data_range=(0, 1), 
        data_precision=256, 
        prior_format="logit", 
        **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

        self.data_format = data_format
        self.data_range = data_range
        self.data_precision = data_precision
        self.prior_format = prior_format

    @property
    def data_step(self):
        return (self.data_range[1] - self.data_range[0]) / (self.data_precision - 1)

    @property
    def data_mid(self):
        return (self.data_range[1] + self.data_range[0]) / 2

    def _data_preprocess(self, data, to_numpy=True):
        if self.data_format == "scalar":
            data = (data - self.data_range[0]) / self.data_step
            data = data.round().long()
        elif self.data_format == "categorical":
            assert data.shape[-1] == self.data_precision
            data = data.argmax(dim=-1)
        else:
            raise NotImplementedError(f"Unknown data format {self.data_format}")
        if to_numpy:
            data = data.detach().cpu().numpy().astype(np.int32)
        return data
    
    def _data_postprocess(self, data):
        if self.data_format == "scalar":
            data = data.astype(np.float32) * self.data_step + self.data_range[0]
        elif self.data_format == "categorical":
            # to one hot vector
            cat_tensor = np.zeros((*data.shape, self.data_precision)).reshape(-1, self.data_precision)
            cat_tensor[np.arange(cat_tensor.shape[0]), data.reshape(-1)] = 1
            data = cat_tensor.reshape(*data.shape, self.data_precision)
            # data = one_hot(torch.as_tensor(data), self.data_precision)
        else:
            raise NotImplementedError(f"Unknown data format {self.data_format}")
        return torch.as_tensor(data)

    # TODO: implement differentiable forward!
    # def forward(self, data, *args, prior=None, **kwargs):
    #     return self._data_postprocess(self._data_preprocess(data))

    def _prior_preprocess(self, prior):
        if self.prior_format == "prob":
            pass
        elif self.prior_format == "logit":
            prior = torch.softmax(prior, dim=-1)
        elif self.prior_format == "gaussian":
            prior_mean, prior_logvar = prior.chunk(2, dim=-1)
            assert prior_mean.shape[-1] == 1 and prior_logvar.shape[-1] == 1
            prior_dist = dist.Normal(prior_mean, torch.exp(prior_logvar))
            pts = torch.arange(self.data_range[0], self.data_range[1]+self.data_step, step=self.data_step).type_as(prior_dist.mean)
            prior = torch.zeros(*prior_dist.mean.shape[:-1], self.data_precision, device=prior_dist.mean.device)
            prior[..., :] = pts
            prior = prior_dist.log_prob(prior)
            prior = torch.softmax(prior, dim=-1)
        else:
            raise NotImplementedError(f"Unknown prior format {self.prior_format}")
        return prior
    
