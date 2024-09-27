import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np

from typing import List, Tuple, Any, Union, Sequence, Optional

from ..base import NNTrainableModule

class BasicParameterGenerator(NNTrainableModule):
    def forward(self, *args, **kwargs):
        return torch.zeros(1)


class Identity(BasicParameterGenerator):
    def forward(self, param, *args, **kwargs):
        return param


class NNParameterGenerator(BasicParameterGenerator):
    def __init__(self, shape : tuple, *args, 
                 init_method="zeros",
                 init_value=None,
                 fix_params=False,
                 freeze_params=False,
                 no_params=False,
                 training_no_params=False,
                 lr_modifier=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.shape = shape
        self.fix_params = fix_params
        self.freeze_params = freeze_params
        self.no_params = no_params
        self.training_no_params = training_no_params

        if self.no_params:
            self.params = None
        else:
            if self.fix_params:
                self.register_buffer("params", torch.zeros(self.shape), persistent=False)
            else:
                self.params = nn.Parameter(torch.zeros(self.shape))
                if self.freeze_params:
                    self.params.requires_grad = False

            if init_method == "zeros":
                nn.init.constant_(self.params, 0)
            elif init_method == "ones":
                nn.init.constant_(self.params, 1)
            elif init_method == "normal":
                nn.init.normal_(self.params)
            elif init_method == "value":
                self.params.data = torch.as_tensor(init_value)
            else:
                raise NotImplementedError()

            if lr_modifier != 1.0:
                for param in self.parameters():
                    param.lr_modifier = lr_modifier

    def forward(self, *args, **kwargs):
        if self.no_params or (self.training and self.training_no_params):
            return None
        # if not self.freeze_params:
        #     self.update_cache("moniter_dict", params_mean=self.params.mean())
        return self.params + 0.0 # simply return a new tensor instead of the param itself


class NNModuleParameterWrapper(BasicParameterGenerator):
    def __init__(self, module : nn.Module, *args, 
                 name_filter=None,
                 unsqueeze_params=False,
                 freeze_params=False,
                 init_method="zeros",
                 init_value=None,
                 lr_modifier=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.freeze_params = freeze_params

        self.params = nn.ParameterList()

        for name, param in module.named_parameters():
            if name_filter is not None:
                if not name_filter in name: continue
            
            # init
            if init_method == "zeros":
                nn.init.constant_(param, 0)
            elif init_method == "ones":
                nn.init.constant_(param, 1)
            elif init_method == "normal":
                nn.init.normal_(param)
            elif init_method == "value":
                param.data = torch.as_tensor(init_value)
            else:
                raise NotImplementedError()

            if self.freeze_params:
                param.requires_grad = False
            if unsqueeze_params:
                param = nn.Parameter(param.unsqueeze(0))
            self.params.append(param)

        if lr_modifier != 1.0:
            for param in self.parameters():
                param.lr_modifier = lr_modifier

    def forward(self, *args, **kwargs):
        return list(self.params)


class GroupedParameterGeneratorWrapper(BasicParameterGenerator):
    def __init__(self, batched_generator : List[BasicParameterGenerator], **kwargs):
        super().__init__(**kwargs)
        self.batched_generator = nn.ModuleList(batched_generator)

    def forward(self, **kwargs):
        return [generator(**kwargs) for generator in self.batched_generator]


class IncreasingVectorGenerator(NNParameterGenerator):
    def __init__(self, shape: Tuple, *args, minimum=0, min_delta=0.0, **kwargs):
        assert len(shape) == 1
        super().__init__(shape, *args, **kwargs)
        self.minimum = minimum
        self.min_delta = min_delta

    def forward(self, *args, **kwargs):
        # force increasing param reset
        if self.minimum is not None:
            self.params.data[:1].clamp_min_(self.minimum)
        delta = (self.params[1:] - self.params[:(self.shape[0]-1)])
        self.params.data[1:] -= (delta - self.min_delta).clamp_max(0)
        return self.params

# similar to EntropyBottleneck in CompressAI
class DifferentiableIncreasingVectorGenerator(BasicParameterGenerator):
    def __init__(
        self,
        channels : int,
        *args: Any,
        default_num_levels : int = 4,
        output_range : Tuple[float, float] = (0, 1),
        include_minmax : bool = False,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.filters = tuple(int(f) for f in filters)
        self.default_num_levels = default_num_levels
        self.output_range = output_range
        self.include_minmax = include_minmax
        self.init_scale = float(init_scale)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

    def _cumulative(self, inputs: torch.Tensor, stop_gradient: bool) -> torch.Tensor:
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            inputs = torch.matmul(F.softplus(matrix), inputs)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            inputs += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                inputs += torch.tanh(factor) * torch.tanh(inputs)
        return inputs

    def forward(self, *args, input=None, **kwargs):
        if input is None:
            input = torch.arange(0, self.default_num_levels, dtype=self._matrix0.dtype, device=self.device)
            input = input.unsqueeze(0).repeat(self.channels, 1)
        output = self._cumulative(input) # output \in (0,1)
        output = self.output_range[0] + output * (self.output_range[1] - self.output_range[0])
        self.include_minmax
        return output


class IndexParameterGenerator(BasicParameterGenerator):
    def __init__(self, shape : tuple, *args, max=1, seed=None, 
                 min=0,
                 sample_non_overlap=False, 
                 sample_continuous_for_training=False, 
                 sample_probability=None,
                 fix_for_inference=False, 
                 fix_for_inference_sample=None,
                 training_no_params=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.shape = shape
        self.max = max
        self.min = min
        self.seed = seed
        self.sample_non_overlap = sample_non_overlap
        self.sample_continuous_for_training = sample_continuous_for_training
        if sample_probability is not None:
            self.register_buffer("sample_probability", torch.as_tensor(sample_probability), persistent=False)
        else:
            self.sample_probability = sample_probability
        self.fix_for_inference = fix_for_inference
        self.fix_for_inference_sample = fix_for_inference_sample
        self.training_no_params = training_no_params

    @property
    def max_sample(self):
        return self.max-1

    @property
    def min_sample(self):
        return self.min

    def forward(self, **kwargs):
        if self.training and self.training_no_params:
            return None
        if not self.training and self.fix_for_inference:
            return self.min_sample if self.fix_for_inference_sample is None else self.fix_for_inference_sample
        rng = None
        # if seed is None:
        #     seed = self.seed
        if self.seed is not None:
            rng = torch.Generator()
            rng.manual_seed(self.seed)
        if self.sample_continuous_for_training and self.training:
            index = torch.empty(*self.shape).uniform_(self.min, self.max-1)
        else:
            if self.sample_probability is not None:
                index = torch.multinomial(self.sample_probability, np.prod(self.shape), replacement=(not self.sample_non_overlap), generator=rng)\
                    .reshape(self.shape)
            elif self.sample_non_overlap:
                if rng is not None:
                    raise NotImplementedError("Non-overlapping random sample does not support manual seed!")
                numel = np.prod(self.shape)
                # if numel == self.max - self.min:
                #     index = torch.arange(self.min, self.max, device=self.device)
                # else:
                index = torch.randperm(self.max-self.min, device=self.device)[:numel].reshape(*self.shape) + self.min
            else:
                index = torch.randint(self.min, self.max, self.shape, generator=rng, device=self.device)
        return index


class IndexSelectParameterGenerator(IndexParameterGenerator):
    def __init__(self, values : List[Any], seed=None, **kwargs):
        super().__init__((1,), max=len(values), seed=seed, **kwargs)

        self.values = values
        self.seed = seed

    def forward(self, index=None, **kwargs):
        if index is None:
            index = super().forward(**kwargs)
        if self.sample_continuous_for_training and self.training:
            # return linear interpolated value
            min, max = self.values[index.floor().long()], self.values[index.ceil().long()]
            return min + (index - index.floor()) * (max - min)
        else:
            return self.values[index]


class IndexSelectParameterGeneratorWrapper(IndexParameterGenerator):
    def __init__(self, batched_generator : Union[int, BasicParameterGenerator, List[BasicParameterGenerator]], seed=None, max=None, **kwargs):
        if isinstance(batched_generator, int):
            max = batched_generator
            super().__init__((1,), max=max, seed=seed, **kwargs)
            self.batched_generator = batched_generator
        elif isinstance(batched_generator, Sequence):
            max = len(batched_generator)
            super().__init__((1,), max=max, seed=seed, **kwargs)
            self.batched_generator = nn.ModuleList(batched_generator)
        else:
            # try generate a batched parameter and get its length
            if max is None:
                max = len(batched_generator())
            super().__init__((1,), max=max, seed=seed, **kwargs)
            self.batched_generator = batched_generator

    def _index_batched_generator(self, index, **kwargs):
        if isinstance(self.batched_generator, int):
            return index
        elif isinstance(self.batched_generator, nn.ModuleList):
            return self.batched_generator[index](**kwargs)
        else:
            return self.batched_generator(**kwargs)[index]

    def forward(self, index=None, **kwargs):
        if index is None:
            index = super().forward(**kwargs)
        if self.sample_continuous_for_training and self.training:
            # return linear interpolated value
            min, max = self._index_batched_generator(index.floor().long(), **kwargs), self._index_batched_generator(index.ceil().long(), **kwargs)
            return min + (index - index.floor()) * (max - min)
        else:
            return self._index_batched_generator(index, **kwargs)


class BernoulliParameterGenerator(BasicParameterGenerator):
    def __init__(self, shape : tuple, *args, 
                 logits_generator : Optional[BasicParameterGenerator] = None,
                 default_logits=0.0,
                 gs_temp=0.5,
                 freeze_params=False,
                 param_lr_modifier=1.0,
                 training_skip_sampling=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.shape = shape
        self.gs_temp = gs_temp
        self.freeze_params = freeze_params
        self.logits_generator = logits_generator
        self.training_skip_sampling = training_skip_sampling
        # TODO: gs annealing?
        
        if logits_generator is None:
            self.logits = nn.Parameter(torch.zeros(self.shape) + default_logits)
        else:
            self.logits_generator = logits_generator

        for param in self.parameters():
            if self.freeze_params:
                param.requires_grad = False
            if param_lr_modifier != 1.0:
                param.lr_modifier = param_lr_modifier

    def _get_default_logits(self, **kwargs):
        if self.logits_generator is None:
            return self.logits
        else:
            return self.logits_generator(**kwargs)
    
    def forward(self, logits=None, **kwargs):
        if logits is None:
            logits = self._get_default_logits(**kwargs)
        if self.training and not self.freeze_params:
            if self.training_skip_sampling:
                bernoulli_weights = logits
            else:
                bernoulli_weights = D.RelaxedBernoulli(self.gs_temp, logits=logits).rsample()
        else:
            bernoulli_weights = (logits > 0).float()

        if not self.freeze_params:
            self.update_cache("moniter_dict", bernoulli_logits_mean=logits.mean())
            # self.update_cache("moniter_dict", bernoulli_weights_mean=bernoulli_weights.mean())
        return bernoulli_weights


class CategoricalParameterGenerator(BasicParameterGenerator):
    def __init__(self, shape : tuple, *args, 
                 logits_generator : Optional[BasicParameterGenerator] = None,
                 num_categories=2,
                 default_logits=None,
                 gs_temp=0.5,
                 freeze_params=False,
                 param_lr_modifier=1.0,
                 training_skip_sampling=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.shape = shape
        self.logits_generator = logits_generator
        self.num_categories = num_categories
        self.gs_temp = gs_temp
        self.freeze_params = freeze_params
        self.training_skip_sampling = training_skip_sampling
        # TODO: gs annealing?

        if logits_generator is None:
            cat_logits = torch.zeros(*self.shape, num_categories)
            if default_logits is not None:
                assert len(default_logits) == num_categories
                cat_logits += torch.tensor(default_logits).repeat(*self.shape, 1)
            self.logits = nn.Parameter(cat_logits)
        else:
            self.logits_generator = logits_generator
        
        for param in self.parameters():
            if self.freeze_params:
                param.requires_grad = False
            if param_lr_modifier != 1.0:
                param.lr_modifier = param_lr_modifier

    def _get_default_logits(self, **kwargs):
        if self.logits_generator is None:
            return self.logits
        else:
            logits = self.logits_generator(**kwargs)
            # split channels to num_categories and stack them at the last dim
            logits = torch.stack(torch.chunk(logits, self.num_categories, dim=1), dim=-1)
            return logits

    def forward(self, logits=None, **kwargs):
        if logits is None:
            logits = self._get_default_logits(**kwargs)
        if self.training and not self.freeze_params:
            if self.training_skip_sampling:
                cat_weights = logits # torch.softmax(logits)
            else:
                cat_weights = D.RelaxedOneHotCategorical(self.gs_temp, logits=logits).rsample()
        else:
            cat_weights = F.one_hot(logits.argmax(-1), logits.shape[-1]).type_as(logits)

        if not self.freeze_params:
            self.update_cache("moniter_dict", categorical_logits_mean=logits.mean())
            self.update_cache("moniter_dict", categorical_samples_mean=cat_weights.argmax(-1).float().mean())
        return cat_weights


# e.g. [0,0,1,0] -> [1,1,1,0]
class CategoricalToRangeGenerator(BasicParameterGenerator):
    def __init__(self, shape : tuple, *args, 
                 shape_output=None,
                 num_categories=2,
                 default_logits=None,
                 gs_temp=0.5,
                 freeze_params=False,
                 param_lr_modifier=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.shape = shape
        self.shape_output = shape_output
        self.num_categories = num_categories
        self.gs_temp = gs_temp
        self.freeze_params = freeze_params
        # TODO: gs annealing?
        
        cat_logits = torch.zeros(*self.shape, num_categories)
        if default_logits is not None:
            assert len(default_logits) == num_categories
            cat_logits += torch.tensor(default_logits).repeat(*self.shape, 1)
        self.logits = nn.Parameter(cat_logits)

        if self.freeze_params:
            self.logits.requires_grad = False
        if param_lr_modifier != 1.0:
            self.logits.lr_modifier = param_lr_modifier

        # buffer for range
        range_matrix = [[1 if j<=i else 0 for j in range(num_categories)] for i in range(num_categories)]
        range_matrix = torch.tensor(range_matrix).type_as(self.logits).repeat(*self.shape, 1, 1)
        self.register_buffer("range_matrix", range_matrix, persistent=False)

    def forward(self, logits=None, **kwargs):
        if logits is None:
            logits = self.logits
        range_weights = torch.matmul(logits.unsqueeze(-2), self.range_matrix).squeeze(-1)

        if self.shape_output is not None:
            range_weights = range_weights.reshape(*self.shape_output, self.num_categories)

        return range_weights


class TensorSplitGenerator(BasicParameterGenerator):
    def __init__(self, split_size_or_sections, *args, 
                 dim=0, index=None, postprocess=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim
        self.index = index
        self.postprocess = postprocess

    def forward(self, input, index=None, **kwargs):
        if index is None: index = self.index
        tensor_splits = torch.split(input, self.split_size_or_sections, dim=self.dim)
        result = tensor_splits if index is None else tensor_splits[index]
        if self.postprocess == "softmax":
            result = torch.softmax(result, dim=self.dim)
        return result


class ConvTranspose2dParameterGenerator(BasicParameterGenerator):
    def __init__(self, *args, in_channels=64, out_channels=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        ngf = self.in_channels
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.in_channels, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, self.out_channels, 4, 2, 1, bias=False),
            # nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, *args, **kwargs):
        z = torch.normal(torch.zeros(1, self.in_channels, 1, 1, device=self.device))
        return self.model(z)


# https://github.com/Mohanned-Elkholy/ResNet-GAN/blob/main/src/generator/utils.py#L100
class ResBlockGenerator(nn.Module):
    """ This class make the standard resblock generator """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        
        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.shortcut(x)

class Dense_block(nn.Module):
    """ This is the initial dense block as in the paper """
    def __init__(self,in_channels,out_channels):
        super(Dense_block, self).__init__()

        self.dense = nn.Linear(in_channels,out_channels)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        self.activation = nn.LeakyReLU(0.2)


    def forward(self,x):
        return self.activation(self.dense(x))

class ResNet2dParameterGenerator(BasicParameterGenerator):
    def __init__(self, *args, in_channels=64, out_channels=64, resnet_channels=64, out_width=64, initial_width=4, num_initial_dense=4, lr_modifier=1.0, batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet_channels = resnet_channels
        self.batch_size = batch_size
        # initial width of the generator
        self.initial_width = self.initial_height = initial_width
        self.out_width = self.out_height = out_width
        # the first dense layer
        self.dense = nn.Linear(self.in_channels, self.initial_width * self.initial_height * resnet_channels)
        # the first dense blocks
        self.initial_dense_blocks = [Dense_block(self.initial_width * self.initial_height * resnet_channels,self.initial_width * self.initial_height * resnet_channels) for i in range(num_initial_dense)]
        # mapping stack
        self.mapping_stack = nn.Sequential(*self.initial_dense_blocks)
        # the final conv layer
        self.final = nn.Conv2d(resnet_channels, out_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)


        # number of Resblocks
        res_block_number = int(np.log(out_width)/np.log(2))-int(np.log(self.initial_width)/np.log(2))
        # make the list of the initial resblocks
        self.generator_blocks = [ResBlockGenerator(resnet_channels, resnet_channels, stride=2) for i in range(res_block_number)]
        # the model
        self.model = nn.Sequential(
            # the initial dense and the reshaping layer are in the forward
            *self.generator_blocks,
            nn.BatchNorm2d(resnet_channels),
            nn.ReLU(),
            self.final,
            # nn.Tanh()

            )
        
        if lr_modifier != 1.0:
            for param in self.parameters():
                param.lr_modifier = lr_modifier

    def forward(self, *args, **kwargs):
        z = torch.normal(torch.zeros(self.batch_size, self.in_channels, device=self.device))
        # initial z
        z = self.dense(z)
        # mapping
        z = self.mapping_stack(z)
        # reshape
        z = z.view(-1, self.resnet_channels, self.initial_width, self.initial_height)
        # get the output
        output = self.model(z)[:, :, :self.out_height, :self.out_width]

        return output

from cbench.nn.models.transgan_generator import Generator
class Transformer2dParameterGenerator(BasicParameterGenerator):
    def __init__(self, *args, in_channels=64, out_channels=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = Generator(
            latent_dim=self.in_channels, 
            out_dim=self.out_channels,
            # bottom_width=8,
            # embed_dim=192,
            # gf_dim=512,
            # g_depth=[2,2,2],
        )


    def forward(self, *args, **kwargs):
        z = torch.normal(torch.zeros(1, self.in_channels, device=self.device))
        output = self.model(z)
        return output