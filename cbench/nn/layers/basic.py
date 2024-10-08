import torch
import torch.nn as nn
import torch.nn.functional as F

# from .base import NNTrainableModule

__all__ = [
    "ResidualLayer",
    "Downsample2DLayer",
    "Upsample2DLayer",
]

class BasicNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, **kwargs):
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        super().__init__()

    def _forward(self, input, **kwargs):
        raise NotImplementedError()

    def forward(self, input, **kwargs):
        assert(input.shape[1] == self.in_channels)
        output = self._forward(input, **kwargs)
        assert(output.shape[1] == self.out_channels)
        return output


class ResidualLayer(BasicNNLayer):
    def __init__(self, in_channels,
        hidden_channels, 
        inplace=True, # Sometimes when gives inplace error, set this to False
        **kwargs):
        super().__init__(in_channels, **kwargs)
        self.block = nn.Sequential(
            nn.ReLU(inplace),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def _forward(self, x, **kwargs):
        return x + self.block(x)


class Downsample2DLayer(BasicNNLayer):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, **kwargs):
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels
        
        super().__init__(in_channels, out_channels, **kwargs)

        layers = [
            # downsample layer
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        ]

        layers.extend([
            # transform layers
            ResidualLayer(hidden_channels, hidden_channels=hidden_channels),
            ResidualLayer(hidden_channels, hidden_channels=hidden_channels),
        ])

        layers.append(
            # output layer
            nn.Conv2d(hidden_channels, out_channels, 1)
        )

        self.block = nn.Sequential(*layers)

    def _forward(self, x, **kwargs):
        return self.block(x)


class Upsample2DLayer(BasicNNLayer):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, **kwargs):
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        super().__init__(in_channels, out_channels, **kwargs)

        layers = [
            # input layer
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        ]

        layers.extend([
            # transform layers
            ResidualLayer(hidden_channels, hidden_channels=hidden_channels),
            ResidualLayer(hidden_channels, hidden_channels=hidden_channels),
        ])

        layers.extend([
            # upsample layer
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            # output layer
            nn.Conv2d(hidden_channels, out_channels, 1),
        ])

        self.block = nn.Sequential(*layers)

    def _forward(self, x, **kwargs):
        return self.block(x)


class AdaptiveResize2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, 
                 num_residual_layers=2,
                 num_downsample_layers=None, 
                 num_upsample_layers=None, 
                 resize_layer_kernel=3,
                 resize_layer_stride=2,
                 adaptive_output_spatial_size=None,
                 adaptive_pooling_method="avg",
                 output_reshape=None,
                 output_permute=None,
                 **kwargs):
        super().__init__(**kwargs)
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_reshape = output_reshape
        self.output_permute = output_permute

        if num_downsample_layers is not None:
            layers = [
                # downsample layer
                nn.Conv2d(in_channels, hidden_channels, resize_layer_kernel, resize_layer_stride, resize_layer_kernel // 2, bias=False),
                nn.BatchNorm2d(hidden_channels),
            ]
            for _ in range(num_downsample_layers-1):
                layers.extend([
                    nn.ReLU(True),
                    nn.Conv2d(hidden_channels, hidden_channels, resize_layer_kernel, resize_layer_stride, resize_layer_kernel // 2, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                ])
        elif num_upsample_layers is not None:
            layers = [
                # downsample layer
                nn.ConvTranspose2d(hidden_channels, hidden_channels, resize_layer_kernel, resize_layer_stride, resize_layer_kernel // 2, bias=False),
                nn.BatchNorm2d(hidden_channels),
            ]
            for _ in range(num_downsample_layers-1):
                layers.extend([
                    nn.ReLU(True),
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, resize_layer_kernel, resize_layer_stride, resize_layer_kernel // 2, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                ])
        else:
            layers = [
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
            ]

        for _ in range(num_residual_layers):
            layers.append(ResidualLayer(hidden_channels, hidden_channels=hidden_channels))

        # output layer
        layers.extend([
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, out_channels, 1) 
        ])

        if adaptive_output_spatial_size is not None:
            if adaptive_pooling_method == "avg":
                layers.append(nn.AdaptiveAvgPool2d(adaptive_output_spatial_size))
            else:
                layers.append(nn.AdaptiveMaxPool2d(adaptive_output_spatial_size))

        self.model = nn.Sequential(*layers)

    def forward(self, input, **kwargs):
        output = self.model(input)
        if self.output_reshape is not None:
            output = output.reshape(*self.output_reshape)
        if self.output_permute is not None:
            output = output.permute(*self.output_permute)
        return output


