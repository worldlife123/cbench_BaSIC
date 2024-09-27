import itertools
import os
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import math
import numpy as np

from cbench.nn.base import BasicNNTrainer, NNTrainableModule, PLNNTrainableModule, SelfTrainableInterface, SelfTrainableModule
from cbench.nn.models.vqvae_model_v2 import GSSOFT, VQVAE, Decoder, Encoder
from cbench.nn.trainer import make_optimizer, make_scheduler
from cbench.nn.models.vqvae import PyramidVQEmbedding, VQEmbeddingEMA, VQEmbeddingGSSoft, VQVAEDecoder, VQVAEEncoder, VectorQuantizer, VectorQuantizerEMA
from cbench.nn.utils import batched_cross_entropy

from .base import PriorModel
from cbench.modules.prior_model.prior_coder.base import PriorCoder
from ..base import TrainableModuleInterface

class AutoEncoderPriorModel(PriorModel, NNTrainableModule):
    def __init__(self, 
            encoder: NNTrainableModule, 
            decoder: NNTrainableModule, 
            latent_coder: NNTrainableModule, 
            *args, 
            optimizer_config=dict(),
            scheduler_config=dict(),
            input_mean=0.0,
            input_scale=1.0,
            lambda_rd=1.0,
            distortion_type="mse", # mse, ce
            **kwargs):
        super().__init__()
        NNTrainableModule.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_coder = latent_coder

        # input params
        self.input_mean = input_mean
        self.input_scale = input_scale

        # loss params
        self.lambda_rd = lambda_rd
        self.distortion_type = distortion_type

        # TODO: use a trainer
        # self.optimizer = None
        # if optimizer_config is not None:
        #     parameters = list(itertools.chain(encoder.parameters(), decoder.parameters(), latent_coder.parameters()))
        #     if len(parameters) > 0:
        #         self.optimizer = make_optimizer(self, **optimizer_config)
        
        # self.scheduler = None
        # if scheduler_config is not None:
        #     if len(parameters) > 0:
        #         self.scheduler = make_scheduler(optimizer=self.optimizer, **scheduler_config)

        # # training state
        # self.global_step = 0

    # helper function to deal with devices
    # def to(self, device=None, **kwargs):
    #     for module in (self.encoder, self.decoder, self.latent_coder):
    #         module.to(device=device, **kwargs)
    
    def extract(self, data, *args, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        z = self.encoder(data)
        _, z_hat = self.latent_coder(z)
        return z_hat

    def predict(self, data, *args, prior=None, **kwargs):
        return self.decoder(prior)

    def _latent_metric(self, x, z, *args, **kwargs):
        pass

    def forward(self, data, *args, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        x = data
        z = self.encoder((x - self.input_mean) / self.input_scale)
        losses, z_hat = self.latent_coder(z)
        if isinstance(losses, torch.Tensor):
            losses = dict(loss_rate=losses.sum())
        self._latent_metric(x, z)
        x_hat = self.decoder(z_hat)

        if self.distortion_type == "mse":
            loss_distortion = F.mse_loss(x_hat, x)
        elif self.distortion_type == "ce":
            loss_distortion = batched_cross_entropy(x_hat, x).mean()
            # update coding length metric
            # if not self.training:
            self.metric_dict.update(estimated_x_epd=loss_distortion)
        else:
            raise NotImplementedError("")

        # TODO: for lossless compression, 
        # loss_rate and loss_distortion should be normalized together!
        if self.training:
            # loss = loss_rate + self.lambda_rd * loss_distortion
            # self.loss_dict.update(loss_autoencoder=loss)
            self.loss_dict.update(**losses)
            assert('loss_distortion' not in losses)
            self.loss_dict.update(loss_distortion=self.lambda_rd * loss_distortion)

        return x_hat

    # def get_parameters(self, *args, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
    #     return dict(
    #         encoder=self.encoder.state_dict(),
    #         decoder=self.decoder.state_dict(),
    #         latent_coder=self.latent_coder.state_dict(),
    #     )

    # def load_parameters(self, parameters: Dict[str, Dict[str, torch.Tensor]], *args, **kwargs) -> None:
    #     for name, param in parameters.items():
    #         getattr(self, name).load_state_dict(param)

    # def train_full(self, dataloader, *args, **kwargs) -> None:
    #     for data in dataloader:
    #         self.train_iter(data, *args, **kwargs)

    # def train_iter(self, data, *args, **kwargs) -> None:
    #     self.forward(data, *args, **kwargs)

    #     loss = sum(list(self.loss_dict.values()))

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # TODO: scheduler
        
    #     self.global_step += 1

    #     # print("{} : loss {}".format(self.global_step, loss))


class VQVAEPriorModel(AutoEncoderPriorModel):
    def __init__(self, in_channels=3, out_channels=768, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, use_batch_norm=False,
                 latent_dim=1, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.999, use_gssoft_vq=False, **kwargs):
        encoder = nn.Sequential(
            VQVAEEncoder(in_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm),
            nn.Conv2d(in_channels=num_hiddens, 
                        out_channels=embedding_dim*latent_dim,
                        kernel_size=1, 
                        stride=1)
        )
        # encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, num_hiddens, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     nn.ReLU(True),
        #     nn.Conv2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     Residual(num_hiddens),
        #     Residual(num_hiddens),
        #     nn.Conv2d(num_hiddens, latent_dim * embedding_dim, 1)
        # )
        if latent_dim == 1:
            if decay > 0.0:
                vq = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                                commitment_cost, decay)
            else:
                vq = VectorQuantizer(num_embeddings, embedding_dim,
                                            commitment_cost)
        else:
            if use_gssoft_vq:
                vq = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
            else:
                vq = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim, 
                    commitment_cost=commitment_cost,
                    decay=decay,
                )

        decoder = VQVAEDecoder(embedding_dim*latent_dim, out_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm)
        # decoder = nn.Sequential(
        #     nn.Conv2d(latent_dim * embedding_dim, num_hiddens, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     Residual(num_hiddens),
        #     Residual(num_hiddens),
        #     nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     nn.ReLU(True),
        #     nn.Conv2d(num_hiddens, out_channels, 1)
        # )

        super().__init__(encoder=encoder, decoder=decoder, latent_coder=vq,
            distortion_type="ce",
            **kwargs
        )

        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _latent_metric(self, x, z):
        super()._latent_metric(x, z)
        latent_total_entropy = math.log(self.num_embeddings) * z.numel() / self.embedding_dim
        self.metric_dict.update(
            latent_total_entropy=latent_total_entropy,
        )

    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        if self.training:
            # normalize loss rate with data dimension (without batch size)
            loss_rate = self.loss_dict.get('loss_rate')
            if loss_rate is not None:
                self.loss_dict.update(
                    loss_rate = loss_rate / (data.numel() / data.size(0))
                )
        # if not self.training:
        # update coding length metric
        latent_total_entropy = self.metric_dict.get("latent_total_entropy")
        data_total_entropy = self.metric_dict.get("estimated_x_epd") * data.numel()
        self.metric_dict.update(
            estimated_total_entropy = (data_total_entropy + latent_total_entropy),
            estimated_epd = (data_total_entropy + latent_total_entropy) / data.numel(),
        )

        return ret


class VQVAEPriorModelV2(AutoEncoderPriorModel):
    def __init__(self, in_channels=3, out_channels=768, num_hiddens=256,
                 latent_dim=8, num_embeddings=128, embedding_dim=32, 
                 commitment_cost=0.25, decay=0.999, use_gssoft_vq=False,
                 dist_type="RelaxedOneHotCategorical", 
                 relax_temp=1.0, relax_temp_min=1.0, relax_temp_anneal=False, relax_temp_anneal_rate=1e-6,
                 kl_cost=1.0, use_st_gumbel=False, commitment_cost_gs=0.0, commitment_over_exp=False,
                 test_sampling=False,
                 gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
                 **kwargs):
        encoder = Encoder(num_hiddens, latent_dim, embedding_dim, in_channels=in_channels)

        if use_gssoft_vq:
            vq = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim,
                dist_type=dist_type,
                relax_temp=relax_temp,
                relax_temp_min=relax_temp_min,
                relax_temp_anneal=relax_temp_anneal,
                relax_temp_anneal_rate=relax_temp_anneal_rate,
                kl_cost=kl_cost,
                use_st_gumbel=use_st_gumbel,
                commitment_cost=commitment_cost_gs,
                commitment_over_exp=commitment_over_exp,
                test_sampling=test_sampling,
                gs_temp=gs_temp,
                gs_temp_min=gs_temp_min,
                gs_anneal=gs_anneal,
                gs_anneal_rate=gs_anneal_rate,
            )
        else:
            vq = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim, 
                commitment_cost=commitment_cost,
                decay=decay,
            )

        decoder = Decoder(num_hiddens, latent_dim, embedding_dim, out_channels=out_channels)
        # decoder = nn.Sequential(
        #     nn.Conv2d(latent_dim * embedding_dim, num_hiddens, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     Residual(num_hiddens),
        #     Residual(num_hiddens),
        #     nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(num_hiddens),
        #     nn.ReLU(True),
        #     nn.Conv2d(num_hiddens, out_channels, 1)
        # )

        super().__init__(encoder=encoder, decoder=decoder, latent_coder=vq,
            distortion_type="ce",
            **kwargs
        )

        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _latent_metric(self, x, z):
        super()._latent_metric(x, z)
        latent_total_entropy = math.log(self.num_embeddings) * z.numel() / self.embedding_dim
        self.metric_dict.update(
            latent_total_entropy=latent_total_entropy,
        )

    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        if self.training:
            # normalize loss rate with data dimension (without batch size)
            loss_rate = self.loss_dict.get('loss_rate')
            if loss_rate is not None:
                self.loss_dict.update(
                    loss_rate = loss_rate / (data.numel() / data.size(0))
                )
        # if not self.training:
        # update coding length metric
        latent_total_entropy = self.metric_dict.get("latent_total_entropy")
        data_total_entropy = self.metric_dict.get("estimated_x_epd") * data.numel()
        self.metric_dict.update(
            estimated_total_entropy = (data_total_entropy + latent_total_entropy),
            estimated_epd = (data_total_entropy + latent_total_entropy) / data.numel(),
        )

        return ret


class PyramidVQVAEPriorModel(AutoEncoderPriorModel):
    def __init__(self, in_channels=3, out_channels=768, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, use_batch_norm=False,
                 latent_dim=1, pyramid_num_embeddings: List[int]=[int(2**i) for i in range(2, 12)], embedding_dim=64,
                 gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
                 commitment_cost=0.25, decay=0.999, use_gssoft_vq=False, **kwargs):
        encoder = nn.Sequential(
            VQVAEEncoder(in_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm),
            nn.Conv2d(in_channels=num_hiddens, 
                        out_channels=(embedding_dim + len(pyramid_num_embeddings)) * latent_dim,
                        kernel_size=1, 
                        stride=1)
        )

        # vq = PyramidVQEmbeddingGSSoft(latent_dim, pyramid_num_embeddings, embedding_dim)
        vq = PyramidVQEmbedding(latent_dim, pyramid_num_embeddings, embedding_dim,
            gs_temp=gs_temp,
            gs_temp_min=gs_temp_min,
            gs_anneal=gs_anneal,
            gs_anneal_rate=gs_anneal_rate,
            use_gssoft=use_gssoft_vq,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        decoder = VQVAEDecoder(embedding_dim*latent_dim, out_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm)

        super().__init__(encoder=encoder, decoder=decoder, latent_coder=vq,
            distortion_type="ce",
            **kwargs
        )

        self.latent_dim = latent_dim
        # self.pyramid_num_embeddings = pyramid_num_embeddings
        self.register_buffer("pyramid_num_embeddings", torch.as_tensor(pyramid_num_embeddings))
        self.embedding_dim = embedding_dim
    
    def _latent_metric(self, x, z):
        super()._latent_metric(x, z)
        z, pyramid_level = torch.split(z, 
            (self.latent_dim*self.embedding_dim, self.latent_dim*len(self.pyramid_num_embeddings)), 
            dim=1
        )
        pyramid_level = pyramid_level.view(pyramid_level.shape[0], self.latent_dim, len(self.pyramid_num_embeddings), -1).argmax(dim=2)
        z_entropy = torch.log(self.pyramid_num_embeddings.float())[pyramid_level.view(-1)].sum()
        pyramid_level_entropy = math.log(len(self.pyramid_num_embeddings)) * pyramid_level.numel()
        self.metric_dict.update(
            latent_total_entropy=z_entropy+pyramid_level_entropy,
            pyramid_level_mean=pyramid_level.float().mean(),
        )

    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        if self.training:
            # normalize loss rate with data dimension (without batch size)
            loss_rate = self.loss_dict.get('loss_rate')
            if loss_rate is not None:
                self.loss_dict.update(
                    loss_rate = loss_rate / (data.numel() / data.size(0))
                )
        # if not self.training:
        # update coding length metric
        latent_total_entropy = self.metric_dict.get("latent_total_entropy")
        data_total_entropy = self.metric_dict.get("estimated_x_epd") * data.numel()
        self.metric_dict.update(
            estimated_total_entropy = (data_total_entropy + latent_total_entropy),
            estimated_epd = (data_total_entropy + latent_total_entropy) / data.numel(),
        )

        return ret


class PyramidVQVAEPriorModelV2(AutoEncoderPriorModel):
    def __init__(self, in_channels=3, out_channels=768, num_hiddens=256,
                 latent_dim=8, pyramid_num_embeddings: List[int]=[int(2**i) for i in range(2, 12)], embedding_dim=32, 
                 commitment_cost=0.25, decay=0.999, use_gssoft_vq=False, 
                 gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
                 **kwargs):
        encoder = Encoder(num_hiddens, latent_dim, embedding_dim, 
            in_channels=in_channels, 
            out_channels=((embedding_dim + len(pyramid_num_embeddings)) * latent_dim),
        )

        vq = PyramidVQEmbedding(latent_dim, pyramid_num_embeddings, embedding_dim,
            gs_temp=gs_temp,
            gs_temp_min=gs_temp_min,
            gs_anneal=gs_anneal,
            gs_anneal_rate=gs_anneal_rate,
            use_gssoft=use_gssoft_vq,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        decoder = Decoder(num_hiddens, latent_dim, embedding_dim, out_channels=out_channels)

        super().__init__(encoder=encoder, decoder=decoder, latent_coder=vq,
            distortion_type="ce",
            **kwargs
        )

        self.latent_dim = latent_dim
        # self.pyramid_num_embeddings = pyramid_num_embeddings
        self.register_buffer("pyramid_num_embeddings", torch.as_tensor(pyramid_num_embeddings))
        self.embedding_dim = embedding_dim
    

    def _latent_metric(self, x, z):
        super()._latent_metric(x, z)
        z, pyramid_level = torch.split(z, 
            (self.latent_dim*self.embedding_dim, self.latent_dim*len(self.pyramid_num_embeddings)), 
            dim=1
        )
        pyramid_level = pyramid_level.view(pyramid_level.shape[0], self.latent_dim, len(self.pyramid_num_embeddings), -1).argmax(dim=2)
        z_entropy = torch.log(self.pyramid_num_embeddings.float())[pyramid_level.view(-1)].sum()
        pyramid_level_entropy = math.log(len(self.pyramid_num_embeddings)) * pyramid_level.numel()
        self.metric_dict.update(
            latent_total_entropy=z_entropy+pyramid_level_entropy,
            pyramid_level_mean=pyramid_level.float().mean(),
        )

    def forward(self, data, *args, **kwargs):
        ret = super().forward(data, *args, **kwargs)
        if self.training:
            # normalize loss rate with data dimension (without batch size)
            loss_rate = self.loss_dict.get('loss_rate')
            if loss_rate is not None:
                self.loss_dict.update(
                    loss_rate = loss_rate / (data.numel() / data.size(0))
                )
        # if not self.training:
        # update coding length metric
        latent_total_entropy = self.metric_dict.get("latent_total_entropy")
        data_total_entropy = self.metric_dict.get("estimated_x_epd") * data.numel()
        self.metric_dict.update(
            estimated_total_entropy = (data_total_entropy + latent_total_entropy),
            estimated_epd = (data_total_entropy + latent_total_entropy) / data.numel(),
        )

        return ret


class SimplePyramidVQVAEPriorModel(PriorModel, NNTrainableModule):
    def __init__(self, in_channels=3, out_channels=768, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, use_batch_norm=False,
                 num_embeddings : List[int] = [int(2**i) for i in range(2, 12)], 
                 embedding_dim=64, commitment_cost=0.25, decay=0.999, lambda_rd=1.0, 
                 single_decoder=False, 
                #  latent_level_pyramid=False,
                 **kwargs):
        super().__init__()
        NNTrainableModule.__init__(self)

        self.single_decoder = single_decoder
        # self.latent_level_pyramid = latent_level_pyramid
        self.embedding_dim = embedding_dim

        encoder_out_channels = embedding_dim
        # if latent_level_pyramid:
        #     encoder_out_channels += len(num_embeddings)

        self.encoder = nn.Sequential(
            VQVAEEncoder(in_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm),
            nn.Conv2d(in_channels=num_hiddens, 
                        out_channels=encoder_out_channels,
                        kernel_size=1, 
                        stride=1)
        )

        vqs = []
        for num_emb in num_embeddings:
            if decay > 0.0:
                vq = VectorQuantizerEMA(num_emb, embedding_dim, 
                                                commitment_cost, decay)
            else:
                vq = VectorQuantizer(num_emb, embedding_dim,
                                            commitment_cost)
            vqs.append(vq)
        self.latent_coders = nn.ModuleList(vqs)
        
        # self.decoder_input = nn.Conv2d(in_channels=in_channels,
        #                          out_channels=num_hiddens,
        #                          kernel_size=3, 
        #                          stride=1, padding=1)
        if single_decoder:
            self.decoders = nn.ModuleList([VQVAEDecoder(embedding_dim, out_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm)])
        else:
            self.decoders = nn.ModuleList([VQVAEDecoder(embedding_dim, out_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, use_batch_norm=use_batch_norm) for _ in num_embeddings])

        # self.num_embeddings = num_embeddings
        self.register_buffer("num_embeddings", torch.as_tensor(num_embeddings))
        self.lambda_rd = lambda_rd

    def extract(self, data, *args, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)
        batch_size = data.size(0)

        # encoder
        z = self.encoder(data)
        # if self.latent_level_pyramid:
        #     z, pyramid_levels = torch.split(z, (self.embedding_dim, len(self.num_embeddings)))

        x_hat_all = []
        z_hat_all = []
        x_entropies = []
        loss_latent_total = dict()
        loss_ce_list = []

        for idx in range(len(self.num_embeddings)):
            loss_latent, z_hat = self.latent_coders[idx](z)
            z_hat_all.append(z_hat)
            decoder = self.decoders[0] if self.single_decoder else self.decoders[idx]
            x_hat = decoder(z_hat)
            x_hat_all.append(x_hat)
            x_ce = batched_cross_entropy(x_hat, data)
            x_entropies.append(x_ce)
            if isinstance(loss_latent, torch.Tensor):
                loss_latent = dict(loss_rate=loss_latent)
            if self.training:
                for k,v in loss_latent.items():
                    if not k in loss_latent_total:
                        loss_latent_total[k] = 0
                    loss_latent_total[k] += v
                loss_ce_list.append(x_ce.view(batch_size, -1).mean(dim=1))
                # loss_ce_total += loss_ce.mean()
        # z_hat_all = torch.cat(z_hat_all, dim=1)
        # x_hat_all = torch.cat(x_hat_all, dim=0)

        # x_hat_all = self.decoders(z_hat_all)

        # get the best x_hat and corresponding index
        # TODO: a better way to get latent dims
        num_latent_dims = np.prod(z.shape[1:]) // self.embedding_dim
        latent_entropies = (torch.log(self.num_embeddings.float().unsqueeze(0))) \
            .repeat(batch_size * num_latent_dims, 1).type_as(data)
        # x_entropies = batched_cross_entropy(
        #     x_hat_all.reshape(batch_size * len(self.num_embeddings), -1, *data.shape[-2:]),
        #     data.repeat_interleave(len(self.num_embeddings), dim=0)
        # ).mean(dim=1)
        x_entropies = torch.stack(x_entropies, dim=-1)
        x_total_entropies = x_entropies.view(batch_size, -1, len(self.num_embeddings)).sum(dim=1)
        latent_total_entropies = latent_entropies.view(batch_size, -1, len(self.num_embeddings)).sum(dim=1)
        total_entropies = x_total_entropies + latent_total_entropies

        # cross entropy loss
        if self.training:
            loss_ce_total = torch.cat(loss_ce_list).sum() / batch_size
            # loss_autoencoder = (loss_latent_total + self.lambda_rd * loss_ce_total) / len(self.num_embeddings) # + self.lambda_rd * x_entropies.mean()
            # self.loss_dict.update(loss_autoencoder=loss_autoencoder)
            self.loss_dict.update(**loss_latent_total)
            assert 'loss_distortion' not in loss_latent_total
            self.loss_dict.update(loss_distortion=self.lambda_rd * loss_ce_total)
            # normalize by len(self.num_embeddings)
            for loss_key in self.loss_dict:
                self.loss_dict[loss_key] /= len(self.num_embeddings)
        
        min_entropies, best_idxs = total_entropies.min(-1)
        batch_idxs = torch.arange(batch_size, device=total_entropies.device)
        min_x_entropies = x_total_entropies[batch_idxs, best_idxs]
        num_dims = np.prod(data.shape[1:])
        self.metric_dict.update(
            estimated_epd=min_entropies.mean() / num_dims,
            estimated_x_epd=min_x_entropies.mean() / num_dims,
            pyramid_level_mean=best_idxs.float().mean(),
        )

        # TODO: a batched way to output z_hats
        # batch_idxs = torch.arange(data.size(0)).type_as(best_idxs)
        z_hats = torch.stack([z_hat_all[model_idx][batch_idx] for batch_idx, model_idx in enumerate(best_idxs.tolist())])
        return z_hats, best_idxs

    def predict(self, data, *args, prior=None, **kwargs):
        z_hats, model_idxs = prior
        # z_hat_expand = z_hat.repeat_interleave(len(self.num_embeddings), dim=1)
        # batch_idxs = torch.arange(data.size(0)).type_as(model_idxs)
        # x_hat_all = self.decoder(z_hat_expand).reshape(data.size(0), len(self.num_embeddings), -1, *data.shape[-2:])
        # return x_hat_all[batch_idxs, model_idxs]
        x_hats = []
        for z_hat, model_idx in zip(z_hats, model_idxs):
            decoder = self.decoders[0] if self.single_decoder else self.decoders[model_idx]
            x_hat = decoder(z_hat.unsqueeze(0))
            x_hats.append(x_hat)
        
        return torch.cat(x_hats, dim=0)

    def forward(self, data, *args, **kwargs):
        # extract process covers all needed for training
        return self.extract(data, *args, **kwargs)


class VQVAEPreTrainedPriorModel(PriorModel):
    def __init__(self, model: SelfTrainableInterface, *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def extract(self, data, *args, **kwargs):
        # handle device
        if data.device != self.model.device:
            data = data.to(device=self.model.device)

        z = self.model.encoder(data)
        _, z_hat = self.model.codebook(z)
        return z_hat

    def predict(self, data, *args, prior=None, **kwargs):
        return self.model.decoder(prior)


class VQVAESelfTrainedPriorModelImpl(PriorModel, PLNNTrainableModule):
    def __init__(self, model: nn.Module, *args,
        trainer : BasicNNTrainer = None, output_dir=None,
        **kwargs):
        super().__init__(*args, **kwargs)
        PLNNTrainableModule.__init__(self, trainer, output_dir=output_dir)
        self.model = model
        self.trainer.set_model(self.model)

    def extract(self, data, *args, **kwargs):
        # handle device
        if data.device != self.model.device:
            data = data.to(device=self.model.device)

        z = self.model.encoder(data)
        _, z_hat = self.model.codebook(z)
        return z_hat

    def predict(self, data, *args, prior=None, **kwargs):
        return self.model.decoder(prior)


class VQVAESelfTrainedPriorModel(VQVAESelfTrainedPriorModelImpl, PLNNTrainableModule):
    def __init__(self, *args, 
            channels=256, latent_dim=8, num_embeddings=128, embedding_dim=32, 
            input_shift=-0.5, lr=0.0005,
            **kwargs):
        model = VQVAE(channels, latent_dim, num_embeddings, embedding_dim,
            input_shift=input_shift, lr=lr)
        super().__init__(model, *args, **kwargs)


class VQVAEGSSOFTSelfTrainedPriorModel(VQVAESelfTrainedPriorModelImpl, PLNNTrainableModule):
    def __init__(self, *args, 
            channels=256, latent_dim=8, num_embeddings=128, embedding_dim=32, training_soft_samples=True,
            gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
            input_shift=-0.5, lr=0.0005,
            **kwargs):
        model = GSSOFT(channels, latent_dim, num_embeddings, embedding_dim, training_soft_samples=training_soft_samples,
            gs_temp=gs_temp, gs_temp_min=gs_temp_min, gs_anneal=gs_anneal, gs_anneal_rate=gs_anneal_rate,
            input_shift=input_shift, lr=lr)
        super().__init__(model, *args, **kwargs)


class SimplePyramidVQVAESelfTrainedPriorModel(PriorModel, PLNNTrainableModule):
    def __init__(self, *args, 
            trainer : BasicNNTrainer = None, output_dir=None,
            channels=256, latent_dim=8, embedding_dim=32, 
            pyramid_num_embeddings : List[int] = [int(2**i) for i in range(2, 12)], 
            **kwargs):

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.trainer = trainer
        super().__init__(*args, **kwargs)
        PLNNTrainableModule.__init__(self, trainer, output_dir=output_dir)

        models = []
        for num_embeddings in pyramid_num_embeddings:
            models.append(VQVAE(channels, latent_dim, num_embeddings, embedding_dim))
        # NOTE: models are independent module, therefore should not be wrapped with nn.ModuleList
        self.models = models # nn.ModuleList(models)
        self.register_buffer("pyramid_num_embeddings", torch.tensor(pyramid_num_embeddings, requires_grad=False))

    def do_train(self):
        if self.trainer is not None:
            root_dir = self.trainer.output_dir
            for idx, model in enumerate(self.models):
                self.logger.info("Training module with {} embeddings".format(self.pyramid_num_embeddings[idx].item()))
                self.trainer.setup_engine(
                    output_dir=os.path.join(
                        root_dir, 
                        "num_embeddings={}".format(self.pyramid_num_embeddings[idx].item())
                    ),
                    logger=self.trainer.logger,
                )
                self.trainer.initialize(**self.trainer_config)
                self.trainer.set_model(model)
                self.trainer.do_train()

    def extract(self, data, *args, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)
        for model in self.models:
            model.to(device=self.device)
        batch_size = data.size(0)

        # encoder
        # z = self.encoder(data)
        # if self.latent_level_pyramid:
        #     z, pyramid_levels = torch.split(z, (self.embedding_dim, len(self.num_embeddings)))

        x_hat_all = []
        z_hat_all = []
        x_entropies = []
        for idx in range(len(self.models)):
            z = self.models[idx].encoder(data)
            loss_latent, z_hat = self.models[idx].codebook(z)
            x_hat = self.models[idx].decoder(z_hat)
            # (z_hat, x_hat), _ = self.models[idx](data)
            z_hat_all.append(z_hat)
            x_hat_all.append(x_hat)
            x_ce = batched_cross_entropy(x_hat, data)
            # NOTE: assume data is in (0,1)
            # targets = data.clamp(0, 1) * (x_hat.shape[-1]-1)
            # targets = targets.long()
            # x_ce = F.cross_entropy(x_hat.reshape(-1, x_hat.shape[-1]), targets.reshape(-1), reduction='none')
            x_entropies.append(x_ce)

        # z_hat_all = torch.cat(z_hat_all, dim=1)
        # x_hat_all = torch.cat(x_hat_all, dim=0)

        # x_hat_all = self.decoders(z_hat_all)

        # get the best x_hat and corresponding index
        # TODO: a better way to get latent dims
        num_latent_dims = np.prod(z.shape[2:]) * self.latent_dim
        latent_entropies = (torch.log(self.pyramid_num_embeddings.float().unsqueeze(0))) \
            .repeat(batch_size * num_latent_dims, 1).type_as(data)
        # x_entropies = batched_cross_entropy(
        #     x_hat_all.reshape(batch_size * len(self.pyramid_num_embeddings), -1, *data.shape[-2:]),
        #     data.repeat_interleave(len(self.pyramid_num_embeddings), dim=0)
        # ).mean(dim=1)
        x_entropies = torch.stack(x_entropies, dim=-1)
        x_total_entropies = x_entropies.reshape(batch_size, -1, len(self.pyramid_num_embeddings)).sum(dim=1)
        latent_total_entropies = latent_entropies.reshape(batch_size, -1, len(self.pyramid_num_embeddings)).sum(dim=1)
        total_entropies = x_total_entropies + latent_total_entropies
        
        min_entropies, best_idxs = total_entropies.min(-1)
        batch_idxs = torch.arange(batch_size, device=total_entropies.device)
        min_x_entropies = x_total_entropies[batch_idxs, best_idxs]
        num_dims = np.prod(data.shape[1:])
        self.metric_dict.update(
            estimated_epd=min_entropies.mean() / num_dims,
            estimated_x_epd=min_x_entropies.mean() / num_dims,
            pyramid_level_mean=best_idxs.float().mean(),
        )

        if not self.training:
            self.update_cache("hist_dict", 
                pyramid_level=best_idxs.float(),
            )

        # print(x_total_entropies / num_dims, latent_total_entropies / num_dims, best_idxs)

        # TODO: a batched way to output z_hats
        # batch_idxs = torch.arange(data.size(0)).type_as(best_idxs)
        z_hats = torch.stack([z_hat_all[model_idx][batch_idx] for batch_idx, model_idx in enumerate(best_idxs.tolist())])
        return z_hats, best_idxs

    def predict(self, data, *args, prior=None, **kwargs):
        for model in self.models:
            model.to(device=self.device)

        z_hats, model_idxs = prior
        # z_hat_expand = z_hat.repeat_interleave(len(self.num_embeddings), dim=1)
        # batch_idxs = torch.arange(data.size(0)).type_as(model_idxs)
        # x_hat_all = self.decoder(z_hat_expand).reshape(data.size(0), len(self.num_embeddings), -1, *data.shape[-2:])
        # return x_hat_all[batch_idxs, model_idxs]
        x_hats = []
        for z_hat, model_idx in zip(z_hats, model_idxs):
            x_hat = self.models[model_idx].decoder(z_hat.unsqueeze(0))
            x_hats.append(x_hat)
        
        return torch.cat(x_hats, dim=0)

    def forward(self, data, *args, **kwargs):
        # extract process covers all needed for training
        return self.extract(data, *args, **kwargs)
