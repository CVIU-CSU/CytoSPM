# Copyright (c) Tencent Inc. All rights reserved.
# ConvNeXt portions Copyright (c) Meta Platforms, Inc. and affiliates (MIT).
# Modified by the CytoSPM Authors, 2026.
import itertools
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from timm.layers import DropPath, trunc_normal_
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return tuple(outputs)


@MODELS.register_module()
class ConvNextVisionBackbone(BaseModule):
    """ConvNeXt image backbone returning the four stage outputs.

    Args:
        model_name (str): One of ``tiny``, ``base`` or ``large``.
        norm_eval (bool): Keep BatchNorm layers in eval mode during training.
        frozen_modules (Sequence[str]): Submodule name prefixes to freeze;
            ``('all',)`` freezes the whole backbone.
    """

    _ARCHS = {
        'tiny': dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
        'base': dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
        'large': dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    }

    def __init__(self,
                 model_name: str,
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model_name = model_name
        self.model = ConvNeXt(**self._ARCHS[model_name])
        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        return self.model(image)

    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            return
        for name, module in self.model.named_modules():
            if self.frozen_modules[0] == 'all' or any(
                    name.startswith(p) for p in self.frozen_modules):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Text "encoder" that looks up pre-computed class-name embeddings.

    CytoSPM keeps the BioMedCLIP text encoder frozen, so class-name embeddings
    are computed once offline and loaded from ``text_embed_path`` (a dict
    mapping class name to a ``[D]`` tensor).
    """

    def __init__(self,
                 text_embed_path: str = '',
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.text_embed = torch.load(
            text_embed_path, map_location='cpu', weights_only=True)
        self.register_buffer('buff', torch.zeros([1]))

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        cache_key = tuple(text[0])
        if any(tuple(t) != cache_key for t in text):
            return self.forward_text(text)
        if not hasattr(self, 'cache_bank'):
            self.cache_bank = {}
        if cache_key not in self.cache_bank:
            self.cache_bank[cache_key] = self.forward_text(
                [list(cache_key)]).detach()
        return self.cache_bank[cache_key].expand(len(text), -1, -1)

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), \
            'number of sequences not equal in batch'
        text = list(itertools.chain(*text))
        text_embeds = torch.stack(
            [self.text_embed[x.split('/')[0]] for x in text])
        text_embeds = text_embeds.to(self.buff.device).requires_grad_(False).float()
        return text_embeds.reshape(-1, num_per_batch[0], text_embeds.shape[-1])


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    """Image backbone plus class-name text backbone (no cross-modal fusion)."""

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.image_model = MODELS.build(image_model)
        self.text_model = MODELS.build(text_model)

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        return self.image_model(image), self.text_model(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        return self.text_model(text)

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)
