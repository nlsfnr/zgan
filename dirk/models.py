from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor
from typing import Any, Optional

from .utils import AttrDict


def radford_init(module: nn.Module) -> None:
    """ Initialize the model weights according to Radford et. al:
    https://arxiv.org/pdf/1511.06434.pdf. """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight, 0.0, 0.02)  # type: ignore
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight, 1.0, 0.02)  # type: ignore
        nn.init.zeros_(module.bias)  # type: ignore


class RandomHorizontalFlip(nn.Module):

    def forward(self, imgs: Tensor) -> Tensor:
        if not self.training:
            return imgs
        self.mask = torch.randn((imgs.size(0),)) > 0.5
        return self._masked_flip(imgs, self.mask)

    def backward(self, imgs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return imgs
        return self._masked_flip(imgs, self.mask)

    @staticmethod
    def _masked_flip(imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        new_imgs = []
        for i in range(len(imgs)):
            img = F.hflip(imgs[i]) if mask[i] else imgs[i]
            new_imgs.append(img)
        return torch.stack(new_imgs)


@dataclass(unsafe_hash=True)
class SpatialEncoding(nn.Module):
    n: int = field(hash=False, default=4)

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        half_pi = 0.5 * torch.pi
        device = x.device
        # Height
        xs = (torch.arange(0, half_pi, step=half_pi / h)
              .unsqueeze(0)
              .unsqueeze(0)
              .unsqueeze(-1)).to(device)
        for i in range(self.n):
            t = torch.sin(2**i * xs)
            t = t.expand(b, 1, -1, w)
            x = torch.concat((x, t), dim=1)
        # Width
        xs = (torch.arange(0, half_pi, step=half_pi / w)
              .unsqueeze(0)
              .unsqueeze(0)
              .unsqueeze(0)).to(device)
        for i in range(self.n):
            t = torch.sin(2**i * xs)
            t = t.expand(b, 1, h, -1)
            x = torch.concat((x, t), dim=1)
        return x


def build_layer(spec: AttrDict, ctx: Optional[AttrDict] = None
                ) -> nn.Module:
    if hasattr(nn, spec.type):
        cls = getattr(nn, spec.type)
    else:
        cls = {'RandomHorizontalFlip': RandomHorizontalFlip,
               'SpatialEncoding': SpatialEncoding}[spec.type]

    # Note that this is obviously a security rist, but we trust user input so
    # its ok lol
    def eval_str(x: Any) -> Any:
        if isinstance(x, str):
            return eval(x, ctx)
        return x

    args = [eval_str(arg) for arg in spec.args]
    kwargs = {k: eval_str(v) for k, v in spec.kwargs.items()}
    module = cls(*args, **kwargs)
    assert isinstance(module, nn.Module)
    return module


def build_module(spec: AttrDict) -> nn.Module:
    ctx = spec.get('context', AttrDict())
    return nn.Sequential(*[build_layer(spec, ctx)
                           for spec in spec.layers])


@dataclass(unsafe_hash=True)
class Generator(nn.Module):
    cfg: AttrDict = field(hash=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.main = build_module(self.cfg.arch.gen)
        self.apply(radford_init)

    def random_z(self, n: int) -> Tensor:
        try:
            z = self.cfg.arch.gen.context.z
        except AttributeError:
            z = self.cfg.arch.gen.z
        return torch.randn(size=(n, z, 1, 1))

    def forward(self, z: Tensor) -> Tensor:
        imgs = self.main(z)
        assert isinstance(imgs, Tensor)
        return imgs

    def __str__(self) -> str:
        # To prevent dataclass from generating this
        return super().__str__()

    def __repr__(self) -> str:
        # To prevent dataclass from generating this
        return super().__repr__()  # type: ignore


@dataclass(unsafe_hash=True)
class Discriminator(nn.Module):
    cfg: AttrDict = field(hash=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.main = build_module(self.cfg.arch.dis)
        self.apply(radford_init)

    def forward(self, imgs: Tensor) -> Tensor:
        preds = self.main(imgs).squeeze(dim=-1).squeeze(dim=-1)
        assert isinstance(preds, Tensor)
        return preds

    def __str__(self) -> str:
        # To prevent dataclass from generating this
        return super().__str__()

    def __repr__(self) -> str:
        # To prevent dataclass from generating this
        return super().__repr__()  # type: ignore
