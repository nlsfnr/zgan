from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor

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


def build_layer(spec: AttrDict) -> nn.Module:
    cls = {'ConvTranspose2d': nn.ConvTranspose2d,
           'Upsample': nn.Upsample,
           'Conv2d': nn.Conv2d,
           'ReLU': nn.ReLU,
           'LeakyReLU': nn.LeakyReLU,
           'BatchNorm2d': nn.BatchNorm2d,
           'Sigmoid': nn.Sigmoid,
           'RandomHorizontalFlip': RandomHorizontalFlip}[spec.type]
    module = cls(*spec.args, **spec.kwargs)
    assert isinstance(module, nn.Module)
    return module


@dataclass(unsafe_hash=True)
class Generator(nn.Module):
    cfg: AttrDict = field(hash=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(*[build_layer(spec)
                                    for spec in self.cfg.arch.gen.layers])
        self.apply(radford_init)

    def random_z(self, n: int) -> Tensor:
        return torch.randn(size=(n, self.cfg.arch.gen.z, 1, 1))

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
        # f = self.cfg.arch.dis.width
        # c = self.cfg.img.channels
        # self.main = nn.Sequential(
        #     # RandomHorizontalFlip(),
        #     # cfg.img_channels x 128 x 128
        #     nn.Conv2d(c, f, 5, 2, 1, bias=False),
        #     nn.BatchNorm2d(f),  # type: ignore
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # cf x 64 x 64
        #     nn.Conv2d(f, f * 4, 5, 2, 1, bias=False),
        #     nn.BatchNorm2d(f * 4),  # type: ignore
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # cf x 32 x 32
        #     nn.Conv2d(f * 4, f * 4, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(f * 4),  # type: ignore
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # cf * 2 x 16 x 16
        #     nn.Conv2d(f * 4, f * 4, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(f * 4),  # type: ignore
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # cf * 4 x 8 x 8
        #     nn.Conv2d(f * 4, f * 8, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(f * 8),  # type: ignore
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # cf * 8 x 4 x 4
        #     nn.Conv2d(f * 8, f * 8, 4, 1, 0),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # cf * 8 x 1 x 1
        #     nn.Conv2d(f * 8, 1, 1, 1, 0),
        #     nn.Sigmoid(),
        #     # 1 x 1 x 1
        # )
        self.main = nn.Sequential(*[build_layer(spec)
                                    for spec in self.cfg.arch.dis.layers])
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
