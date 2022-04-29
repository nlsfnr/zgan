from __future__ import annotations
from torch import Tensor
import numpy as np
import torchvision.utils as vutils  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import yaml
from typing import Dict, Any, Tuple
from pathlib import Path


class AttrDict(Dict[str, Any]):

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key not in self:
                raise e
            return self[key]

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AttrDict:
        return AttrDict(**{k: cls.from_dict(v) if isinstance(v, dict) else v
                           for k, v in d.items()})

    @classmethod
    def from_yaml(cls, path: Path) -> AttrDict:
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d)


def show(img: Tensor, show: bool = True) -> None:
    """ Plots the given image. """
    if img.dim() == 4 and img.size(0) != 1:
        return show_grid(img)
    if isinstance(img, Tensor):
        img = (img
               .cpu()
               .numpy()
               .squeeze()
               .transpose((1, 2, 0)))  # C, H, W -> H, W, C
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()


def show_grid(imgs: Tensor, figsize: Tuple[int, int] = (12, 12),
              show: bool = True) -> None:
    """ Plots the given images in a grid. """
    imgs = imgs.detach().cpu()
    grid = vutils.make_grid(imgs, padding=2, value_range=(0, 1))
    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    if show:
        plt.show()
