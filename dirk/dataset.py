import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.io as io  # type: ignore
from torchvision import transforms
from dataclasses import dataclass
from itertools import chain
from tqdm import tqdm  # type: ignore
import glob
import logging
from pathlib import Path
from typing import cast, Optional, List

from .utils import AttrDict


logger = logging.getLogger('dataset')


@dataclass
class ImgDataset(Dataset[Tensor]):
    """Finds and loads all images from a given directory."""
    cfg: AttrDict
    _cache: Optional[Tensor] = None
    _files: Optional[List[str]] = None

    def __post_init__(self) -> None:
        patterns = (Path(self.cfg.data.root) / f'**/*{ext}'
                    for ext in self.cfg.data.extensions)
        its = (glob.iglob(str(patt), recursive=True) for patt in patterns)
        self.files = list(chain(*its))
        logger.info(f'Found {len(self.files)} image files in '
                    f'"{self.cfg.data.root}" matching '
                    f'"{self.cfg.data.extensions}"')
        # Transforms applied after the image is loaded from the cache
        self.dynamic_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            self.img_to_float,
        ])
        # Transforms applied before the image is saved to the cache
        static_transform_steps = [
            transforms.Resize(self.cfg.img.size),
            transforms.CenterCrop(self.cfg.img.size),
        ]
        if self.cfg.img.channels == 1:
            static_transform_steps.append(transforms.transforms.Grayscale())
        self.static_transform = transforms.Compose(static_transform_steps)
        # Cache
        self._cache = self._get_cache()

    def _load_img(self, path: str) -> Tensor:
        img = io.read_image(path)
        img = img[:3, :, :]  # Only keep the RGB channels
        return cast(Tensor, self.static_transform(img))

    def _get_cache(self) -> Tensor:
        c, s = self.cfg.img.channels, self.cfg.img.size
        cache_file = f'.{c}x{s}x{s}.cache'
        cache_path = Path(self.cfg.data.root) / cache_file
        if cache_path.exists():
            logger.info(f'Loading cache file from {cache_path}')
            cache = torch.load(cache_path)  # type: ignore
            assert isinstance(cache, Tensor)
            assert cache.shape == (cache.size(0), c, s, s)
            return cache
        mb = round((len(self) * c * s * s) / 1e6, 2)
        logger.info(f'Filling cache in "{cache_file}", estimated cache size: '
                    f'{mb} MB')
        imgs = [self._load_img(f).unsqueeze(0) for f in tqdm(self.files)]
        logger.info('Combining images')
        cache = torch.cat(imgs)
        assert isinstance(cache, Tensor)
        logger.info(f'Saving to {cache_path}')
        torch.save(cache, cache_path)
        logger.info('Done.')
        return cache

    def __getitem__(self, index: int) -> Tensor:
        assert self._cache is not None
        img = self._cache[index]
        return cast(Tensor, self.dynamic_transform(img))

    def __len__(self) -> int:
        if self._cache is None:
            return len(self.files)
        return len(self._cache)

    @staticmethod
    def img_to_float(img: Tensor) -> Tensor:
        return img.float() / 255.
