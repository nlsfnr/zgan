import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from logging import getLogger
from PIL import Image  # type: ignore
import base64
from io import BytesIO

from .utils import AttrDict
from .models import Generator, Discriminator


logger = getLogger('inference')


@dataclass
class Inference:
    cfg: AttrDict
    checkpoint: Path
    gen: Generator = field(init=False)
    dis: Discriminator = field(init=False)

    def __post_init__(self) -> None:
        state = torch.load(self.checkpoint)  # type: ignore
        assert isinstance(state, AttrDict)
        self.gen = Generator(self.cfg)
        self.gen.load_state_dict(state.trainer.gen)
        self.dis = Discriminator(self.cfg)
        self.dis.load_state_dict(state.trainer.dis)
        logger.info('Loaded Generator and Discriminator from '
                    f'{self.checkpoint}')

    def threshold(self, n: int, threshold: float, max_attempts: int = 1024
                  ) -> List[Tensor]:
        selection: List[Tensor] = []
        while len(selection) < n and max_attempts != 0:
            remaining = n - len(selection)
            z = self.gen.random_z(128)
            imgs = self.gen(z)
            scores = self.dis(imgs).squeeze(-1)
            indices = scores >= threshold
            selection.extend(imgs[indices][:remaining])
            max_attempts -= 1
            logger.info(f'Found {len(selection)} samples with {max_attempts} '
                        'attempts remaining, mean score is '
                        f'{torch.mean(scores)}')
        return selection

    def best(self, n: int, pop: int) -> List[Tensor]:
        assert pop >= n
        selection: List[Tensor] = []
        scores = []
        while len(selection) < pop:
            z = self.gen.random_z(128)
            imgs = self.gen(z)
            scores_ = self.dis(imgs).squeeze(-1)
            selection.extend(imgs)
            scores.extend(scores_)
        sel = torch.stack(selection)
        sco = torch.stack(scores)
        indices = torch.argsort(sco)[:n]
        sco_mean = float(torch.mean(sco))
        sel_mean = float(torch.mean(sel[indices]))
        logger.info(f'Chose {n} best samples from {len(selection)}, '
                    f'mean pop scores: {round(sco_mean, 4)}, '
                    f'mean selection scores: {round(sel_mean, 4)} ')
        return [sel[idx] for idx in indices]

    @staticmethod
    def to_base64_img(tensor: Tensor) -> str:
        arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = Image.fromarray((255 * arr).astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_byte = buffer.getvalue()
        img_str = base64.b64encode(img_byte).decode()
        return "data:image/png;base64," + img_str

    @classmethod
    def to_html(cls, tensor: Tensor) -> str:
        return f'<img src="{cls.to_base64_img(tensor)}"/>'
