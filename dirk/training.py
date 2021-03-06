from __future__ import annotations
from dataclasses import dataclass, field
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from logging import getLogger
from itertools import count
from typing import Tuple, Optional, Union, Iterator, Dict

from .utils import AttrDict
from .models import Generator, Discriminator
from . import sidecar as sidecar_
from . import dataset as dataset_


logger = getLogger('trainer')

# For the GAN training algorithm
REAL = 1.
GENERATED = 0.


@dataclass
class Trainer:
    """The Trainer contains both the Generator and Discriminator and their
    respective optimisers. It implements the training loop."""
    cfg: AttrDict
    gen: Generator
    dis: Discriminator
    gen_opt: optim.AdamW
    dis_opt: optim.AdamW
    sidecar: sidecar_.Sidecar
    dataset: dataset_.ImgDataset
    _device: torch.device = field(init=False)
    _gen_ema: EMA = field(init=False)
    _gen_gpu: Optional[nn.DataParallel] = None  # type: ignore
    _dis_gpu: Optional[nn.DataParallel] = None  # type: ignore

    def __post_init__(self) -> None:
        self.sidecar.attach(self)
        self._device = self.get_device(self.cfg)
        self._gen_ema = EMA(self.cfg.optim.gen.ema_alpha)
        self._gen_ema.register_module(self.gen)
        if self.cfg.use_gpu:
            gpu_ids = tuple(range(torch.cuda.device_count()))
            logger.info(f'Using CUDA with GPUs: {gpu_ids}')
            self._gen_gpu = nn.DataParallel(self.gen, gpu_ids)  # type: ignore
            self._dis_gpu = nn.DataParallel(self.dis, gpu_ids)  # type: ignore
        else:
            logger.info('Using CPU')

    @classmethod
    def from_config(cls, cfg: AttrDict, sc: sidecar_.Sidecar) -> Trainer:
        # Instantiate modules with default initialisation
        device = cls.get_device(cfg)
        gen = Generator(cfg).to(device)
        dis = Discriminator(cfg).to(device)
        gen_opt, dis_opt = cls._get_optimizers(gen, dis, cfg)
        dataset = dataset_.ImgDataset(cfg)
        logger.info(f'Loading trainer from config in {sc.workdir}')
        return cls(cfg, gen, dis, gen_opt, dis_opt, sc, dataset)

    def get_state(self) -> AttrDict:
        """Returns the state dict of the Trainer."""
        return AttrDict(
                gen=self.gen.state_dict(),
                dis=self.dis.state_dict(),
                gen_opt=self.gen_opt.state_dict(),
                dis_opt=self.dis_opt.state_dict(),
        )

    @classmethod
    def from_state(cls, cfg: AttrDict, sc: sidecar_.Sidecar, state: AttrDict
                   ) -> Trainer:
        """Instantiates a Trainer from a state dict."""
        # Instantiate modules
        device = cls.get_device(cfg)
        # The models need to be on the device before being assigned to an
        # optimizer.
        gen = Generator(cfg).to(device)
        dis = Discriminator(cfg).to(device)
        gen_opt, dis_opt = cls._get_optimizers(gen, dis, cfg)
        dataset = dataset_.ImgDataset(cfg)
        # Initialize modules
        gen.load_state_dict(state.gen)
        dis.load_state_dict(state.dis)
        gen_opt.load_state_dict(state.gen_opt)
        dis_opt.load_state_dict(state.dis_opt)
        return cls(cfg, gen, dis, gen_opt, dis_opt, sc, dataset)

    @staticmethod
    def get_device(cfg: AttrDict) -> torch.device:
        return torch.device('cuda' if cfg.use_gpu else 'cpu')

    @staticmethod
    def _get_optimizers(gen: Generator, dis: Discriminator, cfg: AttrDict
                        ) -> Tuple[optim.AdamW, optim.AdamW]:
        """Instantiates the optimisers for both models."""
        gen_betas: Tuple[float, float] = cfg.optim.gen.betas
        dis_betas: Tuple[float, float] = cfg.optim.dis.betas
        gen_opt = optim.AdamW(gen.parameters(),
                              lr=cfg.optim.gen.lr,
                              weight_decay=cfg.optim.gen.weight_decay,
                              betas=gen_betas)
        dis_opt = optim.AdamW(dis.parameters(),
                              lr=cfg.optim.dis.lr,
                              weight_decay=cfg.optim.dis.weight_decay,
                              betas=dis_betas)
        return gen_opt, dis_opt

    @property
    def gen_gpu(self) -> nn.DataParallel:  # type: ignore
        assert self._gen_gpu is not None, 'GPU disabled'
        return self._gen_gpu

    @property
    def dis_gpu(self) -> nn.DataParallel:  # type: ignore
        assert self._dis_gpu is not None, 'GPU disabled'
        return self._dis_gpu

    def call_gen(self, z_or_n: Union[Tensor, int], eval: bool = False
                 ) -> Tensor:
        """Call the generator with either the number of samples to generate or
        the Zs to use as an input. This is preferred over calling gen directly,
        as it checks the `use_gpu` flag. """
        if eval:
            self.gen.eval()
        else:
            self.gen.train()
        if isinstance(z_or_n, int):
            z = self.gen.random_z(z_or_n)
        else:
            z = z_or_n
        if self.cfg.use_gpu:
            return self.gen_gpu(z)  # type: ignore
        return self.gen(z)  # type: ignore

    def call_dis(self, batch: Tensor, eval: bool = False) -> Tensor:
        """Similar to `call_gen` but for the discriminator."""
        if eval:
            self.dis.eval()
        else:
            self.dis.train()
        if self.cfg.use_gpu:
            return self.dis_gpu(batch)  # type: ignore
        return self.dis(batch)  # type: ignore

    def train(self, iters: int = -1) -> None:
        """Train the networks until interrupted by `KeyboardInterrupt`."""
        self.sidecar.on_training_start()
        try:
            data_loader = DataLoader(self.dataset,
                                     batch_size=self.cfg.optim.batch_size,
                                     shuffle=True)
            imgs_stream = (imgs for _ in count() for imgs in data_loader)
            while True:
                self.sidecar.on_batch_start()
                dis_loss, gen_loss = self.train_batch(imgs_stream)
                self.sidecar.on_batch_stop(dis_loss, gen_loss)
                if iters == 0:
                    break
                iters -= 1
        except KeyboardInterrupt:
            # Allows the user to end training with Ctrl+c
            pass
        self.sidecar.on_training_stop()

    def train_batch(self, batches: Iterator[Tensor]) -> Tuple[float, float]:
        assert self.cfg.optim.dis_iters_per_gen_iter > 0
        for _ in range(self.cfg.optim.dis_iters_per_gen_iter):
            # Train the discriminator...
            self.dis.zero_grad()
            # ... on real images ...
            batch = next(batches)
            batch_size = batch.size(0)
            labels = torch.full((batch_size, 1), REAL, device=self._device)
            preds = self.call_dis(batch)
            dis_real_loss = F.binary_cross_entropy(preds, labels)
            dis_real_loss.backward()  # type: ignore
            # ... and on fake images.
            z = self.gen.random_z(batch_size)
            generated_batch = self.call_gen(z)
            labels = torch.full((batch_size, 1), GENERATED,
                                device=self._device)
            preds = self.call_dis(generated_batch.detach())
            dis_fake_loss = F.binary_cross_entropy(preds, labels)
            dis_fake_loss.backward()  # type: ignore
            # Update the parameters
            self.dis_opt.step()
        # Train the generator using the discriminators 'feedback'.
        self.gen.zero_grad()
        desired_labels = torch.full((batch_size, 1), REAL, device=self._device)
        preds = self.call_dis(generated_batch)
        gen_loss = F.binary_cross_entropy(preds, desired_labels)
        gen_loss.backward()  # type: ignore
        # Update the parameters
        self.gen_opt.step()
        self._gen_ema(self.gen)
        # Return the losses
        dis_loss = (dis_real_loss + dis_fake_loss) / 2.
        return float(dis_loss.item()), float(gen_loss.item())


@dataclass
class EMA():
    """ Exponential moving average of the parameters of a given module.

    Partly taken from here:
    https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-
    decay-for-variables/10856/4 """
    mu: float
    shadow: Dict[str, Tensor] = field(default_factory=dict)

    def register_param(self, name: str, val: Tensor) -> EMA:
        self.shadow[name] = val.clone()
        return self

    def register_module(self, module: nn.Module) -> EMA:
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.register_param(name, param.data)
        return self

    def __call__(self, module: nn.Module) -> None:
        if self.mu == 0.:
            return
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            delta = (1.0 - self.mu) * param.data
            residual = self.mu * self.shadow[name]
            new_average = delta + residual
            self.shadow[name] = new_average.clone()
            param.data = new_average
