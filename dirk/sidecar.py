from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from logging import getLogger

from .utils import AttrDict
from . import training


logger = getLogger('sidecar')


@dataclass
class Sidecar:
    cfg: AttrDict
    workdir: Path
    iteration: int = 0
    _trainer: Optional[training.Trainer] = None

    @property
    def trainer(self) -> training.Trainer:
        assert self._trainer is not None, 'Sidecar not yet attached'
        return self._trainer

    def attach(self, trainer: training.Trainer) -> None:
        assert trainer.sidecar == self, 'Trainer already has different Sidecar'
        assert self._trainer is None, 'Sidecar already attached'
        self._trainer = trainer

    @classmethod
    def from_workdir(cls, workdir: Path) -> Sidecar:
        cfg = AttrDict.from_yaml(workdir / 'config.yaml')
        checkpoint = workdir / 'latest.cp'
        if checkpoint.exists():
            return cls.load(checkpoint)
        sc = Sidecar(cfg, workdir)
        training.Trainer.from_config(cfg, sc)
        return sc

    @classmethod
    def load(cls, checkpoint: Path) -> Sidecar:
        state = torch.load(checkpoint)  # type: ignore
        assert isinstance(state, AttrDict)
        sc_state = state.sidecar
        iteration = sc_state.iteration
        workdir = sc_state.workdir
        cfg = AttrDict.from_yaml(checkpoint.parent / 'config.yaml')
        sc = cls(cfg, workdir, iteration)
        training.Trainer.from_state(cfg, sc, state.trainer)
        logger.info(f'Loaded from {checkpoint}')
        return sc

    def save(self, checkpoint: Optional[Path] = None) -> None:
        checkpoint_dir = self.workdir / 'checkpoints'
        if checkpoint is None:
            name = f'{str(self.iteration).zfill(6)}.cp'
            checkpoint = checkpoint_dir / name
        state = AttrDict(trainer=self.trainer.get_state())
        state.sidecar = AttrDict(iteration=self.iteration,
                                 workdir=self.workdir)
        if checkpoint.parent == checkpoint_dir:
            symlink = self.workdir / 'latest.cp'
            if symlink.exists():
                symlink.unlink()
            dest = Path('checkpoints') / checkpoint.name
            symlink.symlink_to(dest)
            logger.info(f'Pointed {symlink} to {dest}')
        logger.info(f'Saved to {checkpoint}')
        torch.save(state, checkpoint)

    def on_batch_start(self) -> None:
        pass

    def on_batch_stop(self, dis_loss: float, gen_loss: float) -> None:
        logger.info(f'It: {self.iteration}, Dis: {dis_loss}, Gen: {gen_loss}')
        self.iteration += 1

    def on_training_start(self) -> None:
        logger.info('Started training')

    def on_training_stop(self) -> None:
        logger.info('Stopped training')
