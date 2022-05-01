from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from logging import getLogger
import wandb

from .utils import AttrDict
from . import training


logger = getLogger('sidecar')


@dataclass
class Sidecar:
    cfg: AttrDict
    workdir: Path
    iteration: int = 0
    _trainer: Optional[training.Trainer] = None
    _wandb_ext: Optional[WandBExtension] = None

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
        trainer = training.Trainer.from_config(cfg, sc)
        if cfg.wandb.enable:
            sc._wandb_ext = WandBExtension(trainer, None)
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
        trainer = training.Trainer.from_state(cfg, sc, state.trainer)
        if cfg.wandb.enable:
            sc._wandb_ext = WandBExtension.from_state(trainer, state.wandb)
        logger.info(f'Loaded from {checkpoint}')
        return sc

    def save(self, checkpoint: Optional[Path] = None) -> None:
        checkpoint_dir = self.workdir / 'checkpoints'
        if checkpoint is None:
            name = f'{str(self.iteration).zfill(6)}.cp'
            checkpoint = checkpoint_dir / name
        state = AttrDict(trainer=self.trainer.get_state())
        if self.cfg.wandb.enable:
            assert self._wandb_ext is not None
            state.wandb = self._wandb_ext.get_state()
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
        if self.periodically(self.cfg.log.log_freq, skip_first=True):
            logger.info(f'It: {self.iteration}, '
                        f'Dis: {dis_loss}, '
                        f'Gen: {gen_loss}')
        if self.periodically(self.cfg.log.save_freq):
            self.save()
        if self._wandb_ext is not None:
            self._wandb_ext.on_batch_stop()
        self.iteration += 1

    def on_training_start(self) -> None:
        if self._wandb_ext is not None:
            self._wandb_ext.on_training_start()
        logger.info('Started training')

    def on_training_stop(self) -> None:
        if self._wandb_ext is not None:
            self._wandb_ext.on_training_stop()
        self.save()
        logger.info('Stopped training')

    def periodically(self, freq: int, skip_first: bool = False) -> bool:
        if self.iteration == 0 and skip_first:
            return False
        return self.iteration % freq == 0


@dataclass
class WandBExtension:
    trainer: training.Trainer
    trainer_id: Any = None

    @property
    def sidecar(self) -> Sidecar:
        return self.trainer.sidecar

    def on_training_start(self) -> None:
        self.wandb_trainer = wandb.init(
            project=self.trainer.cfg.wandb.project,
            group=self.trainer.cfg.wandb.group,
            name=self.sidecar.workdir.resolve().name,
            tags=self.sidecar.cfg.wandb.tags,
            id=self.trainer_id,
            resume=self.trainer_id is not None,
            notes=(str(self.trainer.gen)
                   + str(self.trainer.dis)
                   + str(self.sidecar.cfg)))

    def on_batch_stop(self) -> None:
        log_freq = self.trainer.cfg.wandb.log_frequency
        if self.sidecar.iteration % log_freq != 0:
            return
        self.trainer.gen.eval()
        with torch.no_grad():
            z = self.trainer.gen.random_z(8)
            gen_imgs = self.trainer.call_gen(z)
        gen_imgs = gen_imgs.cpu().detach().numpy()
        gen_imgs = gen_imgs.transpose((0, 2, 3, 1))  # type: ignore
        log_dict = {
            # 'Dis Loss': self.trainer.dis_rolling_loss.value,
            # 'Gen Loss': self.trainer.gen_rolling_loss.value,
            'Imgs': [wandb.Image(gen_img) for gen_img in gen_imgs],
            # 'S/Iter': self.trainer.rolling_sec_per_iter.value,
        }
        self.wandb_trainer.log(log_dict, step=self.sidecar.iteration)
        logger.debug('Logged to WandB')

    def on_training_stop(self) -> None:
        self.wandb_trainer.finish()

    def get_state(self) -> AttrDict:
        return AttrDict(trainer_id=self.wandb_trainer.id)

    @classmethod
    def from_state(cls, trainer: training.Trainer, state: AttrDict
                   ) -> WandBExtension:
        return cls(trainer, state.trainer_id)
