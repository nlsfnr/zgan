import click
from logging import getLogger
from pathlib import Path
import shutil
import torch
from typing import Optional

from .sidecar import Sidecar
from .utils import AttrDict, show_grid
from . import models


logger = getLogger('apps')


@click.option('--name', type=str, default=None)
@click.option('--zoo', type=Path, default=Path('./zoo/'))
@click.option('--config', type=Path,
              default=Path('./configs/default.yaml'))
def create_proj(name: Optional[str], zoo: Path, config: Path) -> None:
    if name is None:
        name = random_name()
    wd = zoo / name
    if wd.exists():
        raise ValueError(f'Path already exists: {wd}')
    logger.info(f'Creating new project: Name: {name}, Zoo: {zoo}, '
                f'Config: {config}, Working directory: {wd}')
    wd.mkdir(parents=True)
    shutil.copy(config, wd / 'config.yaml')
    (wd / 'checkpoints').mkdir()
    _point_latest_to(zoo, name)


@click.argument('name', type=str)
@click.option('--zoo', type=Path, default=Path('./zoo/'))
def focus(name: str, zoo: Path) -> None:
    if name == 'latest':
        raise ValueError('Name can not be "latest"')
    _point_latest_to(zoo, name)


@click.option('--name', type=str, default='latest')
@click.option('--zoo', type=Path, default=Path('./zoo/'))
def train(name: str, zoo: Path) -> None:
    wd = zoo / name
    if not wd.exists():
        raise FileNotFoundError(f'Project not found: {wd}')
    sc = Sidecar.from_workdir(wd)
    trainer = sc.trainer
    trainer.train()


@click.option('--name', type=str, default='latest')
@click.option('--zoo', type=Path, default=Path('./zoo/'))
@click.option('-n', type=int, default=16)
def show(name: str, zoo: Path, n: int) -> None:
    wd = zoo / name
    if not wd.exists():
        raise FileNotFoundError(f'Project not found: {wd}')
    cfg = AttrDict.from_yaml(wd / 'config.yaml')
    checkpoint = wd / 'latest.cp'
    logger.info(f'Loading Generator from {checkpoint}')
    state = torch.load(checkpoint)  # type: ignore
    assert isinstance(state, AttrDict)
    gen_state = state.trainer.gen
    gen = models.Generator(cfg)
    gen.load_state_dict(gen_state)
    logger.info(f'Generating {n} images')
    z = gen.random_z(n)
    imgs = gen(z)
    show_grid(imgs)


def _point_latest_to(zoo: Path, name: str) -> None:
    if name == 'latest':
        return
    wd = zoo / name
    if not wd.exists():
        raise FileNotFoundError(f'Project does not exist: {wd}')
    symlink = zoo / 'latest'
    if symlink.exists():
        if wd.resolve() == symlink.resolve():
            return
        symlink.unlink()
    symlink.symlink_to(name)
    logger.info(f'Pointed {symlink} to {zoo / name}')


def random_name(prefix: str = 'proj') -> str:
    counter_file = Path('.name_counter')
    if counter_file.exists():
        with open(counter_file, 'r') as fh:
            number = int(fh.read())
    else:
        number = 0
    with open(counter_file, 'w') as fh:
        fh.write(str(number + 1))
    return f'{prefix}-{str(number).zfill(4)}'
