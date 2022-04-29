import click
from logging import getLogger
from pathlib import Path
import shutil
from typing import Optional


logger = getLogger('apps')


@click.option('--name', '-n', type=str, default=None)
@click.option('--zoo', '-z', type=Path, default=Path('./zoo/'))
@click.option('--config', '-c', type=Path,
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
    # Point 'latest' symlink in zoo to the newly created project
    symlink = zoo / 'latest'
    if symlink.exists():
        symlink.unlink()
    symlink.symlink_to(name)
    logger.info(f'Pointed {symlink} to {wd}')


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
