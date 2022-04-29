# type: ignore
import click
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BatchConfig:
    output: Path
    hours: int
    minutes: int = 0
    cpus: int = 4
    mem: int = 10  # in GB


def submit_job(cfg: BatchConfig) -> None:
    cmd = ['sbatch',
           f'--time={str(cfg.hours).zfill(2)}:{str(cfg.minutes).zfill(2)}:00',
           '--gres=gpu:1',
           '--partition=gpu'
           f'--mem={cfg.mem}gb',
           '--ntasks=1',
           f'--cpus-per-task={cfg.cpus}',
           f'--output={cfg.output}',
          ]
    print(cmd)
    # sp.run(cmd)


if __name__ == '__main__':
    main()
