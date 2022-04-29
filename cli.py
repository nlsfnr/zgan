#!/usr/bin/env python3
import click
import logging

import dirk.apps as apps


@click.group()
def cli() -> None:
    fmt = '[%(asctime)s|%(name)s|%(levelname)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)


cli.command('create')(apps.create_proj)
cli.command('train')(apps.train)
cli.command('focus')(apps.focus)


if __name__ == '__main__':
    cli()
