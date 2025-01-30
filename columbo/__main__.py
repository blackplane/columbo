import click
from columbo.runners import run_embeddings, run_training
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.argument('source')
@click.argument('dest')
def embed(source: str, dest: str):
    logger.info("Starting 'embeddings' ...")
    run_embeddings(src_csv=source, dest_csv=dest)


@click.command()
@click.option('-e', '--epochs', type=int)
def train(epochs: int):
    logger.info("Starting 'train' ...")
    run_training(epochs)


@click.group()
@click.version_option()
def cli():
    pass


def main():
    cli.add_command(embed)
    cli.add_command(train)
    cli()


main()
