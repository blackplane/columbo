import click
from columbo.runners import run_embeddings
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
def embed():
    logger.info("Starting 'embeddings' ...")


@click.command()
def train():
    logger.info("Starting 'train' ...")


@click.group()
@click.version_option()
def cli():
    pass


def main():
    cli.add_command(embed)
    cli.add_command(train)
    cli()


main()
