import click
from columbo.runners import run_embeddings, run_training, run_eval
import logging
from rich.logging import RichHandler

logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.argument('source')
@click.argument('dest')
def embed(source: str, dest: str):
    logger.info("Starting [bold blue]embeddings[/] ...", extra={"markup": True})
    run_embeddings(src_csv=source, dest_parquet=dest)


@click.command()
@click.option('-e', '--epochs', type=int)
@click.option('-c', '--classifier', type=str)
def train(epochs: int, classifier: str):
    logger.info("Starting [bold blue]train[/] ...", extra={"markup": True})
    run_training(epochs, classifier)


@click.command()
@click.option('-p', '--path', type=str)
@click.option('-r', '--run', type=str)
@click.option('-c', '--classifier', type=str)
def eval(path: str, classifier: str, run: str):
    logger.info("Starting [bold blue]eval[/] ...", extra={"markup": True})
    run_eval(path, classifier, "cpu", run)


@click.group()
@click.version_option()
def cli():
    pass


def main():
    cli.add_command(embed)
    cli.add_command(train)
    cli.add_command(eval)
    cli()


main()
