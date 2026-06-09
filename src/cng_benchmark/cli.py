"""Command-line entry point for the benchmark runner.

The runner is the containerised batch component of the system. In this
milestone the CLI exposes only its shape; the metric collectors, config
loading, and result emission land in later milestones.
"""

from __future__ import annotations

import typer

from cng_benchmark import __version__

app = typer.Typer(
    help="Benchmark cloud-native geospatial formats.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the runner version."""
    typer.echo(__version__)


@app.command()
def run(config: str) -> None:
    """Run a benchmark described by a config file.

    Execution is not implemented in this milestone; this command exists so
    the deployable runner image has a stable entry point to wire the stack
    against.
    """
    typer.echo(f"Would run benchmark from config: {config}")
    typer.echo("Benchmark execution is not implemented in this milestone.")


if __name__ == "__main__":
    app()
