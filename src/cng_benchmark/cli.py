"""Command-line entry point for the benchmark runner.

The runner is the containerised batch component of the system. This CLI is a
thin shell over :func:`cng_benchmark.runner.run_benchmark`: it loads and
validates a config and, given a local object listing, emits an object-size
profile. The metric collectors that need live services (TiTiler, object
storage) are wired in with the deployable stack (M2); nothing here performs a
real conversion or benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from cng_benchmark import __version__
from cng_benchmark.config import load_benchmark_config
from cng_benchmark.runner import run_benchmark

app = typer.Typer(
    help="Benchmark cloud-native geospatial formats.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the runner version."""
    typer.echo(__version__)


def _load_object_sizes(path: str) -> list[int]:
    """Read object sizes from a JSON listing.

    The file holds either a list of integer byte sizes, or a list of
    ``{"name": ..., "size": ...}`` objects (the ``name`` is ignored here).
    Raises :class:`ValueError` for any malformed entry, identifying it.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("object listing must be a JSON list")
    sizes: list[int] = []
    for i, entry in enumerate(data):
        try:
            sizes.append(int(entry["size"] if isinstance(entry, dict) else entry))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"object listing entry {i} is not a size or {{name, size}}: {entry!r}"
            ) from exc
    return sizes


@app.command()
def run(
    config: str,
    objects: str | None = typer.Option(
        None,
        "--objects",
        help="Path to a JSON object listing to profile (sizes or {name,size}).",
    ),
) -> None:
    """Run a benchmark described by a config file.

    Loads and validates the config. With ``--objects``, profiles the listed
    object sizes and prints the resulting BenchmarkRun as JSON. Without it, the
    command validates the config and exits (no IO), so the deployable runner has
    a stable entry point.
    """
    try:
        cfg = load_benchmark_config(config)
    except Exception as exc:  # noqa: BLE001 — surface any load/validation error cleanly
        typer.echo(f"Invalid config {config}: {exc}", err=True)
        raise typer.Exit(1) from exc

    if objects is None:
        typer.echo(f"Config {config} is valid (benchmark id: {cfg.id}).")
        typer.echo("Provide --objects <listing.json> to emit an object profile.")
        return

    try:
        sizes = _load_object_sizes(objects)
        result = run_benchmark(cfg, sizes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Cannot profile objects from {objects}: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
