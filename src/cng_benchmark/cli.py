"""Command-line entry point for the benchmark runner.

The runner is the containerised batch component of the system. This CLI is a
thin shell over :func:`cng_benchmark.runner.run_benchmark`: it loads and
validates a config, sources the objects to profile (from a local listing or a
storage location), and persists a result artifact. ``seed`` generates a
synthetic fixture COG so the deployable stack can prove itself end-to-end
without external data. The collectors that need live services (TiTiler) are
wired into the COG path in a follow-up; nothing here reaches real data in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from cng_benchmark import __version__, storage
from cng_benchmark.config import load_benchmark_config
from cng_benchmark.report import write_artifacts
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
    objects_uri: str | None = typer.Option(
        None,
        "--objects-uri",
        help="Storage location (local dir, file://, s3://prefix) to list and "
        "profile. Overrides the config's object_source.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Storage location to write result.json + summary.md. Overrides the "
        "config's output.",
    ),
) -> None:
    """Run a benchmark described by a config file.

    Resolves the objects to profile in priority order: ``--objects`` (a JSON
    listing), then ``--objects-uri`` or the config's ``object_source`` (a
    storage location to list). With neither, the command just validates the
    config and exits — the deployable runner's stable no-IO entry point.

    The result is always printed as JSON; if an output location is given (via
    ``--output`` or the config), ``result.json`` and ``summary.md`` are also
    written there.
    """
    try:
        cfg = load_benchmark_config(config)
    except Exception as exc:  # noqa: BLE001 — surface any load/validation error cleanly
        typer.echo(f"Invalid config {config}: {exc}", err=True)
        raise typer.Exit(1) from exc

    source_uri = objects_uri or cfg.object_source
    if objects is None and source_uri is None:
        typer.echo(f"Config {config} is valid (benchmark id: {cfg.id}).")
        typer.echo("Provide --objects, --objects-uri, or an object_source to run.")
        return

    origin = objects if objects is not None else source_uri
    try:
        if objects is not None:
            sizes = _load_object_sizes(objects)
        else:
            sizes = storage.list_object_sizes(source_uri)
        result = run_benchmark(cfg, sizes)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        typer.echo(f"Cannot profile objects from {origin}: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(result.model_dump_json(indent=2))

    output_uri = output or cfg.output
    if output_uri is not None:
        try:
            written = write_artifacts(result, output_uri)
        except (OSError, ValueError, RuntimeError) as exc:
            typer.echo(f"Cannot write artifacts to {output_uri}: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Wrote {written['result']} and {written['summary']}", err=True)


@app.command()
def seed(
    dest: str = typer.Option(
        ...,
        "--dest",
        help="Destination URI for the fixture COG (local path, file://, or "
        "s3://bucket/key).",
    ),
    size: int = typer.Option(512, "--size", help="Fixture raster width/height (px)."),
) -> None:
    """Generate a small synthetic fixture COG and write it to ``--dest``.

    Used by the deployable stack to seed object storage so the runner can prove
    itself end-to-end without external data. Requires the ``cog`` extra.
    """
    from cng_benchmark.fixtures import generate_cog_bytes

    try:
        data = generate_cog_bytes(size=size)
        storage.write_bytes(dest, data)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Cannot seed fixture to {dest}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Seeded {len(data)} byte fixture COG to {dest}", err=True)


if __name__ == "__main__":
    app()
