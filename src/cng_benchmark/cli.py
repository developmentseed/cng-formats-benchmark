"""Command-line entry point for the benchmark runner.

The runner is the containerised batch component of the system. This CLI is a
thin shell over the runner: it loads and validates a config and either profiles
an existing object listing or runs the COG end-to-end path (convert a baseline
raster, then collect write/object-size/read/display metrics against the produced
object), persisting a result artifact. ``seed`` generates a synthetic fixture
COG so the deployable stack can prove itself end-to-end without external data.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from cng_benchmark import __version__, storage
from cng_benchmark.config import load_benchmark_config
from cng_benchmark.report import write_artifacts
from cng_benchmark.runner import run_benchmark, run_conversion_benchmark

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


def _emit(result, output_uri: str | None) -> None:
    """Print the run as JSON and, if an output URI is given, write artifacts."""
    typer.echo(result.model_dump_json(indent=2))
    if output_uri is not None:
        try:
            written = write_artifacts(result, output_uri)
        except (OSError, ValueError, RuntimeError) as exc:
            typer.echo(f"Cannot write artifacts to {output_uri}: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Wrote {written['result']} and {written['summary']}", err=True)


@app.command()
def run(
    config: str,
    source: str | None = typer.Option(
        None,
        "--source",
        help="Baseline raster URI to convert (COG end-to-end path). Overrides "
        "the config's source.",
    ),
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
        help="Storage location to write result.json + summary.md (and, on the "
        "conversion path, the produced object). Overrides the config's output.",
    ),
    titiler_endpoint: str | None = typer.Option(
        None,
        "--titiler-endpoint",
        envvar="TITILER_ENDPOINT",
        help="Base URL of the TiTiler service for the display metric.",
    ),
) -> None:
    """Run a benchmark described by a config file.

    Two paths, selected by inputs:

    * Conversion (COG end-to-end): with ``--source`` (or the config's
      ``source``), convert the baseline and run the configured metrics
      (write/object_size/read/display) against the produced object. Requires an
      output location to publish the object and artifacts to.
    * Object-only: with ``--objects`` (a JSON listing) or ``--objects-uri`` /
      ``object_source`` (a storage location to list), profile object sizes.

    With none of these, the command validates the config and exits — the
    deployable runner's stable no-IO entry point. The result is printed as JSON;
    with an output location, ``result.json`` and ``summary.md`` are written too.
    """
    try:
        cfg = load_benchmark_config(config)
    except Exception as exc:  # noqa: BLE001 — surface any load/validation error cleanly
        typer.echo(f"Invalid config {config}: {exc}", err=True)
        raise typer.Exit(1) from exc

    source_raster = source or cfg.source
    object_listing = objects_uri or cfg.object_source
    output_uri = output or cfg.output

    if source_raster is not None:
        if output_uri is None:
            typer.echo(
                "The conversion path requires --output (or config output).", err=True
            )
            raise typer.Exit(1)
        try:
            result = run_conversion_benchmark(
                cfg, source_raster, output_uri, titiler_endpoint=titiler_endpoint
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            typer.echo(
                f"Conversion benchmark failed for {source_raster}: {exc}", err=True
            )
            raise typer.Exit(1) from exc
        _emit(result, output_uri)
        return

    if objects is None and object_listing is None:
        typer.echo(f"Config {config} is valid (benchmark id: {cfg.id}).")
        typer.echo(
            "To run, give an input: --source / --objects / --objects-uri, or set "
            "source / object_source in the config."
        )
        return

    origin = objects if objects is not None else object_listing
    try:
        if objects is not None:
            sizes = _load_object_sizes(objects)
        else:
            sizes = storage.list_object_sizes(object_listing)
        result = run_benchmark(cfg, sizes)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        typer.echo(f"Cannot profile objects from {origin}: {exc}", err=True)
        raise typer.Exit(1) from exc

    _emit(result, output_uri)


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
