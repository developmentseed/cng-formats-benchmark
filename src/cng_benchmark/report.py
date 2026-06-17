"""Result artifacts: the JSON record and a human-readable Markdown summary.

A deployed run produces two artifacts under its configured output location: the
machine-readable ``result.json`` (the full :class:`~cng_benchmark.models.BenchmarkRun`)
and a compact ``summary.md`` for humans skimming a results bucket. Rendering is
pure and stdlib-only, so it is fully unit-testable; persistence is delegated to
:mod:`cng_benchmark.storage`, which handles both local paths and S3.
"""

from __future__ import annotations

from cng_benchmark import storage
from cng_benchmark.models import BenchmarkRun

RESULT_FILENAME = "result.json"
SUMMARY_FILENAME = "summary.md"


def _format_bytes(n: float) -> str:
    """Render a byte count with a binary unit suffix (KiB, MiB, …)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} {units[-1]}"  # pragma: no cover - unreachable


def render_markdown_summary(run: BenchmarkRun) -> str:
    """Render a compact Markdown summary of a :class:`BenchmarkRun`."""
    lines: list[str] = [
        f"# Benchmark result: {run.dataset_id} → {run.format_id}",
        "",
        f"- **Timestamp:** {run.timestamp.isoformat()}",
        f"- **Dataset:** `{run.dataset_id}`",
        f"- **Format:** `{run.format_id}`",
    ]
    versions = ", ".join(f"{k} {v}" for k, v in sorted(run.tool_versions.items()))
    if versions:
        lines.append(f"- **Tool versions:** {versions}")

    profile = run.object_profile
    if profile is not None:
        lines += [
            "",
            "## Object-size profile",
            "",
            f"- **Objects:** {profile.count}",
            f"- **Total:** {_format_bytes(profile.total_bytes)}",
            f"- **Mean:** {_format_bytes(profile.mean)}",
            f"- **Median / p90 / p99:** {_format_bytes(profile.median)} / "
            f"{_format_bytes(profile.p90)} / {_format_bytes(profile.p99)}",
            f"- **Min / max:** {_format_bytes(profile.min_bytes)} / "
            f"{_format_bytes(profile.max_bytes)}",
            f"- **Tier fit:** {', '.join(profile.tier_fit) or 'none'}"
            f" (highest: {profile.highest_tier or 'none'})",
        ]

    if run.metrics:
        lines += ["", "## Metrics", ""]
        lines += ["| Metric | Value | Unit |", "| --- | --- | --- |"]
        for m in run.metrics:
            lines.append(f"| {m.name} | {m.value:g} | {m.unit or ''} |")

    lines.append("")
    return "\n".join(lines)


def write_artifacts(run: BenchmarkRun, output_uri: str) -> dict[str, str]:
    """Write ``result.json`` and ``summary.md`` under ``output_uri``.

    ``output_uri`` is treated as a directory/prefix (local path, ``file://`` or
    ``s3://bucket/prefix``). Returns the URIs of the artifacts written.
    """
    result_uri = storage.join(output_uri, RESULT_FILENAME)
    summary_uri = storage.join(output_uri, SUMMARY_FILENAME)
    storage.write_text(result_uri, run.model_dump_json(indent=2))
    storage.write_text(summary_uri, render_markdown_summary(run))
    return {"result": result_uri, "summary": summary_uri}
