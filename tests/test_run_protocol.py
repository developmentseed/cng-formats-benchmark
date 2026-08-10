"""Tests for the run protocol: replicates, cold/warm cache, concurrency (#87).

``params.run_protocol`` is opt-in and purely additive (``run.conditions``);
without it, behaviour is unchanged (see :mod:`tests.test_runner` for that
path). These tests exercise the new machinery end-to-end against the
synthetic COG fixture, so they spawn real subprocesses for cold/concurrent
conditions — kept to small replicate/worker counts to bound runtime.
"""

import pytest

from cng_benchmark.config import load_benchmark_config
from cng_benchmark.runner import _parse_run_protocol, run_conversion_benchmark

SYNTHETIC = "configs/benchmarks/synthetic_cog.yaml"


def _synthetic_source(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    from cng_benchmark.fixtures import generate_cog_bytes

    source = tmp_path / "source.tif"
    source.write_bytes(generate_cog_bytes(size=128, blocksize=128))
    return source


def _benchmark(metrics, params):
    return load_benchmark_config(SYNTHETIC).model_copy(
        update={"metrics": metrics, "params": params}
    )


# --- config parsing -----------------------------------------------------


def test_parse_run_protocol_defaults_to_one_warm_isolated_condition():
    replicates, conditions = _parse_run_protocol({})
    assert replicates == 1
    assert len(conditions) == 1
    assert conditions[0].cache == "warm"
    assert conditions[0].concurrency == 1


def test_parse_run_protocol_reads_replicates_and_conditions():
    replicates, conditions = _parse_run_protocol(
        {
            "run_protocol": {
                "replicates": 3,
                "conditions": [
                    {"cache": "warm", "concurrency": 1},
                    {"cache": "cold", "concurrency": 4},
                ],
            }
        }
    )
    assert replicates == 3
    assert [(c.cache, c.concurrency) for c in conditions] == [("warm", 1), ("cold", 4)]


@pytest.mark.parametrize(
    "run_protocol",
    [
        {"replicates": 0},
        {"conditions": [{"cache": "lukewarm"}]},
        {"conditions": [{"concurrency": 0}]},
    ],
)
def test_parse_run_protocol_rejects_bad_values(run_protocol):
    with pytest.raises(ValueError):
        _parse_run_protocol({"run_protocol": run_protocol})


# --- end-to-end: no run_protocol leaves conditions empty -----------------


def test_no_run_protocol_leaves_conditions_empty(tmp_path):
    source = _synthetic_source(tmp_path)
    cfg = _benchmark(["write", "object_size", "read"], {"block_size": 128})
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))
    assert run.conditions == []


# --- end-to-end: warm isolated (fast path, no subprocess overhead) -------


def test_warm_isolated_replicates_report_spread(tmp_path):
    source = _synthetic_source(tmp_path)
    cfg = _benchmark(
        ["write", "object_size", "read"],
        {
            "block_size": 128,
            "run_protocol": {
                "replicates": 3,
                "conditions": [{"cache": "warm", "concurrency": 1}],
            },
        },
    )
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))

    assert len(run.conditions) == 1
    condition = run.conditions[0]
    assert condition.phase == "read"
    assert condition.cache == "warm"
    assert condition.concurrency == 1
    assert [r.index for r in condition.replicates] == [0, 1, 2]
    for replicate in condition.replicates:
        assert any(m.name == "read_latency_mean" for m in replicate.metrics)

    latency = next(m for m in condition.aggregate if m.name == "read_latency_mean")
    assert len(latency.detail["replicate_values"]) == 3
    assert latency.detail["median"] == pytest.approx(
        sorted(latency.detail["replicate_values"])[1]
    )
    assert latency.detail["stdev"] >= 0

    # run.metrics is untouched by the run protocol.
    assert sum(m.name == "read_window_count" for m in run.metrics) == 1


# --- end-to-end: cold isolated (spawns a fresh subprocess per replicate) --


def test_cold_isolated_spawns_fresh_replicates(tmp_path):
    source = _synthetic_source(tmp_path)
    cfg = _benchmark(
        ["write", "object_size", "read"],
        {
            "block_size": 128,
            "run_protocol": {
                "replicates": 2,
                "conditions": [{"cache": "cold", "concurrency": 1}],
            },
        },
    )
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))

    assert len(run.conditions) == 1
    condition = run.conditions[0]
    assert condition.cache == "cold"
    assert len(condition.replicates) == 2
    latency = next(m for m in condition.aggregate if m.name == "read_latency_mean")
    assert len(latency.detail["replicate_values"]) == 2


# --- end-to-end: concurrency reports a per-worker breakdown --------------


def test_concurrency_pools_and_reports_per_worker_values(tmp_path):
    source = _synthetic_source(tmp_path)
    cfg = _benchmark(
        ["write", "object_size", "read"],
        {
            "block_size": 128,
            "run_protocol": {
                "replicates": 1,
                "conditions": [{"cache": "warm", "concurrency": 3}],
            },
        },
    )
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))

    condition = run.conditions[0]
    assert condition.concurrency == 3
    (replicate,) = condition.replicates
    latency = next(m for m in replicate.metrics if m.name == "read_latency_mean")
    assert latency.detail["workers"] == 3
    assert len(latency.detail["worker_values"]) == 3


# --- multiple conditions in one run ---------------------------------------


def test_multiple_conditions_each_produce_a_result(tmp_path):
    source = _synthetic_source(tmp_path)
    cfg = _benchmark(
        ["write", "object_size", "read"],
        {
            "block_size": 128,
            "run_protocol": {
                "replicates": 1,
                "conditions": [
                    {"cache": "warm", "concurrency": 1},
                    {"cache": "cold", "concurrency": 1},
                ],
            },
        },
    )
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))

    assert [(c.cache, c.concurrency) for c in run.conditions] == [
        ("warm", 1),
        ("cold", 1),
    ]


# --- display: replicates only, always warm/isolated -----------------------


def test_display_run_protocol_replicates_and_stays_warm_isolated(tmp_path):
    source = _synthetic_source(tmp_path)
    cfg = _benchmark(
        ["display"],
        {
            "block_size": 128,
            "run_protocol": {
                "replicates": 2,
                # A cold/concurrent condition is requested, but display can't
                # honour it (no TiTiler endpoint here either) — it is
                # best-effort and reports a single skipped condition.
                "conditions": [{"cache": "cold", "concurrency": 4}],
            },
        },
    )
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))

    assert len(run.conditions) == 1
    condition = run.conditions[0]
    assert condition.phase == "display"
    assert condition.cache == "warm"
    assert condition.concurrency == 1
