"""Tests for the runner-image capability contract and the check-drivers command.

The probes (``check_capability``) are tested by monkeypatching the GDAL/OGR driver
lookups, so they run in CI without rasterio/pyogrio installed; the contract itself
(``REQUIRED``) is asserted to still declare the implicit, droppable drivers the
issue was about.
"""

from typer.testing import CliRunner

from cng_benchmark import drivers
from cng_benchmark.cli import app
from cng_benchmark.drivers import (
    GDAL_RASTER,
    OGR_VECTOR,
    PYTHON,
    Capability,
    check_all,
    check_capability,
)

runner = CliRunner()


def test_gdal_raster_probe_uses_rasterio_drivers(monkeypatch):
    monkeypatch.setattr(drivers, "_gdal_raster_drivers", lambda: {"GTiff", "netCDF"})
    present, detail = check_capability(Capability("a", GDAL_RASTER, "netCDF", "why"))
    assert present is True
    assert detail == "rasterio GDAL"
    absent, _ = check_capability(Capability("a", GDAL_RASTER, "HDF5", "why"))
    assert absent is False


def test_ogr_vector_probe_uses_pyogrio_drivers(monkeypatch):
    monkeypatch.setattr(drivers, "_ogr_drivers", lambda: {"ESRI Shapefile"})
    present, detail = check_capability(
        Capability("a", OGR_VECTOR, "ESRI Shapefile", "why")
    )
    assert present is True
    assert detail == "pyogrio OGR"
    assert not check_capability(Capability("a", OGR_VECTOR, "GPKG", "why"))[0]


def test_python_probe_uses_find_spec():
    assert check_capability(Capability("a", PYTHON, "json", "why"))[0] is True
    missing = check_capability(Capability("a", PYTHON, "no_such_module_xyz", "why"))
    assert missing[0] is False


def test_probe_reports_missing_extra_without_raising(monkeypatch):
    # If the probe stack itself is absent, the capability reports not-present
    # rather than crashing the whole report.
    def _boom():
        raise ModuleNotFoundError("rasterio")

    monkeypatch.setattr(drivers, "_gdal_raster_drivers", _boom)
    present, detail = check_capability(Capability("a", GDAL_RASTER, "netCDF", "why"))
    assert present is False
    assert "unavailable" in detail


def test_required_declares_the_implicit_drivers():
    pairs = {(c.arm, c.kind, c.name) for c in drivers.REQUIRED}
    # The two drivers the issue was about — bundled in a wheel, previously undeclared.
    assert ("swot-raster100m", GDAL_RASTER, "netCDF") in pairs
    assert ("geoparquet", OGR_VECTOR, "ESRI Shapefile") in pairs
    # And the point-cloud arm's write/read libraries.
    assert ("copc", PYTHON, "copclib") in pairs
    assert ("copc", PYTHON, "laspy") in pairs


def test_check_all_returns_one_row_per_capability():
    caps = (
        Capability("x", PYTHON, "json", "stdlib"),
        Capability("y", PYTHON, "no_such_module_xyz", "missing"),
    )
    rows = check_all(caps)
    assert [(cap.name, present) for cap, present, _ in rows] == [
        ("json", True),
        ("no_such_module_xyz", False),
    ]


def test_check_drivers_command_passes_when_all_present(monkeypatch):
    caps = (Capability("x", PYTHON, "json", "stdlib"),)
    monkeypatch.setattr(drivers, "REQUIRED", caps)
    result = runner.invoke(app, ["check-drivers"])
    assert result.exit_code == 0
    assert "All 1 runner-image capabilities present." in result.stdout


def test_check_drivers_command_fails_on_missing(monkeypatch):
    caps = (
        Capability("x", PYTHON, "json", "stdlib"),
        Capability("y", PYTHON, "no_such_module_xyz", "missing"),
    )
    monkeypatch.setattr(drivers, "REQUIRED", caps)
    result = runner.invoke(app, ["check-drivers"])
    assert result.exit_code == 1
    assert "MISSING" in result.stdout
    assert "no_such_module_xyz" in result.stdout
