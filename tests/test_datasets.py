"""Tests for the dataset hierarchy, registry, and the MAJA reader."""

import pytest
from pydantic import ValidationError

from cng_benchmark.config import DatasetConfig
from cng_benchmark.datasets import Product, SourceObject, build_dataset
from cng_benchmark.datasets.sentinel2 import Sentinel2MajaDataset
from cng_benchmark.datasets.single_object import SingleObjectDataset
from cng_benchmark.registry import DATASETS


def _dataset_config(**overrides) -> DatasetConfig:
    base = {
        "id": "ds",
        "source": "s3://bucket/scene.tif",
        "baseline_format": "geotiff",
        "target_formats": ["cog"],
    }
    base.update(overrides)
    return DatasetConfig.model_validate(base)


def test_builtin_readers_registered():
    for name in ("single-object", "sentinel2-maja"):
        assert name in DATASETS


def test_config_defaults_to_single_object_reader():
    cfg = _dataset_config()
    assert cfg.reader == "single-object"
    assert cfg.options == {}


def test_single_object_one_product_one_component():
    ds = build_dataset(_dataset_config())
    assert isinstance(ds, SingleObjectDataset)
    products = ds.products()
    assert products == [
        Product(
            id="ds", components=[SourceObject(name="ds", uri="s3://bucket/scene.tif")]
        )
    ]


def test_build_dataset_unknown_reader_raises():
    with pytest.raises(KeyError):
        build_dataset(_dataset_config(reader="nonesuch"))


def test_options_validated_against_reader_model():
    # An unknown option key is rejected by the reader's typed Options model.
    with pytest.raises(ValidationError):
        build_dataset(_dataset_config(reader="sentinel2-maja", options={"bogus": 1}))


# A captured listing of the relevant members of a MAJA L2A V4-0 zip.
MAJA_MEMBERS = [
    "SENTINEL2A_20200101_L2A_T31TCJ_FRE_B2.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_FRE_B3.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_FRE_B4.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_FRE_B8.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_FRE_B5.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_SRE_B2.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_SRE_B3.tif",
    "SENTINEL2A_20200101_L2A_T31TCJ_QKL_ALL.jpg",
    "MASKS/SENTINEL2A_20200101_L2A_T31TCJ_CLM_R1.tif",
    "MASKS/SENTINEL2A_20200101_L2A_T31TCJ_CLM_R2.tif",
    "MASKS/SENTINEL2A_20200101_L2A_T31TCJ_EDG_R1.tif",
    "MASKS/SENTINEL2A_20200101_L2A_T31TCJ_SAT_R1.tif",
    "MASKS/SENTINEL2A_20200101_L2A_T31TCJ_MG2_R1.tif",
    "DATA/SENTINEL2A_20200101_L2A_T31TCJ_MTD_ALL.xml",
]


def _maja(**options) -> Sentinel2MajaDataset:
    cfg = _dataset_config(
        reader="sentinel2-maja",
        source="s3://bucket/T31TCJ/",
        options=options,
    )
    return build_dataset(cfg)


def test_maja_selects_reflectance_bands():
    ds = _maja(reflectance=["FRE"], bands=["B2", "B3", "B4", "B8"])
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    names = [c.name for c in components]
    assert names == ["FRE_B2", "FRE_B3", "FRE_B4", "FRE_B8"]
    # Members are addressed on the fly via /vsizip//vsis3, not pre-extracted.
    assert components[0].uri == (
        "/vsizip//vsis3/bucket/T31TCJ/scene.zip/"
        "SENTINEL2A_20200101_L2A_T31TCJ_FRE_B2.tif"
    )


def test_maja_fans_in_masks_and_both_reflectances():
    ds = _maja(
        reflectance=["FRE", "SRE"],
        bands=["B2", "B3"],
        masks=["CLM", "EDG", "SAT", "MG2"],
    )
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    names = {c.name for c in components}
    assert names == {
        "FRE_B2",
        "FRE_B3",
        "SRE_B2",
        "SRE_B3",
        "CLM_R1",
        "CLM_R2",
        "EDG_R1",
        "SAT_R1",
        "MG2_R1",
    }


def test_maja_empty_masks_excludes_masks():
    ds = _maja(reflectance=["FRE"], bands=["B2"], masks=[])
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    assert [c.name for c in components] == ["FRE_B2"]


def test_maja_ignores_non_raster_members():
    ds = _maja(reflectance=["FRE", "SRE"], bands=["B2", "B3", "B4", "B5", "B8"])
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    # The QKL jpg and the MTD xml are never picked.
    assert all(c.uri.endswith(".tif") for c in components)


def test_zip_delivery_enumerates_scenes_from_local_zips(tmp_path):
    import zipfile

    tile_root = tmp_path / "T31TCJ"
    tile_root.mkdir()
    for scene in ("2015_sceneA", "2016_sceneB"):
        zip_path = tile_root / f"{scene}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for member in MAJA_MEMBERS:
                zf.writestr(member, b"")

    cfg = _dataset_config(
        reader="sentinel2-maja",
        source=str(tile_root),
        options={"reflectance": ["FRE"], "bands": ["B2", "B3"], "masks": ["CLM"]},
    )
    ds = build_dataset(cfg)

    products = ds.products()
    assert [p.id for p in products] == ["2015_sceneA", "2016_sceneB"]
    names = [c.name for c in products[0].components]
    assert names == ["CLM_R1", "CLM_R2", "FRE_B2", "FRE_B3"]
    assert products[0].components[2].uri.startswith("/vsizip/")
    assert products[0].components[2].uri.endswith("_FRE_B2.tif")

    # Prefix + limit bound a product-set enumeration.
    bounded = ds.products(prefix="2016", limit=1)
    assert [p.id for p in bounded] == ["2016_sceneB"]
