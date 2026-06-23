"""Tests for the dataset hierarchy, registry, and the MAJA reader."""

import pytest
from pydantic import ValidationError

from cng_benchmark.config import DatasetConfig
from cng_benchmark.datasets import Product, SourceObject, build_dataset
from cng_benchmark.datasets.sentinel2 import Sentinel2MajaDataset
from cng_benchmark.datasets.single_object import SingleObjectDataset
from cng_benchmark.datasets.swot import DEFAULT_VARIABLES, SwotRaster100mDataset
from cng_benchmark.datasets.swot_pixc import DEFAULT_GROUPS, SwotPixcDataset
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
    for name in (
        "single-object",
        "sentinel2-maja",
        "sentinel1-otb-rtc",
        "swot-raster100m",
        "swot-lakesp-prior",
        "swot-pixc",
    ):
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


def test_maja_orders_reflectance_then_10m_first():
    # With a non-10 m band (B5) and masks fanned in, the first component must be
    # a 10 m reflectance band so the default read/display sample is representative
    # (#13): 10 m FRE bands, then the 20 m FRE band, then masks.
    ds = _maja(
        reflectance=["FRE"],
        bands=["B2", "B4", "B5"],
        masks=["CLM", "EDG"],
    )
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    assert [c.name for c in components] == [
        "FRE_B2",
        "FRE_B4",
        "FRE_B5",
        "CLM_R1",
        "CLM_R2",
        "EDG_R1",
    ]


def test_maja_empty_masks_excludes_masks():
    ds = _maja(reflectance=["FRE"], bands=["B2"], masks=[])
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    assert [c.name for c in components] == ["FRE_B2"]


def test_maja_ignores_non_raster_members():
    ds = _maja(reflectance=["FRE", "SRE"], bands=["B2", "B3", "B4", "B5", "B8"])
    components = ds._select_members(MAJA_MEMBERS, "s3://bucket/T31TCJ/scene.zip")
    # The QKL jpg and the MTD xml are never picked.
    assert all(c.uri.endswith(".tif") for c in components)


# A captured listing of an S1 RTC (S1Tiling gamma0) zip: two band rasters per
# polarisation plus a quicklook jpg and a GDAL .aux.xml sidecar.
S1_MEMBERS = [
    "S1A_L1ORT_31TCH_VH_GAM_ASC_030_20200101T060000.tif",
    "S1A_L1ORT_31TCH_VH_GAM_ASC_030_20200101T060000_QKL_ALL.jpg",
    "S1A_L1ORT_31TCH_VV_GAM_ASC_030_20200101T060000.tif",
    "S1A_L1ORT_31TCH_VV_GAM_ASC_030_20200101T060000.tif.aux.xml",
]


def _s1(**options):
    cfg = _dataset_config(
        reader="sentinel1-otb-rtc",
        source="s3://bucket/T31TCH/",
        options=options,
    )
    return build_dataset(cfg)


def test_s1_selects_both_polarizations_in_configured_order():
    ds = _s1(polarizations=["VV", "VH"])
    components = ds._select_members(S1_MEMBERS, "s3://bucket/T31TCH/scene.zip")
    assert [c.name for c in components] == ["VV", "VH"]
    # Members are addressed on the fly via /vsizip//vsis3, not pre-extracted.
    assert components[0].uri == (
        "/vsizip//vsis3/bucket/T31TCH/scene.zip/"
        "S1A_L1ORT_31TCH_VV_GAM_ASC_030_20200101T060000.tif"
    )


def test_s1_order_follows_polarizations_option():
    ds = _s1(polarizations=["VH", "VV"])
    components = ds._select_members(S1_MEMBERS, "s3://bucket/T31TCH/scene.zip")
    assert [c.name for c in components] == ["VH", "VV"]


def test_s1_single_polarization():
    ds = _s1(polarizations=["VV"])
    components = ds._select_members(S1_MEMBERS, "s3://bucket/T31TCH/scene.zip")
    assert [c.name for c in components] == ["VV"]


def test_s1_ignores_quicklook_and_sidecar():
    ds = _s1()  # default both polarisations
    components = ds._select_members(S1_MEMBERS, "s3://bucket/T31TCH/scene.zip")
    # Only the two .tif bands; the _QKL_ALL.jpg and .aux.xml are never picked.
    assert {c.name for c in components} == {"VV", "VH"}
    assert all(c.uri.endswith(".tif") for c in components)


def test_s1_enumerates_scenes_from_local_zips(tmp_path):
    import zipfile

    tile_root = tmp_path / "T31TCH"
    tile_root.mkdir()
    for scene in ("2020_sceneA", "2021_sceneB"):
        with zipfile.ZipFile(tile_root / f"{scene}.zip", "w") as zf:
            for member in S1_MEMBERS:
                zf.writestr(member, b"")

    cfg = _dataset_config(
        reader="sentinel1-otb-rtc",
        source=str(tile_root),
        options={"polarizations": ["VV", "VH"]},
    )
    products = build_dataset(cfg).products()
    assert [p.id for p in products] == ["2020_sceneA", "2021_sceneB"]
    assert [c.name for c in products[0].components] == ["VV", "VH"]
    assert products[0].components[0].uri.startswith("/vsizip/")


# --- SWOT Raster100m (netCDF-raster granule) -------------------------------


def _swot(**options) -> SwotRaster100mDataset:
    cfg = _dataset_config(
        reader="swot-raster100m",
        source="s3://bucket/Raster100m_Nom_France/",
        options=options,
    )
    return build_dataset(cfg)


def test_swot_defaults_to_primary_variable():
    ds = _swot()
    granule = "s3://bucket/Raster100m_Nom_France/SWOT_L2_HR_Raster_100m_UTM31N.nc"
    components = ds._select_components(granule)
    assert [c.name for c in components] == DEFAULT_VARIABLES
    # The variable is read in place via a GDAL CF subdataset path, not extracted.
    assert components[0].uri == (
        'NETCDF:"/vsis3/bucket/Raster100m_Nom_France/'
        'SWOT_L2_HR_Raster_100m_UTM31N.nc":wse'
    )


def test_swot_selects_variables_in_configured_order():
    ds = _swot(variables=["water_area", "wse", "sig0"])
    granule = "s3://bucket/Raster100m_Nom_France/SWOT_L2_HR_Raster_100m_UTM31N.nc"
    components = ds._select_components(granule)
    assert [c.name for c in components] == ["water_area", "wse", "sig0"]


def test_swot_rejects_unknown_option():
    with pytest.raises(ValidationError):
        build_dataset(_dataset_config(reader="swot-raster100m", options={"bogus": 1}))


def test_swot_enumerates_granules_from_local_files(tmp_path):
    root = tmp_path / "Raster100m_Nom_France"
    root.mkdir()
    for name in ("SWOT_cycle048_UTM31N", "SWOT_cycle048_UTM32N"):
        (root / f"{name}.nc").write_bytes(b"")
    # A non-granule sibling is ignored (suffix match only).
    (root / "manifest.json").write_bytes(b"{}")

    cfg = _dataset_config(
        reader="swot-raster100m",
        source=str(root),
        options={"variables": ["wse"]},
    )
    ds = build_dataset(cfg)

    products = ds.products()
    assert [p.id for p in products] == ["SWOT_cycle048_UTM31N", "SWOT_cycle048_UTM32N"]
    assert [c.name for c in products[0].components] == ["wse"]
    uri = products[0].components[0].uri
    assert uri.startswith('NETCDF:"') and uri.endswith('.nc":wse')

    # Prefix + limit bound a granule-set enumeration (path-prefix match).
    bounded = ds.products(prefix="SWOT_cycle048_UTM32N", limit=1)
    assert [p.id for p in bounded] == ["SWOT_cycle048_UTM32N"]


# --- SWOT LakeSP Prior (zipped-shapefile vector granule) -------------------


# A captured listing of a SWOT LakeSP Prior zip: one shapefile (its four members)
# plus an XML metadata sidecar.
LAKESP_MEMBERS = [
    "SWOT_L2_HR_LakeSP_Prior_048_EU_001.shp",
    "SWOT_L2_HR_LakeSP_Prior_048_EU_001.shx",
    "SWOT_L2_HR_LakeSP_Prior_048_EU_001.dbf",
    "SWOT_L2_HR_LakeSP_Prior_048_EU_001.prj",
    "SWOT_L2_HR_LakeSP_Prior_048_EU_001.shp.xml",
]


def _lakesp(**options):
    cfg = _dataset_config(
        reader="swot-lakesp-prior",
        source="s3://bucket/LakeSP_Prior_Nom_France/",
        options=options,
    )
    return build_dataset(cfg)


def test_lakesp_selects_the_shapefile_layer():
    ds = _lakesp()
    components = ds._select_members(
        LAKESP_MEMBERS, "s3://bucket/LakeSP_Prior_Nom_France/pass048.zip"
    )
    # One component per pass: the .shp member, named for the layer; the sidecars
    # and the .shp.xml metadata are never picked.
    assert [c.name for c in components] == ["SWOT_L2_HR_LakeSP_Prior_048_EU_001"]
    # The shapefile is read on the fly via /vsizip//vsis3 — the OGR driver finds
    # the .shx/.dbf/.prj sidecars inside the same archive.
    assert components[0].uri == (
        "/vsizip//vsis3/bucket/LakeSP_Prior_Nom_France/pass048.zip/"
        "SWOT_L2_HR_LakeSP_Prior_048_EU_001.shp"
    )


def test_lakesp_rejects_unknown_option():
    with pytest.raises(ValidationError):
        build_dataset(_dataset_config(reader="swot-lakesp-prior", options={"bogus": 1}))


def test_lakesp_enumerates_passes_from_local_zips(tmp_path):
    import zipfile

    root = tmp_path / "LakeSP_Prior_Nom_France"
    root.mkdir()
    for pass_id in ("SWOT_pass048", "SWOT_pass049"):
        with zipfile.ZipFile(root / f"{pass_id}.zip", "w") as zf:
            for member in LAKESP_MEMBERS:
                zf.writestr(member.replace("048", pass_id[-3:]), b"")

    cfg = _dataset_config(
        reader="swot-lakesp-prior",
        source=str(root),
    )
    products = build_dataset(cfg).products()
    assert [p.id for p in products] == ["SWOT_pass048", "SWOT_pass049"]
    assert len(products[0].components) == 1
    assert products[0].components[0].uri.startswith("/vsizip/")
    assert products[0].components[0].uri.endswith(".shp")


# --- SWOT PIXC (netCDF point-cloud granule) --------------------------------


def _pixc(**options) -> SwotPixcDataset:
    cfg = _dataset_config(
        reader="swot-pixc",
        source="s3://bucket/PIXC_Nom_France/",
        options=options,
    )
    return build_dataset(cfg)


def test_pixc_defaults_to_pixel_cloud_group():
    ds = _pixc()
    granule = "s3://bucket/PIXC_Nom_France/SWOT_L2_HR_PIXC_048.nc"
    components = ds._select_components(granule)
    assert [c.name for c in components] == DEFAULT_GROUPS
    # The group is read in place via the PIXC point-source scheme, not extracted;
    # the original granule URI is kept intact for the xarray/fsspec loader.
    assert components[0].uri == (
        "PIXC:s3://bucket/PIXC_Nom_France/SWOT_L2_HR_PIXC_048.nc::pixel_cloud"
    )


def test_pixc_selects_groups_in_configured_order():
    ds = _pixc(groups=["pixel_cloud", "tvp"])
    granule = "s3://bucket/PIXC_Nom_France/SWOT_L2_HR_PIXC_048.nc"
    components = ds._select_components(granule)
    assert [c.name for c in components] == ["pixel_cloud", "tvp"]


def test_pixc_rejects_unknown_option():
    with pytest.raises(ValidationError):
        build_dataset(_dataset_config(reader="swot-pixc", options={"bogus": 1}))


def test_pixc_enumerates_granules_from_local_files(tmp_path):
    root = tmp_path / "PIXC_Nom_France"
    root.mkdir()
    for name in ("SWOT_L2_HR_PIXC_048_A", "SWOT_L2_HR_PIXC_048_B"):
        (root / f"{name}.nc").write_bytes(b"")
    (root / "manifest.json").write_bytes(b"{}")  # non-granule sibling ignored

    cfg = _dataset_config(reader="swot-pixc", source=str(root))
    products = build_dataset(cfg).products()
    assert [p.id for p in products] == [
        "SWOT_L2_HR_PIXC_048_A",
        "SWOT_L2_HR_PIXC_048_B",
    ]
    assert [c.name for c in products[0].components] == ["pixel_cloud"]
    uri = products[0].components[0].uri
    assert uri.startswith("PIXC:") and uri.endswith(".nc::pixel_cloud")


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
    # Reflectance-first so the default read/display sample lands on a 10 m band,
    # not a tiny mask (#13); masks follow.
    names = [c.name for c in products[0].components]
    assert names == ["FRE_B2", "FRE_B3", "CLM_R1", "CLM_R2"]
    assert products[0].components[0].uri.startswith("/vsizip/")
    assert products[0].components[0].uri.endswith("_FRE_B2.tif")

    # Prefix + limit bound a product-set enumeration (path-prefix match).
    bounded = ds.products(prefix="2016", limit=1)
    assert [p.id for p in bounded] == ["2016_sceneB"]
