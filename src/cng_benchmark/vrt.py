"""Hand-built GDAL VRT mosaics for manual TiTiler inspection.

The basic ``titiler.application`` has no STAC API and no band-combination UI, so a
single-band reflectance COG is awkward to look at and a natural-colour image
can't be made at all. The cheap fix is a GDAL VRT that stacks the per-band COGs a
run produced into a 3-band RGB mosaic, whose ``s3://`` path drops straight into
TiTiler's viewer.

The constraint shaping this module: the runner and TiTiler images carry **only
rasterio's bundled GDAL** — there is no ``gdalbuildvrt`` CLI and no ``osgeo``
Python bindings — so the VRT is emitted as **hand-built XML**. rasterio (the
``cog`` extra) is used only to read each source's grid; the XML is a plain string
the caller uploads next to the run. TiTiler opens ``s3://…/run-*.vrt`` and
rasterio resolves it to ``/vsis3`` at the top level, so the sources inside the
VRT are absolute ``/vsis3/bucket/key`` paths (one CRS/resolution across the
mosaic is assumed — true within a single MGRS tile / UTM zone).
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass

#: rasterio numpy dtype name → VRT ``dataType``.
_VRT_DTYPES = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}

_COLOR_INTERP = ("Red", "Green", "Blue")
_EPSG_RE = _re.compile(r'ID\["EPSG",(\d+)\]|AUTHORITY\["EPSG","(\d+)"\]')


@dataclass(frozen=True)
class GridMeta:
    """The geo grid of one source raster, read from its header."""

    path: str  # GDAL-openable path, e.g. ``/vsis3/bucket/key``
    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]  # Affine a..f
    crs_wkt: str
    dtype: str
    nodata: float | None
    overviews: list[int]  # scale factors from the first band, e.g. [2, 4, 8, 16, 32]

    @property
    def px(self) -> float:
        return self.transform[0]

    @property
    def py(self) -> float:
        return -self.transform[4]

    @property
    def left(self) -> float:
        return self.transform[2]

    @property
    def top(self) -> float:
        return self.transform[5]

    @property
    def right(self) -> float:
        return self.left + self.px * self.width

    @property
    def bottom(self) -> float:
        return self.top - self.py * self.height


def read_grid(path: str) -> GridMeta:
    """Read ``path``'s grid header via rasterio (wrap in a ``gdal_session``).

    Only the header is touched (no pixels), so reading a remote COG over
    ``/vsis3`` is a couple of ranged GETs, not a download.
    """
    import rasterio

    with rasterio.open(path) as src:
        t = src.transform
        return GridMeta(
            path=path,
            width=src.width,
            height=src.height,
            transform=(t.a, t.b, t.c, t.d, t.e, t.f),
            crs_wkt=src.crs.to_wkt() if src.crs else "",
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            overviews=list(src.overviews(1)),
        )


def crs_epsg(crs_wkt: str) -> str | None:
    """Extract the outermost EPSG code from a CRS WKT string, or ``None``.

    WKT embeds AUTHORITY/ID tags at multiple nesting levels (spheroid, datum,
    CRS); the outermost one — last in the string — is the CRS's own EPSG code.
    """
    matches = _EPSG_RE.findall(crs_wkt)
    if not matches:
        return None
    # Each match is (group1, group2); take the non-empty group from the last hit.
    last = matches[-1]
    return last[0] or last[1] or None


def build_single_band_vrt_xml(grids: list[GridMeta]) -> str:
    """Build a 1-band Gray mosaic VRT XML from source grids.

    ``grids`` is a list of COG sources, all sharing a CRS and pixel size; the
    dataset extent is their union and each source is placed by its ``DstRect``.
    """
    if not grids:
        raise ValueError("no sources to mosaic")

    ref = grids[0]
    px, py = ref.px, ref.py
    minx = min(g.left for g in grids)
    maxx = max(g.right for g in grids)
    miny = min(g.bottom for g in grids)
    maxy = max(g.top for g in grids)
    x_size = max(1, round((maxx - minx) / px))
    y_size = max(1, round((maxy - miny) / py))

    dtype = _VRT_DTYPES.get(ref.dtype, "Float32")
    nodata = next((g.nodata for g in grids if g.nodata is not None), None)
    band_nodata = nodata if nodata is not None else 0

    lines = [
        f'<VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}">',
    ]
    if ref.crs_wkt:
        lines.append(f"  <SRS>{ref.crs_wkt}</SRS>")
    geo = (minx, px, 0.0, maxy, 0.0, -py)
    lines.append(
        "  <GeoTransform>" + ", ".join(f"{v:.10g}" for v in geo) + "</GeoTransform>"
    )
    if ref.overviews:
        ov = " ".join(str(f) for f in ref.overviews)
        lines.append(f"  <OverviewList>{ov}</OverviewList>")

    lines.append(f'  <VRTRasterBand dataType="{dtype}" band="1">')
    lines.append("    <ColorInterp>Gray</ColorInterp>")
    lines.append(f"    <NoDataValue>{band_nodata:.10g}</NoDataValue>")
    for g in grids:
        dx = round((g.left - minx) / px)
        dy = round((maxy - g.top) / py)
        lines.append("    <ComplexSource>")
        lines.append(
            f'      <SourceFilename relativeToVRT="0">{g.path}</SourceFilename>'
        )
        lines.append("      <SourceBand>1</SourceBand>")
        lines.append(
            f'      <SrcRect xOff="0" yOff="0" xSize="{g.width}" ySize="{g.height}"/>'
        )
        lines.append(
            f'      <DstRect xOff="{dx}" yOff="{dy}" '
            f'xSize="{g.width}" ySize="{g.height}"/>'
        )
        if g.nodata is not None:
            lines.append(f"      <NODATA>{g.nodata:.10g}</NODATA>")
        lines.append("    </ComplexSource>")
    lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    return "\n".join(lines) + "\n"


def build_rgb_vrt_xml(band_grids: list[list[GridMeta]]) -> str:
    """Build a 3-band mosaic VRT XML from per-band source grids.

    ``band_grids`` is exactly three lists (Red, Green, Blue); each inner list is
    the sources to mosaic into that band (one COG per product). Every source is
    assumed to share a CRS and pixel size; the dataset extent is their union, and
    each source is placed by a ``DstRect`` offset into that union grid.
    """
    if len(band_grids) != 3:
        raise ValueError(f"an RGB VRT needs 3 bands, got {len(band_grids)}")
    flat = [g for band in band_grids for g in band]
    if not flat:
        raise ValueError("no sources to mosaic")

    ref = flat[0]
    px, py = ref.px, ref.py
    minx = min(g.left for g in flat)
    maxx = max(g.right for g in flat)
    miny = min(g.bottom for g in flat)
    maxy = max(g.top for g in flat)
    x_size = max(1, round((maxx - minx) / px))
    y_size = max(1, round((maxy - miny) / py))

    dtype = _VRT_DTYPES.get(ref.dtype, "Float32")

    lines = [
        f'<VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}">',
    ]
    if ref.crs_wkt:
        lines.append(f"  <SRS>{ref.crs_wkt}</SRS>")
    geo = (minx, px, 0.0, maxy, 0.0, -py)
    lines.append(
        "  <GeoTransform>" + ", ".join(f"{v:.10g}" for v in geo) + "</GeoTransform>"
    )
    if ref.overviews:
        ov = " ".join(str(f) for f in ref.overviews)
        lines.append(f"  <OverviewList>{ov}</OverviewList>")

    for idx, sources in enumerate(band_grids):
        lines.append(f'  <VRTRasterBand dataType="{dtype}" band="{idx + 1}">')
        lines.append(f"    <ColorInterp>{_COLOR_INTERP[idx]}</ColorInterp>")
        nodata = next((g.nodata for g in sources if g.nodata is not None), None)
        # Always emit NoDataValue: GDAL uses it to mask uncovered mosaic gaps.
        # Default to 0 when no source declares one (gaps are filled with 0 anyway).
        band_nodata = nodata if nodata is not None else 0
        lines.append(f"    <NoDataValue>{band_nodata:.10g}</NoDataValue>")
        for g in sources:
            dx = round((g.left - minx) / px)
            dy = round((maxy - g.top) / py)
            lines.append("    <ComplexSource>")
            lines.append(
                f'      <SourceFilename relativeToVRT="0">{g.path}</SourceFilename>'
            )
            lines.append("      <SourceBand>1</SourceBand>")
            lines.append(
                f'      <SrcRect xOff="0" yOff="0" '
                f'xSize="{g.width}" ySize="{g.height}"/>'
            )
            lines.append(
                f'      <DstRect xOff="{dx}" yOff="{dy}" '
                f'xSize="{g.width}" ySize="{g.height}"/>'
            )
            if g.nodata is not None:
                lines.append(f"      <NODATA>{g.nodata:.10g}</NODATA>")
            lines.append("    </ComplexSource>")
        lines.append("  </VRTRasterBand>")

    lines.append("</VRTDataset>")
    return "\n".join(lines) + "\n"
