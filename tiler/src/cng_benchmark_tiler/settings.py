"""cng-benchmark-tiler API settings."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """FastAPI application settings, overridable via ``TILER_API_*`` env vars."""

    name: str = "cng-benchmark-tiler"
    description: str = (
        "Bench TiTiler application: /cog (rasterio/GDAL), /zarr (stock xarray "
        "defaults), /geozarr (titiler-eopf's GeoZarrReader)."
    )
    cors_origins: str = "*"
    cors_allow_methods: str = "GET"
    cachecontrol: str = "public, max-age=3600"

    model_config = SettingsConfigDict(env_prefix="TILER_API_", extra="ignore")

    @field_validator("cors_origins")
    def parse_cors_origins(cls, v: str) -> list[str]:
        """Split a comma-separated origins string into a list."""
        return [origin.strip() for origin in v.split(",")]

    @field_validator("cors_allow_methods")
    def parse_cors_allow_methods(cls, v: str) -> list[str]:
        """Split a comma-separated methods string into an upper-cased list."""
        return [method.strip().upper() for method in v.split(",")]
