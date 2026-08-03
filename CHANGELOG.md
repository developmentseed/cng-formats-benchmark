# Changelog

## [0.6.0](https://github.com/developmentseed/cng-formats-benchmark/compare/v0.5.0...v0.6.0) (2026-06-30)


### Features

* **deploy:** drop titiler-xarray in favour of TiTiler 2.0+ /zarr endpoint ([#62](https://github.com/developmentseed/cng-formats-benchmark/issues/62)) ([ad8159e](https://github.com/developmentseed/cng-formats-benchmark/commit/ad8159e884fb23f08257a9419407f3a1656d96e1))
* **deploy:** tune TiTiler GDAL caching and uvicorn workers for display metric ([#61](https://github.com/developmentseed/cng-formats-benchmark/issues/61)) ([bd12d78](https://github.com/developmentseed/cng-formats-benchmark/commit/bd12d78b34ee2310d06a57ec4cb0cbfe5d91549a))
* **vrt:** single-band viewer VRTs for scalar datasets ([#68](https://github.com/developmentseed/cng-formats-benchmark/issues/68)) ([cb46390](https://github.com/developmentseed/cng-formats-benchmark/commit/cb46390cb72be45b47dbde61e510d6970c6e9b6d))


### Bug Fixes

* **deploy:** scope the titiler Service/Deployment selector by release ([#65](https://github.com/developmentseed/cng-formats-benchmark/issues/65)) ([e7b4b6d](https://github.com/developmentseed/cng-formats-benchmark/commit/e7b4b6d701511d31f220e0cd851776106586637b))
* **runner:** make per-product read/display metrics best-effort ([#64](https://github.com/developmentseed/cng-formats-benchmark/issues/64)) ([735b477](https://github.com/developmentseed/cng-formats-benchmark/commit/735b47724350d8dfe8b96c9690726f9e619152a4))
* **storage:** avoid broad S3 scan for multi-tile alternation patterns ([#59](https://github.com/developmentseed/cng-formats-benchmark/issues/59)) ([6efb57c](https://github.com/developmentseed/cng-formats-benchmark/commit/6efb57c7dfd765371937fd854cdcbfcacb41b829))

## [0.5.0](https://github.com/developmentseed/cng-formats-benchmark/compare/v0.4.0...v0.5.0) (2026-06-29)


### Features

* **datasets:** regex product filter to select a multi-tile AOI in one run ([#50](https://github.com/developmentseed/cng-formats-benchmark/issues/50)) ([0b18795](https://github.com/developmentseed/cng-formats-benchmark/commit/0b18795a3205c3c405a87807a1d30a1d10d55e2e))
* **models:** uniform codec and compression_ratio across all ObjectLayout kinds ([#57](https://github.com/developmentseed/cng-formats-benchmark/issues/57)) ([603ccd3](https://github.com/developmentseed/cng-formats-benchmark/commit/603ccd37e922f890d8f73d678f54b59e986ff46a))
* RGB composite VRTs for manual TiTiler inspection ([#45](https://github.com/developmentseed/cng-formats-benchmark/issues/45)) ([1812af9](https://github.com/developmentseed/cng-formats-benchmark/commit/1812af91079412b28475879cc22e88dec7459339))
* **runner:** timestamped progress logging to stderr ([#51](https://github.com/developmentseed/cng-formats-benchmark/issues/51)) ([5029453](https://github.com/developmentseed/cng-formats-benchmark/commit/5029453a9684f5a4a6b26293c318ac6cfe652783))


### Bug Fixes

* **metrics:** bytes_in for zip-delivered sources (closes [#47](https://github.com/developmentseed/cng-formats-benchmark/issues/47)) ([#48](https://github.com/developmentseed/cng-formats-benchmark/issues/48)) ([084b1f2](https://github.com/developmentseed/cng-formats-benchmark/commit/084b1f2f44a902a09cea77592fef3e52a37f5b06))
* **vrt:** emit OverviewList and always set NoDataValue; fix COG nodata param ([#53](https://github.com/developmentseed/cng-formats-benchmark/issues/53)) ([3def77b](https://github.com/developmentseed/cng-formats-benchmark/commit/3def77b1ebde0f9a5c03a69edd95412af9afbf47))

## [0.4.0](https://github.com/developmentseed/cng-formats-benchmark/compare/v0.3.0...v0.4.0) (2026-06-25)


### Features

* **copc:** carry the full PIXC pixel_cloud variable set (content-complete COPC) ([#38](https://github.com/developmentseed/cng-formats-benchmark/issues/38)) ([0c3b11d](https://github.com/developmentseed/cng-formats-benchmark/commit/0c3b11db0c81d759793590cd0c7555e5bb689b69))
* **copc:** publish the octree level-of-detail figure as a systematic run output ([#44](https://github.com/developmentseed/cng-formats-benchmark/issues/44)) ([335fa85](https://github.com/developmentseed/cng-formats-benchmark/commit/335fa85a86c5c512a169ae0fe6a1db404b79adec))
* **metrics:** source size (bytes_in) for the PIXC arm + COPC compression ratio ([#43](https://github.com/developmentseed/cng-formats-benchmark/issues/43)) ([7e6b2b9](https://github.com/developmentseed/cng-formats-benchmark/commit/7e6b2b97bcb39684f4ad1532ad1c07843fe99c02))
* swot-lakesp-prior vector reader + GeoParquet adapter (LakeSP -&gt; GeoParquet) ([#31](https://github.com/developmentseed/cng-formats-benchmark/issues/31)) ([3d2dc32](https://github.com/developmentseed/cng-formats-benchmark/commit/3d2dc327bb20b458a010efe7d15f5acfedfd85d7))
* swot-pixc point-cloud reader + COPC adapter (PIXC -&gt; COPC) ([#34](https://github.com/developmentseed/cng-formats-benchmark/issues/34)) ([5b179b1](https://github.com/developmentseed/cng-formats-benchmark/commit/5b179b1c1d6ee20baa8e853be2e5e0a5d35d919a))
* swot-raster100m netCDF-raster reader (GranuleDataset -&gt; GeoZarr) ([#29](https://github.com/developmentseed/cng-formats-benchmark/issues/29)) ([b2af313](https://github.com/developmentseed/cng-formats-benchmark/commit/b2af313c7475b96439ab79035acf566c259322d1))


### Bug Fixes

* **copc:** bound octree node size by the per-node budget (no giant leaf) ([#42](https://github.com/developmentseed/cng-formats-benchmark/issues/42)) ([da78818](https://github.com/developmentseed/cng-formats-benchmark/commit/da7881858e3fb5d826f724d680f500af5aa51c49))
* **copc:** download the PIXC granule with boto3, not s3fs (read-timeout) ([#40](https://github.com/developmentseed/cng-formats-benchmark/issues/40)) ([96ed02d](https://github.com/developmentseed/cng-formats-benchmark/commit/96ed02de1580804034d369d5889f78f235f82b71))
* **copc:** read the PIXC source granule in one GET, not HDF5-over-s3fs random access ([#39](https://github.com/developmentseed/cng-formats-benchmark/issues/39)) ([6712fe8](https://github.com/developmentseed/cng-formats-benchmark/commit/6712fe88bd26b0f03e50e59230e149bbc8b2d1b7))
* default display_titiler_path to "" for GeoZarr display ([#30](https://github.com/developmentseed/cng-formats-benchmark/issues/30)) ([1354c72](https://github.com/developmentseed/cng-formats-benchmark/commit/1354c72e88c460692bf999837ee36fd46068fcdb))
* gdal_session overlays os.environ for non-rasterio GDAL bindings (pyogrio) ([#32](https://github.com/developmentseed/cng-formats-benchmark/issues/32)) ([64a232d](https://github.com/developmentseed/cng-formats-benchmark/commit/64a232dc00f26f7dc1941e7401fa70698411d07f))
* route the runner to titiler-xarray for GeoZarr display ([#23](https://github.com/developmentseed/cng-formats-benchmark/issues/23)) ([3293dfd](https://github.com/developmentseed/cng-formats-benchmark/commit/3293dfddea66cbe393f1c7446b5efcabf5d17ff9))


### Performance Improvements

* **copc:** cut content-complete build peak memory (free source arrays) ([#41](https://github.com/developmentseed/cng-formats-benchmark/issues/41)) ([2729764](https://github.com/developmentseed/cng-formats-benchmark/commit/27297641689bd831dba2a46034846e0325e4e3e9))

## [0.3.0](https://github.com/developmentseed/cng-formats-benchmark/compare/v0.2.0...v0.3.0) (2026-06-23)


### Features

* Add sentinel1 RTC reader (S1Tiling/OTB gamma0 VV/VH bands) ([ffc4e88](https://github.com/developmentseed/cng-formats-benchmark/commit/ffc4e88250d47d7236c1473420ddf5e8bb211297))
* GeoZarr v3 (2D, per-component) adapter + per-format layout describer ([ed80143](https://github.com/developmentseed/cng-formats-benchmark/commit/ed80143a80a93b0f85baa1b8932ee3bda1a51f84))
* M2.5: GeoZarr v3 (2D, per-component) adapter + per-format layout describer ([0d4eed5](https://github.com/developmentseed/cng-formats-benchmark/commit/0d4eed52770bc297c869a68d38bbf0510ca18540))
* sentinel1 RTC reader (S1Tiling/OTB gamma0 VV/VH bands) ([015bdca](https://github.com/developmentseed/cng-formats-benchmark/commit/015bdca5bd6e0fe099afd99197d93b020d2fd2e2))


### Bug Fixes

* address Copilot review on the GeoZarr PR ([5097282](https://github.com/developmentseed/cng-formats-benchmark/commit/5097282fbea851d1109c913c5ddd6c96ae777c63))
* address second Copilot review pass on the GeoZarr PR ([117d56d](https://github.com/developmentseed/cng-formats-benchmark/commit/117d56d3a76e348f166671c728e19011e50b937a))
* address third Copilot review pass on the GeoZarr PR ([3096f52](https://github.com/developmentseed/cng-formats-benchmark/commit/3096f5239e40cacb53358e90ba169f8f1c8a4c55))

## [0.2.0](https://github.com/developmentseed/cng-formats-benchmark/compare/v0.1.0...v0.2.0) (2026-06-18)


### Features

* capture each object's tiling layout in the result ([bea4d07](https://github.com/developmentseed/cng-formats-benchmark/commit/bea4d07f1597b0b18facead31a876fd0226a2592))
* capture each object's tiling layout in the result ([a662584](https://github.com/developmentseed/cng-formats-benchmark/commit/a662584be10edc88bd8ef8089d51b6ed54c6321c))
* layout-aware datasets — multi-component products + roll-up ([39bb6c3](https://github.com/developmentseed/cng-formats-benchmark/commit/39bb6c34d87ce0bb0438012f3c727400a48cde9f))


### Bug Fixes

* _S3SeekableReader.close() so zip-delivery datasets work over S3 ([f717169](https://github.com/developmentseed/cng-formats-benchmark/commit/f71716988dcaa8b9916dca0ddf683817f02c48b2))
* _S3SeekableReader.close() so zip-delivery datasets work over S3 ([7efb862](https://github.com/developmentseed/cng-formats-benchmark/commit/7efb862e78bb55e815ff13e60614221b2dbf4739))
* bound dataset enumeration server-side and sample a representative band ([6edb0eb](https://github.com/developmentseed/cng-formats-benchmark/commit/6edb0ebcaaf848cb5e3f4e166369b01432039915))
* bound dataset enumeration server-side and sample a representative band ([eb53001](https://github.com/developmentseed/cng-formats-benchmark/commit/eb5300159fd908bf45bbc3e45204a0922454a638))

## 0.1.0 (2026-06-18)


### Features

* chunk-aware display benchmark with tile/chunk layout image ([1c593c0](https://github.com/developmentseed/cng-formats-benchmark/commit/1c593c04b654f7d5f10e02e0a478da59d499bdd9))
* chunk-aware display benchmark with tile/chunk layout image ([bb92050](https://github.com/developmentseed/cng-formats-benchmark/commit/bb92050d43568097cc3a9191102e37196c78458b))
* support nodeSelector; stop blocking archive sources over /vsis3 ([4d5f9f7](https://github.com/developmentseed/cng-formats-benchmark/commit/4d5f9f794cf546012237f5f04eb322f1e1f423c9))


### Bug Fixes

* **helm:** gate the runner Job on TiTiler readiness ([2db57ed](https://github.com/developmentseed/cng-formats-benchmark/commit/2db57edfdd7dfdc2ddaa5346dfa8cf22d81b137a))


### Documentation

* MkDocs site (architecture + getting started + reference) and consolidated README ([a065a78](https://github.com/developmentseed/cng-formats-benchmark/commit/a065a78011ebd5b6b9e5a384c7fb66ca84ed7253))
* MkDocs site (architecture + getting started + reference) and consolidated README ([e7146aa](https://github.com/developmentseed/cng-formats-benchmark/commit/e7146aac22e0c1494f48a086d6990929c364a245))
