# Changelog

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
