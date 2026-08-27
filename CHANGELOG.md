# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Historical entries were backfilled from git tags and release commit subjects.

## [Unreleased]

### Added

- Migrated project workflows and tooling toward `uv`.

### Changed

- Updated project dependencies.
- Applied repository-wide Ruff formatting updates.
- Updated README instructions for `uv` usage.

### Removed

- Removed `requirements.txt` in favor of `uv`-based dependency management.

### Fixed

- Fixed GitHub test workflow configuration for `uv`.
- Resolved Ruff-related CI/tooling issues.
- Fixed minor documentation and formatting issues.

## [1.1.5] - 2026-08-19

### Changed

- Released version `1.1.5`.

## [1.1.4] - 2026-08-11

### Changed

- Unpinned older dependencies.

## [1.1.3] - 2026-04-18

### Changed

- Enabled trusted publishing for PyPI in CI.

## [1.1.2] - 2025-04-21

### Changed

- Released version `1.1.2`.

## [1.1.1] - 2025-04-21

### Changed

- Released version `1.1.1`.

## [1.1.0] - 2024-09-27

### Changed

- Released version `1.1.0`.

## [1.0.0] - 2024-07-25

### Changed

- Released version `1.0.0`.

## [0.9.2] - 2024-04-19

### Added

- Added the pyOpenSci peer-review badge to the README.

## [0.9.1] - 2024-04-19

### Changed

- Released version `0.9.1`.

## [0.9.0] - 2024-04-13

### Changed

- Dropped Python 3.8 support.

## [0.8.6] - 2024-03-21

### Added

- Added project badges.

## [0.8.5] - 2023-11-06

### Changed

- Updated `mkdocsstrings` and `mkdocstrings-python` to restore documentation generation.

## [0.8.4] - 2023-02-06

### Changed

- Merged interpolation updates.

## [0.8.3] - 2023-01-11

### Fixed

- Fixed unit-conversion behavior for bandpasses.

## [0.8.2] - 2023-01-11

### Changed

- Released version `0.8.2`.

## [0.8.1] - 2023-01-10

### Removed

- Excluded test file from package build artifacts.

## [0.8.0] - 2023-01-09

### Changed

- Merged development branch updates into main.

## [0.7.5] - 2022-10-24

### Fixed

- Fixed bandpass-related behavior and expanded tests.

## [0.7.3] - 2022-10-14

### Changed

- Released version `0.7.3`.

## [0.7.2] - 2022-10-13

### Changed

- Reverted default `parallel=False` behavior and updated docs.

## [0.7.1] - 2022-10-11

### Removed

- Removed suppression of an OpenMP warning no longer needed.

## [0.7.0] - 2022-10-06

### Changed

- Disabled Numba JIT compilation and suppressed overflow warnings in tests.

## [0.6.6] - 2022-07-05

### Changed

- Released version `0.6.6`.

## [0.6.5] - 2022-07-04

### Changed

- Released version `0.6.5`.

## [0.6.4] - 2022-07-04

### Changed

- Released version `0.6.4`.

## [0.6.3] - 2022-06-24

### Fixed

- Fixed compatibility issue affecting Python versions newer than 3.10.

## [0.6.2] - 2022-06-22

### Added

- Added `solar_cutoff` support in `Zodipy` initialization; pointings within the cutoff now return `np.nan`.

## [0.6.1] - 2022-06-20

### Removed

- Removed Read the Docs configuration after moving from Sphinx to MkDocs.

## [0.6.0] - 2022-04-26

### Changed

- Updated README to match the new API.

## [0.5.6] - 2022-04-19

### Fixed

- Attempted fixes for GitHub Actions behavior.

## [0.5.0] - 2022-03-20

### Changed

- Released version `0.5.0`.

## [0.4.0] - 2022-02-21

### Changed

- Released version `0.4.0`.

## [0.3.0] - 2021-11-24

### Changed

- Updated README.

## [0.2.2] - 2021-09-14

### Changed

- Released patch version `0.2.2`.

## [0.2.1] - 2021-09-06

### Changed

- Released version `0.2.1`.

## [0.2.0] - 2021-09-04

### Changed

- Released version `0.2.0`.

## [0.1.9] - 2021-09-03

### Added

- Implemented manual step-wise trapezoidal integration.

## [0.1.8] - 2021-08-23

### Changed

- Released version `0.1.8`.

## [0.1.7] - 2021-08-22

### Added

- Added a factory-pattern method for integration configurations.

## [0.1.6] - 2021-08-20

### Changed

- Updated README.

## [0.1.5] - 2021-08-18

### Changed

- Released version `0.1.5`.

## [0.1.4] - 2021-08-16

### Removed

- Removed redundant models.

## [0.1.3] - 2021-08-15

### Changed

- Improved import structure.

## [0.1.2] - 2021-08-14

### Fixed

- Fixed default-time behavior.

[Unreleased]: https://github.com/Cosmoglobe/zodipy/compare/v1.1.5...HEAD
[1.1.5]: https://github.com/Cosmoglobe/zodipy/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/Cosmoglobe/zodipy/compare/v.1.1.3...v1.1.4
[1.1.3]: https://github.com/Cosmoglobe/zodipy/compare/v.1.1.2...v.1.1.3
[1.1.2]: https://github.com/Cosmoglobe/zodipy/compare/v.1.1.1...v.1.1.2
[1.1.1]: https://github.com/Cosmoglobe/zodipy/compare/v.1.1.0...v.1.1.1
[1.1.0]: https://github.com/Cosmoglobe/zodipy/compare/v.1.0.0...v.1.1.0
[1.0.0]: https://github.com/Cosmoglobe/zodipy/compare/v.0.9.2...v.1.0.0
[0.9.2]: https://github.com/Cosmoglobe/zodipy/compare/v.0.9.1...v.0.9.2
[0.9.1]: https://github.com/Cosmoglobe/zodipy/compare/v.0.9.0...v.0.9.1
[0.9.0]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.6...v.0.9.0
[0.8.6]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.5...v.0.8.6
[0.8.5]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.4...v.0.8.5
[0.8.4]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.3...v.0.8.4
[0.8.3]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.2...v.0.8.3
[0.8.2]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.1...v.0.8.2
[0.8.1]: https://github.com/Cosmoglobe/zodipy/compare/v.0.8.0...v.0.8.1
[0.8.0]: https://github.com/Cosmoglobe/zodipy/compare/v.0.7.5...v.0.8.0
[0.7.5]: https://github.com/Cosmoglobe/zodipy/compare/v.0.7.3...v.0.7.5
[0.7.3]: https://github.com/Cosmoglobe/zodipy/compare/v.0.7.2...v.0.7.3
[0.7.2]: https://github.com/Cosmoglobe/zodipy/compare/v.0.7.1...v.0.7.2
[0.7.1]: https://github.com/Cosmoglobe/zodipy/compare/v.0.7.0...v.0.7.1
[0.7.0]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.6...v.0.7.0
[0.6.6]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.5...v.0.6.6
[0.6.5]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.4...v.0.6.5
[0.6.4]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.3...v.0.6.4
[0.6.3]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.2...v.0.6.3
[0.6.2]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.1...v.0.6.2
[0.6.1]: https://github.com/Cosmoglobe/zodipy/compare/v.0.6.0...v.0.6.1
[0.6.0]: https://github.com/Cosmoglobe/zodipy/compare/v0.5.6...v.0.6.0
[0.5.6]: https://github.com/Cosmoglobe/zodipy/compare/v0.5.0...v0.5.6
[0.5.0]: https://github.com/Cosmoglobe/zodipy/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Cosmoglobe/zodipy/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Cosmoglobe/zodipy/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Cosmoglobe/zodipy/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Cosmoglobe/zodipy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Cosmoglobe/zodipy/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Cosmoglobe/zodipy/releases/tag/v0.1.2
