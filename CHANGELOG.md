# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-01-08

### Breaking Changes

- **Minimum Python version increased to 3.11** (dropped Python 3.10 support)
- Deprecated `sqlp` module (use `sqlclz` if needed)

### Changed

- Migrated `Self` type imports from `typing_extensions` to `typing` (17 files updated)
- Updated `ParamSpec` and `Concatenate` imports to use standard `typing` module
- Updated classifiers to reflect Python 3.11+ support

### Infrastructure

- Updated ReadTheDocs configuration to Python 3.11
- Updated GitHub Actions workflows to Python 3.11

[Unreleased]: https://github.com/ytsimon2004/neuralib/compare/v0.6.0...HEAD

[0.6.0]: https://github.com/ytsimon2004/neuralib/compare/v0.5.6...v0.6.0