# Polygraphy Trtexec Extension Change Log

Dates are in YYYY-MM-DD format.
## vNext

## v0.0.8 (2022-09-08)
### Added
- `trtexec-export-times`, `trtexec-export-output`, `trtexec-export-profile` and `trtexec-export-layer-info` flags
- Report e2e Host Latency instead of GPU Compute time

### Changed
- Use latency as the inference metric instead of GPU Compute Time

## v0.0.7 (2022-08-10)
### Fixed
- Fixed README rendering in wheel build

## v0.0.6 (2020-07-21)
### Changed
- Bug fix in reporting inference time

## v0.0.5 (2020-07-21)
### Changed
- Bug fix in handling `refit` flag for `polygraphy<'0.39.0`

## v0.0.4 (2020-07-18)
### Added
- `trtexec-path` flag applies to allow specifying path to a custom `trtexec` binary

## v0.0.3 (2020-07-15)
### Changed
- `refittable` flag applies to `trtexec` backend's `refit` only if `polygraphy>=0.39.0`
- Renamed `no-builder-cache` to `trtexec-no-builder-cache`
- Renamed `profiling-verbosity` to `trtexec-profiling-verbosity`

## v0.0.2 (2020-07-05)
### Added
- `--trtexec-iterations` flag. Avoids unwanted behavior of running N times more iterations than expected when using the --iterations flag

### Changed
- Renamed `trtexec-warm-up` to `trtexec-warmup`

### Fixed
- Bug fix for failing pytests

## v0.0.1 (2022-06-23)
- Initial integration
