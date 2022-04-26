# HF-OSS Demo changelog

Uses [changelog conventions](https://keepachangelog.com/en/1.0.0/).
Uses [semantic versioning](https://semver.org/).

## Guiding Principles
- Changelogs are for humans, not machines.
- There should be an entry for every single version.
- The same types of changes should be grouped.
- Versions and sections should be linkable.
- The latest version comes first.
- The release date of each version is displayed.
- Mention whether you follow Semantic Versioning.

## Types of changes
- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

# [1.2.0]

- Added `benchmark` action to T5 TRT for performance benchmarking. It uses random inputs with fixed lengths and disables
  early stopping such that we can compare the performance with other frameworks.

# [1.1.0] - 2022-02-09

- Added `-o` or `--save-output-fpath` which saves a pickled version of the `NetworkResult` object. Useful for testing.

# [1.0.0] - 2022

- Added initial working example of HF samples and notebooks.
