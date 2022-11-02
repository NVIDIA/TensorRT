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


# [1.2.5] - 2022-10-08
- Added beam search support for BART
 
# [1.2.4] - 2022-09-30
- Added notebooks for BART demo
- Enabled flexible control on (a) percentile latency reports (b) engine building profile other than standard maximum input/output length config

# [1.2.3] - 2022-06-30
- Added KV cache support for BART demo
  
# [1.2.2] - 2022-06-14
- Added BART demo

# [1.2.1] - 2022-05-20

- Added `benchmark` action to T5 frameworks/onnxrt and GPT2 frameworks/trt for performance benchmarking. It uses random
  inputs with fixed lengths and disables early stopping such that we can compare the performance with other frameworks.
- Added `batch_size > 1` support to GPT2 trt sample.

# [1.2.0] - 2022-03-29

- Added `benchmark` action to T5 trt for performance benchmarking. It uses random inputs with fixed lengths and disables
  early stopping such that we can compare the performance with other frameworks.

# [1.1.0] - 2022-02-09

- Added `-o` or `--save-output-fpath` which saves a pickled version of the `NetworkResult` object. Useful for testing.

# [1.0.0] - 2022

- Added initial working example of HF samples and notebooks.
