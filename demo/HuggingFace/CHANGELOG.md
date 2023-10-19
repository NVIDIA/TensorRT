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

# [2.0.6] - 2023-10-10
- Added runtime dynamic batch support for `generate` function

# [2.0.5] - 2023-09-06
- Added `accuracy` command for LAMBADA topN accuracy for HF Demo

# [2.0.4] - 2023-08-15
- Added support for BLIP models

# [2.0.3] - 2023-06-29
- Added `attention_mask` support for multi-batch inference with various seq len

# [2.0.2] - 2023-06-23
- Added support for BLOOM models
- Added support for OPT models

# [2.0.1] - 2023-05-31
- Changed PyTorch==2.0.1 to officially support H100
- Added `chat` command to accept customized inputs
- Added `--engine-postfix` for differentiating engine name for different platforms
- Changed `NetworkCheckpointResult` and `NetworkResult` to avoid duplicate long output.

# [2.0.0] - 2023-05-09
- Added Seq2Seq class to extract common features from Seq2Seq models
- Changed NNDF.interface to include common workflow for all commands
- Added flags to take TRT engines without ONNX and PyTorch file.
- Added playground.ipynb for customized testing with model selection
- Added `generate` and `__init__` API support for all the model classes in parallel to CLI
- Changed all notebooks to use direct API
- Added support for more GPT2 and T5 models
- Removed old playground.ipynb per model
- Deprecated `--enable-kv-cache` to `--use-cache`
- Fixed bs > 1 kv cache accuracy issues
- Added per step time for `--info` mode

# [1.3.4] - 2023-02-02
- Changed GPT2 demo kv cache TRT to 1 engine, 2 optimization profiles
- Added fp16 support for GPT2

# [1.3.3] - 2023-01-04
- Deprecated max workspace size flag to memory pool limits for TensorRT
- Added t5-11b support
- Changed T5 demo kv cache TRT memory organization to avoid D2D copy

# [1.3.2] - 2022-11-17
- Added beam search support for GPT2 demo
- Added KV cache support for GPT2 demo
- Fixed perplexity calculation array size out of max_length
- Fixed trt KV cache engine profile to only accept input_length = 1
- Fixed external onnx weight file name overwrite issue

# [1.3.1] - 2022-11-04
- Added beam search support for T5 demo
- Added KV cache support for T5 demo

# [1.3.0] - 2022-11-03
- Added perplexity calculation for all samples
- Added precision override to checkpoints.
- Fixed TensorRT BART checkpoint not working.

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
