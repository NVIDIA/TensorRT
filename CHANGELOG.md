# TensorRT OSS Release Changelog

## [20.12](https://github.com/NVIDIA/TensorRT/releases/tag/20.12) - 2020-12-18
### Added
- Add configurable input size for TLT MaskRCNN Plugin

### Changed
- Update symbol export map for plugins
- Correctly use channel dimension when creating Prelu node
- Fix Jetson cross compilation CMakefile

### Removed
- N/A


## [20.11](https://github.com/NVIDIA/TensorRT/releases/tag/20.11) - 2020-11-20
### Added
- API documentation for [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/docs)

### Changed
- Support for SM86 in [demoBERT](https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT)
- Updated NGC checkpoint URLs for [demoBERT](https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT) and [Tacotron2](https://github.com/NVIDIA/TensorRT/tree/master/demo/Tacotron2).

### Removed
- N/A


## [20.10](https://github.com/NVIDIA/TensorRT/releases/tag/20.10) - 2020-10-22
### Added
- [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy) v0.20.13 - Deep Learning Inference Prototyping and Debugging Toolkit
- [PyTorch-Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization) v2.0.0
- Updated BERT plugins for [variable sequence length inputs](https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT#variable-sequence-length)
- Optimized kernels for sequence lengths of 64 and 96 added
- Added Tacotron2 + Waveglow TTS demo [#677](https://github.com/NVIDIA/TensorRT/pull/677)
- Re-enable `GridAnchorRect_TRT` plugin with rectangular feature maps [#679](https://github.com/NVIDIA/TensorRT/pull/679)
- Update batchedNMS plugin to IPluginV2DynamicExt interface [#738](https://github.com/NVIDIA/TensorRT/pull/738)
- Support 3D inputs in InstanceNormalization plugin [#745](https://github.com/NVIDIA/TensorRT/pull/745)
- Added this CHANGELOG.md

### Changed
- ONNX GraphSurgeon - v0.2.7 with bugfixes, new examples.
- demo/BERT bugfixes for Jetson Xavier
- Updated build Dockerfile to cuda-11.1
- Updated ClangFormat style specification according to TensorRT coding guidelines

### Removed
- N/A


## [7.2.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-7.html#rel_7-2-1) - 2020-10-20
### Added
- [Polygraphy](tools/Polygraphy) v0.20.13 - Deep Learning Inference Prototyping and Debugging Toolkit
- [PyTorch-Quantization Toolkit](tools/pytorch-quantization) v2.0.0
- Updated BERT plugins for [variable sequence length inputs](demo/BERT#variable-sequence-length)
  - Optimized kernels for sequence lengths of 64 and 96 added
- Added Tacotron2 + Waveglow TTS demo [#677](https://github.com/NVIDIA/TensorRT/pull/677)
- Re-enable `GridAnchorRect_TRT` plugin with rectangular feature maps [#679](https://github.com/NVIDIA/TensorRT/pull/679)
- Update batchedNMS plugin to IPluginV2DynamicExt interface [#738](https://github.com/NVIDIA/TensorRT/pull/738)
- Support 3D inputs in InstanceNormalization plugin [#745](https://github.com/NVIDIA/TensorRT/pull/745)
- Added this CHANGELOG.md

### Changed
- ONNX GraphSurgeon - [v0.2.7](tools/onnx-graphsurgeon/CHANGELOG.md#v027-2020-09-29) with bugfixes, new examples.
- demo/BERT bugfixes for Jetson Xavier
- Updated build Dockerfile to cuda-11.1
- Updated ClangFormat style specification according to TensorRT coding guidelines

### Removed
- N/A
