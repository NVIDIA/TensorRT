# TensorRT OSS Release Changelog

## [8.0.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-8.html#tensorrt-8) - 2021-07-02
### Added
- Added support for the following ONNX operators: `Celu`, `CumSum`, `EyeLike`, `GatherElements`, `GlobalLpPool`, `GreaterOrEqual`, `LessOrEqual`, `LpNormalization`, `LpPool`, `ReverseSequence`, and `SoftmaxCrossEntropyLoss` [details](). 
- Rehauled `Resize` ONNX operator, now fully supporting the following modes:
  - Coordinate Transformation modes: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, and `align_corners`.
  - Modes: `nearest`, `linear`.
  - Nearest Modes: `floor`, `ceil`, `round_prefer_floor`, `round_prefer_ceil`.
- Added support for multi-input ONNX `ConvTranpose` operator.
- Added support for 3D spatial dimensions in ONNX `InstanceNormalization`.
- Added support for generic 2D padding in ONNX.
- ONNX `QuantizeLinear` and `DequantizeLinear` operators leverage `IQuantizeLayer` and `IDequantizeLayer`.
  - Added support for tensor scales.
  - Added support for per-axis quantization.
- Added `EfficientNMS_TRT`, `EfficientNMS_ONNX_TRT` plugins and experimental support for ONNX `NonMaxSuppression` operator.
- Added `ScatterND` plugin.
- Added TensorRT [QuickStart Guide](https://github.com/NVIDIA/TensorRT/tree/master/quickstart).
- Added new samples: [engine_refit_onnx_bidaf](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#engine_refit_onnx_bidaf) builds an engine from ONNX BiDAF model and refits engine with new weights, [efficientdet](samples/python/efficientdet) and [efficientnet](samples/python/efficientnet) samples for demonstrating Object Detection using TensorRT.
- Added support for Ubuntu20.04 and RedHat/CentOS 8.3.
- Added Python 3.9 support.

### Changed
- Update Polygraphy to [v0.30.3](tools/Polygraphy/CHANGELOG.md#v0303-2021-06-25).
- Update ONNX-GraphSurgeon to [v0.3.10](tools/onnx-graphsurgeon/CHANGELOG.md#v0310-2021-05-20).
- Update Pytorch Quantization toolkit to v2.1.0.
- Notable TensorRT API updates
  - TensorRT now declares API’s with the `noexcept` keyword. All TensorRT classes that an application inherits from (such as IPluginV2) must guarantee that methods called by TensorRT do not throw uncaught exceptions, or the behavior is undefined.
  - Destructors for classes with `destroy()` methods were previously protected. They are now public, enabling use of smart pointers for these classes. The `destroy()` methods are deprecated.
- Moved `RefitMap` API from ONNX parser to core TensorRT.
- Various bugfixes for plugins, samples and ONNX parser.
- Port demoBERT to tensorflow2 and update UFF samples to leverage nvidia-tensorflow1 container.

### Removed
- `IPlugin` and `IPluginFactory` interfaces were deprecated in TensorRT 6.0 and have been removed in TensorRT 8.0. We recommend that you write new plugins or refactor existing ones to target the `IPluginV2DynamicExt` and `IPluginV2IOExt` interfaces. For more information, refer to [Migrating Plugins From TensorRT 6.x Or 7.x To TensorRT 8.x.x](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#migrating-plugins-6x-7x-to-8x).
  - For plugins based on `IPluginV2DynamicExt` and `IPluginV2IOExt`, certain methods with legacy function signatures (derived from `IPluginV2` and `IPluginV2Ext` base classes) which were deprecated and marked for removal in TensorRT 8.0 will no longer be available.
- Removed `samplePlugin` since it showcased IPluginExt interface, which is no longer supported in TensorRT 8.0.
- Removed `sampleMovieLens` and `sampleMovieLensMPS`.
- Removed Dockerfile for Ubuntu 16.04. TensorRT 8.0 debians for Ubuntu 16.04 require python 3.5 while minimum required python version for TensorRT OSS is 3.6.
- Removed support for PowerPC builds, consistent with TensorRT GA releases.

### Notes
- We had deprecated the Caffe Parser and UFF Parser in TensorRT 7.0. They are still tested and functional in TensorRT 8.0, however, we plan to remove the support in a future release. Ensure you migrate your workflow to use `tf2onnx`, `keras2onnx` or [TensorFlow-TensorRT (TF-TRT)](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html).
- Refer to [TensorRT 8.0.1 GA Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-801/release-notes/tensorrt-8.html#rel_8-0-1) for additional details


## [21.06](https://github.com/NVIDIA/TensorRT/releases/tag/21.06) - 2021-06-23
### Added
- Add switch for batch-agnostic mode in NMS plugin
- Add missing model.py in `uff_custom_plugin` sample

### Changed
- Update to [Polygraphy v0.29.2](tools/Polygraphy/CHANGELOG.md#v0292-2021-04-30)
- Update to [ONNX-GraphSurgeon v0.3.9](tools/onnx-graphsurgeon/CHANGELOG.md#v039-2021-04-20)
- Fix numerical errors for float type in NMS/batchedNMS plugins
- Update demoBERT input dimensions to match Triton requirement [#1051](https://github.com/NVIDIA/TensorRT/pull/1051)
- Optimize TLT MaskRCNN plugins:
  - enable fp16 precision in multilevelCropAndResizePlugin and multilevelProposeROIPlugin
  - Algorithms optimization for NMS kernels and ROIAlign kernel
  - Fix invalid cuda config issue when bs is larger than 32
  - Fix issues found  on Jetson NANO

### Removed
- Removed fcplugin from demoBERT to improve latency


## [21.05](https://github.com/NVIDIA/TensorRT/releases/tag/21.05) - 2021-05-20
### Added
- Extended support for ONNX operator `InstanceNormalization` to 5D tensors
- Support negative indices in ONNX `Gather` operator
- Add support for importing ONNX double-typed weights as float
- [ONNX-GraphSurgeon (v0.3.7)](tools/onnx-graphsurgeon/CHANGELOG.md#v037-2021-03-31) support for models with externally stored weights

### Changed
- Update ONNX-TensorRT to [21.05](https://github.com/onnx/onnx-tensorrt/releases/tag/21.05)
- [Relicense ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt/blob/master/LICENSE) under Apache2
- demoBERT builder fixes for multi-batch
- Speedup demoBERT build using global timing cache and disable cuDNN tactics
- Standardize python package versions across OSS samples
- Bugfixes in multilevelProposeROI and bertQKV plugin
- Fix memleaks in samples logger


## [21.04](https://github.com/NVIDIA/TensorRT/releases/tag/21.04) - 2021-04-12
### Added
- SM86 kernels for BERT MHA plugin
- Added opset13 support for `SoftMax`, `LogSoftmax`, `Squeeze`, and `Unsqueeze`.
- Added support for the `EyeLike` and `GatherElements` operators.

### Changed
- Updated TensorRT version to v7.2.3.4.
- Update to ONNX-TensorRT [21.03](https://github.com/onnx/onnx-tensorrt/releases/tag/21.03)
- ONNX-GraphSurgeon (v0.3.4) - updates fold_constants to correctly exit early.
- Set default CUDA_INSTALL_DIR [#798](https://github.com/NVIDIA/TensorRT/pull/798)
- Plugin bugfixes, qkv kernels for sm86
- Fixed GroupNorm CMakeFile for cu sources [#1083](https://github.com/NVIDIA/TensorRT/pull/1083)
- Permit groupadd with non-unique GID in build containers [#1091](https://github.com/NVIDIA/TensorRT/pull/1091)
- Avoid `reinterpret_cast` [#146](https://github.com/NVIDIA/TensorRT/pull/146)
- Clang-format plugins and samples
- Avoid arithmetic on void pointer in multilevelProposeROIPlugin.cpp [#1028](https://github.com/NVIDIA/TensorRT/pull/1028)
- Update BERT plugin documentation.

### Removed
- Removes extra terminate call in InstanceNorm


## [21.03](https://github.com/NVIDIA/TensorRT/releases/tag/21.03) - 2021-03-09
### Added
- Optimized FP16 NMS/batchedNMS plugins with n-bit radix sort and based on `IPluginV2DynamicExt`
- `ProposalDynamic` and `CropAndResizeDynamic` plugins based on `IPluginV2DynamicExt`

### Changed
- [ONNX-TensorRT v21.03 update](https://github.com/onnx/onnx-tensorrt/blob/master/docs/Changelog.md#2103-container-release---2021-03-09)
- [ONNX-GraphSurgeon v0.3.3 update](tools/onnx-graphsurgeon/CHANGELOG.md#v03-2021-03-04)
- Bugfix for `scaledSoftmax` kernel

### Removed
- N/A


## [21.02](https://github.com/NVIDIA/TensorRT/releases/tag/21.02) - 2021-02-01
### Added
- [TensorRT Python API bindings](python)
- [TensorRT Python samples](samples/python)
- FP16 support to batchedNMSPlugin [#1002](https://github.com/NVIDIA/TensorRT/pull/1002)
- Configurable input size for TLT MaskRCNN Plugin [#986](https://github.com/NVIDIA/TensorRT/pull/986)

### Changed
- TensorRT version updated to 7.2.2.3
- [ONNX-TensorRT v21.02 update](https://github.com/onnx/onnx-tensorrt/blob/master/docs/Changelog.md#2102-container-release---2021-01-22)
- [Polygraphy v0.21.1 update](tools/Polygraphy/CHANGELOG.md#v0211-2021-01-12)
- [PyTorch-Quantization Toolkit](tools/pytorch-quantization) v2.1.0 update
  - Documentation update, ONNX opset 13 support, ResNet example
- [ONNX-GraphSurgeon v0.28 update](tools/onnx-graphsurgeon/CHANGELOG.md#v028-2020-10-08)
- [demoBERT builder](demo/BERT) updated to work with Tensorflow2 (in compatibility mode)
- Refactor [Dockerfiles](docker) for OSS container

### Removed
- N/A


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
