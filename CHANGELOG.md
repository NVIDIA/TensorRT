# TensorRT OSS Release Changelog

## 10.14 GA - 2025-11-7
- Sample changes
  - Replace all pycuda usages with cuda-python APIs
  - Removed the efficientnet samples
  - Deprecated tensorflow_object_detection and efficientdet samples
  - Samples will no longer be released with the packages. The TensorRT GitHub repository will be the single source.


- Parsers:
  - Added support for the `Attention` operator
  - Improved refit for `ConstantOfShape` nodes

- Demos
  - demoDiffusion:
    - Added support for the Cosmos-Predict2 text2image and video2world pipelines


## 10.13.3 GA - 2025-9-8
- Added support for TensorRT API Capture and Replay feature, see the [developer guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html) for more information.
- Demo changes
  - Added support for Flux Kontext pipeline.

## 10.13.2 GA - 2025-8-18
- Added support for CUDA 13.0, dropped support for CUDA 11.X
- Dropped support for Ubuntu 20.04
- Dropped support for Python versions < 3.10 for samples and demos

## 10.13.0 GA - 2025-7-24
- Plugin changes
  - Fixed a division-by-zero error in geluPlugin that occured when the bias is omitted.
  - Completed transition away from using static plugin field/attribute member variables in standard plugins. There's no such need since presently, TRT does not access field information after plugin creators are destructed (deregistered from the plugin registry), nor does access such information without a creator instance.
- Sample changes
  - Deprecated the `yolov3_onnx` sample due to unstable url of yolo weights.
  - Updated the `1_run_onnx_with_tensorrt` and `2_construct_network_with_layer_apis` samples to use `cuda-python` instead of `PyCUDA` for latest GPU/CUDA support.
- Parser changes
  - Decreased memory usage when importing models with external weights
  - Added `loadModelProto`, `loadInitializer` and `parseModelProto` APIs for IParser. These APIs are meant to be used to load user initializers when parsing ONNX models.
  - Added `loadModelProto`, `loadInitializer` and `refitModelProto` APIs for IParserRefitter. These APIs are meant to be used to load user initializers when refitting ONNX models.
  - Deprecated `IParser::parseWithWeightDescriptors`.

## 10.12.0 GA - 2025-6-10
- Plugin changes
  - Migrated `IPluginV2`-descendent version 1 of `cropAndResizeDynamic`, to version 2, which implements `IPluginV3`.
  - Note: The newer versions preserve the attributes and I/O of the corresponding older plugin version. The older plugin versions are deprecated and will be removed in a future release
  - Deprecated the listed versions of the following plugins:
    - `DecodeBbox3DPlugin` (version 1)
    - `DetectionLayer_TRT` (version 1)
    - `EfficientNMS_TRT` (version 1)
    - `FlattenConcat_TRT` (version 1)
    - `GenerateDetection_TRT` (version 1)
    - `GridAnchor_TRT` (version 1)
    - `GroupNormalizationPlugin` (version 1)
    - `InstanceNormalization_TRT` (version 2)
    - `ModulatedDeformConv2d` (version 1)
    - `MultilevelCropAndResize_TRT` (version 1)
    - `MultilevelProposeROI_TRT` (version 1)
    - `RPROI_TRT` (version 1)
    - `PillarScatterPlugin` (version 1)
    - `PriorBox_TRT` (version 1)
    - `ProposalLayer_TRT` (version 1)
    - `ProposalDynamic` (version 1)
    - `Region_TRT` (version 1)
    - `Reorg_TRT` (version 2)
    - `ResizeNearest_TRT` (version 1)
    - `ScatterND` (version 1)
    - `VoxelGeneratorPlugin` (version 1)
- Demo changes
  - Added [Image-to-Image](demo/Diffusion#generate-an-image-with-stable-diffusion-v35-large-with-controlnet-guided-by-an-image-and-a-text-prompt) support for Stable Diffusion v3.5-large ControlNet models.
  - Enabled download of [pre-exported ONNX models](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-tensorrt) for the Stable Diffusion v3.5-large pipeline.
- Sample changes
  - Added two refactored python samples [1_run_onnx_with_tensorrt](samples/python/refactored/1_run_onnx_with_tensorrt) and [2_construct_network_with_layer_apis](samples/python/refactored/2_construct_network_with_layer_apis)
- Parser changes
  - Added support for integer-typed base tensors for `Pow` operations
  - Added support for custom `MXFP8` quantization operations
  - Added support for ellipses, diagonal, and broadcasting in `Einsum` operations


## 10.11.0 GA - 2025-5-16

Key Features and Updates:

- Plugin changes
  - Migrated `IPluginV2`-descendent version 1 of `cropAndResizePluginDynamic`, to version 2, which implements `IPluginV3`.
  - Migrated `IPluginV2`-descendent version 1 of `DisentangledAttention_TRT`, to version 2, which implements `IPluginV3`.
  - Migrated `IPluginV2`-descendent version 1 of `MultiscaleDeformableAttnPlugin_TRT`, to version 2, which implements `IPluginV3`.
  - Note: The newer versions preserve the attributes and I/O of the corresponding older plugin version. The older plugin versions are deprecated and will be removed in a future release.
- Demo changes
  - demoDiffusion
    - Added support for Stable Diffusion 3.5-medium and 3.5-large pipelines in BF16 and FP16 precisions.
    - Added support for Stable Diffusion 3.5-large pipeline in FP8 precision.
- Parser changes
  - Added `kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA` parser flag to enable UINT8 asymmetric quantization on engines targeting DLA.
  - Removed restriction that inputs to `RandomNormalLike` and `RandomUniformLike` must be tensors.
  - Clarified limitations of scan outputs for `Loop` nodes.

## 10.10.0 GA - 2025-4-28

Key Features and Updates:

- Plugin changes
  - Deprecated the enum classes [PluginVersion](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/namespacenvinfer1.html#a6fb3932a2896d82a94c8783e640afb34) & [PluginCreatorVersion](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/namespacenvinfer1.html#a43c4159a19c23f74234f3c34124ea0c5). `PluginVersion` & `PluginCreatorVersion` are used only in relation to `IPluginV2`-descendent plugin interfaces, which are all deprecated.
  - Added the following APIs that enable users to obtain a list of all Plugin Creators hierarchically registered to a TensorRT `IPluginRegistry` (`C++`, `Python`) instance.
    - C++ API: `IPluginRegistry::getAllCreatorsRecursive()`
    - Python API: `IPluginRegistry.all_creators_recursive`
- Demo changes
  - demoDiffusion
    - Added FP16 and FP8 LoRA support for the SDXL and FLUX pipelines.
    - Added FP16 ControlNet support for the SDXL pipeline.
- Sample changes
  - Added support for the [python_plugin](https://github.com/NVIDIA/TensorRT/tree/release/10.9/samples/python/python_plugin) sample to compile targets to Blackwell.
- Parser changes
  - Cleaned up log spam when the ONNX network contained a mixture of Plugins and LocalFunctions.
  - UINT8 constants are now properly imported for `QuantizeLinear` & `DequantizeLinear` nodes.
  - Plugin fallback importer now also reads its namespace from a Node's domain field.

## 10.9.0 GA - 2025-3-10

Key Features and Updates:

- Demo changes
  - demoDiffusion
    - Added Canny ControlNet support for the SDXL pipeline
- Plugin changes
  - Added a readme to the GroupNormalization plugin (`GroupNormalizationPlugin`) - [4314](https://github.com/NVIDIA/TensorRT/issues/4314)
  - Fixed bug in `CustomQKVToConte mxtPluginDynamic` version 3 where SM 100 was not considered a supported platform.
- Parser changes
  - Added support for Python AOT plugins
  - Added support for opset 21 GroupNorm - [4336](https://github.com/NVIDIA/TensorRT/issues/4336)
  - Fixed support for opset 18+ ScatterND
- Sample changes
  - Added a new sample `dds_faster_rcnn` which demonstrates how to handle data-dependent shaped outputs with `IOutputAllocator`.
- Fixed issues:
  - Fixed streamReaderV2 Python API performance issue - [4327](https://github.com/NVIDIA/TensorRT/issues/4327)

## 10.8.0 GA - 2025-1-31

Key Features and Updates:

- Demo changes
  - demoDiffusion
    - Added [Image-to-Image](demo/Diffusion#generate-an-image-guided-by-an-initial-image-and-a-text-prompt-using-flux) support for Flux-1.dev and Flux.1-schnell pipelines.
    - Added [ControlNet](demo/Diffusion#generate-an-image-guided-by-a-text-prompt-and-a-control-image-using-flux-controlnet) support for [FLUX.1-Canny-dev](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) and [FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) pipelines. Native FP8 quantization is also supported for these pipelines.
    - Added support for ONNX model export only mode. See [--onnx-export-only](demo/Diffusion/README.md#4-export-onnx-models-only-skip-inference).
    - Added FP16, BF16, FP8, and FP4 support for all Flux Pipelines.
- Plugin changes
  - Added SM 100 and SM 120 support to bertQKVToContextPlugin. This enables demo/BERT on Blackwell GPUs.
- Sample changes
  - Added a new `sampleEditableTimingCache` to demonstrate how to build an engine with the desired tactics by modifying the timing cache.
  - Deleted the `sampleAlgorithmSelector` sample.
  - Fixed `sampleOnnxMNIST` by updating the correct INT8 dynamic range.
- Parser changes
  - Added support for `FLOAT4E2M1` types for quantized networks.
  - Added support for dynamic axes and improved performance of `CumSum` operations.
  - Fixed the import of local functions when their input tensor names aliased one from an outside scope.
  - Added support for `Pow` ops with integer-typed exponent values.
- Fixed issues
  - Fixed segmentation of boolean constant nodes - [4224](https://github.com/NVIDIA/TensorRT/issues/4224).
  - Fixed accuracy issue when multiple optimization profiles were defined [4250](https://github.com/NVIDIA/TensorRT/issues/4250).

## 10.7.0 GA - 2024-12-4

Key Feature and Updates:

- Demo Changes

  - demoDiffusion
    - Enabled low-vram for the Flux pipeline. Users can now run the pipelines on systems with 32GB VRAM.
    - Added support for [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) pipeline.
    - Enabled weight streaming mode for Flux pipeline.

- Plugin Changes

  - On Blackwell and later platforms, TensorRT will drop cuDNN support on the following categories of plugins
    - User-written `IPluginV2Ext`, `IPluginV2DynamicExt`, and `IPluginV2IOExt` plugins that are dependent on cuDNN handles provided by TensorRT (via the `attachToContext()` API).
    - TensorRT standard plugins that use cuDNN, specifically:
      - `InstanceNormalization_TRT` (version: 1, 2, and 3) present in `plugin/instanceNormalizationPlugin/`.
      - `GroupNormalizationPlugin` (version: 1) present in `plugin/groupNormalizationPlugin/`.
      - Note: These normalization plugins are superseded by TensorRT’s native `INormalizationLayer` ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_normalization_layer.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Normalization.html)). TensorRT support for cuDNN-dependent plugins remain unchanged on pre-Blackwell platforms.

- Parser Changes

  - Now prioritizes using plugins over local functions when a corresponding plugin is available in the registry.
  - Added dynamic axes support for `Squeeze` and `Unsqueeze` operations.
  - Added support for parsing mixed-precision `BatchNormalization` nodes in strongly-typed mode.

- Addressed Issues
  - Fixed [4113](https://github.com/NVIDIA/TensorRT/issues/4113).

## 10.6.0 GA - 2024-11-05

Key Feature and Updates:

- Demo Changes

  - demoBERT: The use of `fcPlugin` in demoBERT has been removed.
  - demoBERT: All TensorRT plugins now used in demoBERT (`CustomEmbLayerNormDynamic`, `CustomSkipLayerNormDynamic`, and `CustomQKVToContextDynamic`) now have versions that inherit from IPluginV3 interface classes. The user can opt-in to use these V3 plugins by specifying `--use-v3-plugins` to the builder scripts.
    - Opting-in to use V3 plugins does not affect performance, I/O, or plugin attributes.
    - There is a known issue in the V3 (version 4) of `CustomQKVToContextDynamic` plugin from TensorRT 10.6.0, causing an internal assertion error if either the batch or sequence dimensions differ at runtime from the ones used to serialize the engine. See the “known issues” section of the [TensorRT-10.6.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html#rel-10-6-0).
    - For smoother migration, the default behavior is still using the deprecated `IPluginV2DynamicExt`-derived plugins, when the flag: `--use-v3-plugins` isn't specified in the builder scripts. The flag `--use-deprecated-plugins` was added as an explicit way to enforce the default behavior, and is mutually exclusive with `--use-v3-plugins`.
  - demoDiffusion
    - Introduced BF16 and FP8 support for the [Flux.1-dev](demo/Diffusion#generate-an-image-guided-by-a-text-prompt-using-flux) pipeline.
    - Expanded FP8 support on Ada platforms.
    - Enabled LoRA adapter compatibility for SDv1.5, SDv2.1, and SDXL pipelines using Diffusers version 0.30.3.

- Sample Changes

  - Added the Python sample [quickly_deployable_plugins](samples/python/quickly_deployable_plugins), which demonstrates quickly deployable Python-based plugin definitions (QDPs) in TensorRT. QDPs are a simple and intuitive decorator-based approach to defining TensorRT plugins, requiring drastically less code.

- Plugin Changes

  - The `fcPlugin` has been deprecated. Its functionality has been superseded by the [IMatrixMultiplyLayer](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_matrix_multiply_layer.html) that is natively provided by TensorRT.
  - Migrated `IPluginV2`-descendent version 1 of `CustomEmbLayerNormDynamic`, to version 6, which implements `IPluginV3`.
    - The newer versions preserve the attributes and I/O of the corresponding older plugin version.
    - The older plugin versions are deprecated and will be removed in a future release.

- Parser Changes

  - Updated ONNX submodule version to 1.17.0.
  - Fixed issue where conditional layers were incorrectly being added.
  - Updated local function metadata to contain more information.
  - Added support for parsing nodes with Quickly Deployable Plugins.
  - Fixed handling of optional outputs.

- Tool Updates
  - ONNX-Graphsurgeon updated to version 0.5.3
  - Polygraphy updated to 0.49.14.

## 10.5.0 GA - 2024-09-30

Key Features and Updates:

- Demo changes
  - Added [Flux.1-dev](demo/Diffusion) pipeline
- Sample changes
  - None
- Plugin changes
  - Migrated `IPluginV2`-descendent versions of `bertQKVToContextPlugin` (1, 2, 3) to newer versions (4, 5, 6 respectively) which implement `IPluginV3`.
  - Note:
    - The newer versions preserve the attributes and I/O of the corresponding older plugin version
    - The older plugin versions are deprecated and will be removed in a future release
- Quickstart guide
  - None
- Parser changes
  - Added support for real-valued `STFT` operations
  - Improved error handling in `IParser`

Known issues:

- Demos:
  - TensorRT engine might not be build successfully when using `--fp8` flag on H100 GPUs.

## 10.4.0 GA - 2024-09-18

Key Features and Updates:

- Demo changes
  - Added [Stable Cascade](demo/Diffusion) pipeline.
  - Enabled INT8 and FP8 quantization for Stable Diffusion v1.5, v2.0 and v2.1 pipelines.
  - Enabled FP8 quantization for Stable Diffusion XL pipeline.
- Sample changes
  - Add a new python sample `aliased_io_plugin` which demonstrates how in-place updates to plugin inputs can be achieved through I/O aliasing.
- Plugin changes

  - Migrated IPluginV2-descendent versions (a) of the following plugins to newer versions (b) which implement IPluginV3 (a->b):
    - scatterElementsPlugin (1->2)
    - skipLayerNormPlugin (1->5, 2->6, 3->7, 4->8)
    - embLayerNormPlugin (2->4, 3->5)
    - bertQKVToContextPlugin (1->4, 2->5, 3->6)
  - Note
    - The newer versions preserve the corresponding attributes and I/O of the corresponding older plugin version.
    - The older plugin versions are deprecated and will be removed in a future release.

- Quickstart guide
  - Updated deploy_to_triton guide and removed legacy APIs.
  - Removed legacy TF-TRT code as the project is no longer supported.
  - Removed quantization_tutorial as pytorch_quantization has been deprecated. Check out https://github.com/NVIDIA/TensorRT-Model-Optimizer for the latest quantization support. Check [Stable Diffusion XL (Base/Turbo) and Stable Diffusion 1.5 Quantization with Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/diffusers/quantization) for integration with TensorRT.
- Parser changes

  - Added support for tensor `axes` for `Pad` operations.
  - Added support for `BlackmanWindow`, `HammingWindow`, and `HannWindow` operations.
  - Improved error handling in `IParserRefitter`.
  - Fixed kernel shape inference in multi-input convolutions.

- Updated tooling
  - polygraphy-extension-trtexec v0.0.9

## 10.3.0 GA - 2024-08-02

Key Features and Updates:

- Demo changes
  - Added [Stable Video Diffusion](demo/Diffusion)(`SVD`) pipeline.
- Plugin changes
  - Deprecated Version 1 of [ScatterElements plugin](plugin/scatterElementsPlugin). It is superseded by Version 2, which implements the `IPluginV3` interface.
- Quickstart guide
  - Updated the [SemanticSegmentation](quickstart/SemanticSegmentation) guide with latest APIs.
- Parser changes
  - Added support for tensor `axes` inputs for `Slice` node.
  - Updated `ScatterElements` importer to use Version 2 of [ScatterElements plugin](plugin/scatterElementsPlugin), which implements the `IPluginV3` interface.
- Updated tooling
  - Polygraphy v0.49.13

## 10.2.0 GA - 2024-07-09

Key Features and Updates:

- Demo changes
  - Added [Stable Diffusion 3 demo](demo/Diffusion).
- Plugin changes
  - Version 3 of the [InstanceNormalization plugin](plugin/instanceNormalizationPlugin/) (`InstanceNormalization_TRT`) has been added. This version is based on the `IPluginV3` interface and is used by the TensorRT ONNX parser when native `InstanceNormalization` is disabled.
- Tooling changes
  - Pytorch Quantization development has transitioned to [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer). All developers are encouraged to use TensorRT Model Optimizer to benefit from the latest advancements on quantization and compression.
- Build containers
  - Updated default cuda versions to `12.5.0`.

## 10.1.0 GA - 2024-06-17

Key Features and Updates:

- Parser changes
  - Added `supportsModelV2` API
  - Added support for `DeformConv` operation
  - Added support for `PluginV3` TensorRT Plugins
  - Marked all IParser and IParserRefitter APIs as `noexcept`
- Plugin changes
  - Added version 2 of ROIAlign_TRT plugin, which implements the IPluginV3 plugin interface. When importing an ONNX model with the RoiAlign op, this new version of the plugin will be inserted to the TRT network.
- Samples changes
  - Added a new sample [non_zero_plugin](samples/python/non_zero_plugin), which is a Python version of the C++ sample [sampleNonZeroPlugin](samples/sampleNonZeroPlugin).
- Updated tooling
  - Polygraphy v0.49.12
  - ONNX-GraphSurgeon v0.5.3

## 10.0.1 GA - 2024-04-24

Key Features and Updates:

- Parser changes
  - Added support for building with `protobuf-lite`.
  - Fixed issue when parsing and refitting models with nested `BatchNormalization` nodes.
  - Added support for empty inputs in custom plugin nodes.
- Demo changes
  - The following demos have been removed: Jasper, Tacotron2, HuggingFace Diffusers notebook
- Updated tooling
  - Polygraphy v0.49.10
  - ONNX-GraphSurgeon v0.5.2
- Build Containers
  - Updated default cuda versions to `12.4.0`.
  - Added Rocky Linux 8 and Rocky Linux 9 build containers

## 10.0.0 EA - 2024-03-27

Key Features and Updates:

- Samples changes
  - Added a [sample](samples/python/sample_weight_stripping) showcasing weight-stripped engines.
  - Added a [sample](samples/python/python_plugin/circ_pad_plugin_multi_tactic.py) demonstrating the use of custom tactics with IPluginV3.
  - Added a [sample](samples/sampleNonZeroPlugin) to showcase plugins with data-dependent output shapes, using IPluginV3.
- Parser changes
  - Added a new class `IParserRefitter` that can be used to refit a TensorRT engine with the weights of an ONNX model.
  - `kNATIVE_INSTANCENORM` is now set to ON by default.
  - Added support for `IPluginV3` interfaces from TensorRT.
  - Added support for `INT4` quantization.
  - Added support for the `reduction` attribute in `ScatterElements`.
  - Added support for `wrap` padding mode in `Pad`
- Plugin changes
  - A [new plugin](plugin/scatterElementsPlugin) has been added in compliance with [ONNX ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements).
  - The TensorRT plugin library no longer has a load-time link dependency on cuBLAS or cuDNN libraries.
  - All plugins which relied on cuBLAS/cuDNN handles passed through `IPluginV2Ext::attachToContext()` have moved to use cuBLAS/cuDNN resources initialized by the plugin library itself. This works by dynamically loading the required cuBLAS/cuDNN library. Additionally, plugins which independently initialized their cuBLAS/cuDNN resources have also moved to dynamically loading the required library. If the respective library is not discoverable through the library path(s), these plugins will not work.
  - bertQKVToContextPlugin: Version 2 of this plugin now supports head sizes less than or equal to 32.
  - reorgPlugin: Added a version 2 which implements IPluginV2DynamicExt.
  - disentangledAttentionPlugin: Fixed a kernel bug.
- Demo changes
  - HuggingFace demos have been removed. For all users using TensorRT to accelerate Large Language Model inference, please use [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/).
- Updated tooling
  - Polygraphy v0.49.9
  - ONNX-GraphSurgeon v0.5.1
  - TensorRT Engine Explorer v0.1.8
- Build Containers
  - RedHat/CentOS 7.x are no longer officially supported starting with TensorRT 10.0. The corresponding container has been removed from TensorRT-OSS.

## 9.3.0 GA - 2024-02-09

Key Features and Updates:

- Demo changes
  - Faster Text-to-image using SDXL & INT8 quantization using AMMO
- Updated tooling
  - Polygraphy v0.49.7

## 9.2.0 GA - 2023-11-27

Key Features and Updates:

- `trtexec` enhancement: Added `--weightless` flag to mark the engine as weightless.
- Parser changes
  - Added support for Hardmax operator.
  - Changes to a few operator importers to ensure that TensorRT preserves the precision of operations when using strongly typed mode.
- Plugin changes
  - Explicit INT8 support added to `bertQKVToContextPlugin`.
  - Various bug fixes.
- Updated HuggingFace demo to use transformers v4.31.0 and PyTorch v2.1.0.

## 9.1.0 GA - 2023-10-18

Key Features and Updates:

- Update the [trt_python_plugin](samples/python/python_plugin) sample.
  - Python plugins API reference is part of the offical TRT Python API.
- Added samples demonstrating the usage of the progress monitor API.
  - Check [sampleProgressMonitor](samples/sampleProgressMonitor) for the C++ sample.
  - Check [simple_progress_monitor](samples/python/simple_progress_monitor) for the Python sample.
- Remove dependencies related to python<3.8 in python samples as we no longer support python<3.8 for python samples.
- Demo changes
  - Added LAMBADA dataset accuracy checks in the [HuggingFace](demo/HuggingFace) demo.
  - Enabled structured sparsity and FP8 quantized batch matrix multiplication(BMM)s in attention in the [NeMo](demo/NeMo) demo.
  - Replaced deprecated APIs in the [BERT](demo/BERT) demo.
- Updated tooling
  - Polygraphy v0.49.1

## 9.0.1 GA - 2023-09-07

Key Features and Updates:

- TensorRT plugin autorhing in Python is now supported
  - See the [trt_python_plugin](samples/python/python_plugin) sample for reference.
- Updated default CUDA version to 12.2
- Support for BLIP models, Seq2Seq and Vision2Seq abstractions in HuggingFace demo.
- demoDiffusion refactoring and SDXL enhancements
- Additional validation asserts for NV Plugins
- Updated tooling
  - TensorRT Engine Explorer v0.1.7: graph rendering for TensorRT 9.0 `kgen` kernels
  - ONNX-GraphSurgeon v0.3.29
  - PyTorch quantization toolkit v2.2.0

## 9.0.0 EA - 2023-08-06

Key Features and Updates:

- Added the NeMo demo to demonstrate the performance benefit of using E4M3 FP8 data type with the GPT models trained with the [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo) and [TransformerEngine](https://github.com/NVIDIA/TransformerEngine).
- Demo Diffusion updates

  - Added SDXL 1.0 txt2img pipeline
  - Added ControlNet pipeline
  - Huggingface demo updates
    - Added Flan-T5, OPT, BLOOM, BLOOMZ, GPT-Neo, GPT-NeoX, Cerebras-GPT support with accuracy check
    - Refactored code and extracted common utils into Seq2Seq class
    - Optimized shape-changing overhead and achieved a >30% e2e performance gain
    - Added stable KV-cache, beam search and fp16 support for all models
    - Added dynamic batch size TRT inference
    - Added uneven-length multi-batch inference with attention_mask support
    - Added `chat` command – interactive CLI
    - Upgraded PyTorch and HuggingFace version to support Hopper GPU
    - Updated notebooks with much simplified demo API.

- Added two new TensorRT samples: sampleProgressMonitor (C++) and simple_progress_reporter (Python) that are examples for using Progress Monitor during engine build.
- The following plugins were deprecated:

  - `BatchedNMS_TRT`
  - `BatchedNMSDynamic_TRT`
  - `BatchTilePlugin_TRT`
  - `Clip_TRT`
  - `CoordConvAC`
  - `CropAndResize`
  - `EfficientNMS_ONNX_TRT`
  - `CustomGeluPluginDynamic`
  - `LReLU_TRT`
  - `NMSDynamic_TRT`
  - `NMS_TRT`
  - `Normalize_TRT`
  - `Proposal`
  - `SpecialSlice_TRT`
  - `Split`

- Ubuntu 18.04 has reached end of life and is no longer supported by TensorRT starting with 9.0, and the corresponding Dockerfile(s) have been removed.
- Support for aarch64 builds will not be available in this release, and the corresponding Dockerfiles have been removed.

## [8.6.1 GA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-6-1) - 2023-05-02

TensorRT OSS release corresponding to TensorRT 8.6.1.6 GA release.

- Updates since [TensorRT 8.6.0 EA release](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-6-0-EA).
- Please refer to the [TensorRT 8.6.1.6 GA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-6-1) for more information.

Key Features and Updates:

- Added a new flag `--use-cuda-graph` to demoDiffusion to improve performance.
- Optimized GPT2 and T5 HuggingFace demos to use fp16 I/O tensors for fp16 networks.

## [8.6.0 EA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-6-0-EA) - 2023-03-10

TensorRT OSS release corresponding to TensorRT 8.6.0.12 EA release.

- Updates since [TensorRT 8.5.3 GA release](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-3).
- Please refer to the [TensorRT 8.6.0.12 EA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-6-0-EA) for more information.

Key Features and Updates:

- demoDiffusion acceleration is now supported out of the box in TensorRT without requiring plugins.
  - The following plugins have been removed accordingly: GroupNorm, LayerNorm, MultiHeadCrossAttention, MultiHeadFlashAttention, SeqLen2Spatial, and SplitGeLU.
- Added a new sample called onnx_custom_plugin.

## [8.5.3 GA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-3) - 2023-01-30

TensorRT OSS release corresponding to TensorRT 8.5.3.1 GA release.

- Updates since [TensorRT 8.5.2 GA release](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-2).
- Please refer to the [TensorRT 8.5.3 GA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-3) for more information.

Key Features and Updates:

- Added the following HuggingFace demos: GPT-J-6B, GPT2-XL, and GPT2-Medium
- Added nvinfer1::plugin namespace
- Optimized KV Cache performance for T5

## [8.5.2 GA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-2) - 2022-12-12

TensorRT OSS release corresponding to TensorRT 8.5.2.2 GA release.

- Updates since [TensorRT 8.5.1 GA release](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-1).
- Please refer to the [TensorRT 8.5.2 GA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-2) for more information.

Key Features and Updates:

- Plugin enhancements
  - Added [LayerNormPlugin](plugin/layerNormPlugin), [SplitGeLUPlugin](plugin/splitGeLUPlugin), [GroupNormPlugin](plugin/groupNormPlugin), and [SeqLen2SpatialPlugin](plugin/seqLen2SpatialPlugin) to support [stable diffusion demo](demo/Diffusion).
- KV-cache and beam search to GPT2 and T5 demos

## [22.12](https://github.com/NVIDIA/TensorRT/releases/tag/22.12) - 2022-12-06

### Added

- Stable Diffusion demo using TensorRT Plugins
- KV-cache and beam search to GPT2 and T5 demos
- Perplexity calculation to all HF demos

### Changed

- Updated trex to v0.1.5
- Increased default workspace size in demoBERT to build BS=128 fp32 engines
- Use `avg_iter=8` and timing cache to make demoBERT perf more stable

### Removed

- None

## [8.5.1 GA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-1) - 2022-11-01

TensorRT OSS release corresponding to TensorRT 8.5.1.7 GA release.

- Updates since [TensorRT 8.4.1 GA release](https://github.com/NVIDIA/TensorRT/releases/tag/8.4.1).
- Please refer to the [TensorRT 8.5.1 GA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-5-1) for more information.

Key Features and Updates:

- Samples enhancements

  - Added [sampleNamedDimensions](samples/sampleNamedDimensions) which works with named dimensions.
  - Updated `sampleINT8API` and `introductory_parser_samples` to use `ONNX` models over `Caffe`/`UFF`
  - Removed UFF/Caffe samples including `sampleMNIST`, `end_to_end_tensorflow_mnist`, `sampleINT8`, `sampleMNISTAPI`, `sampleUffMNIST`, `sampleUffPluginV2Ext`, `engine_refit_mnist`, `int8_caffe_mnist`, `uff_custom_plugin`, `sampleFasterRCNN`, `sampleUffFasterRCNN`, `sampleGoogleNet`, `sampleSSD`, `sampleUffSSD`, `sampleUffMaskRCNN` and `uff_ssd`.

- Plugin enhancements

  - Added [GridAnchorRectPlugin](plugin/gridAnchorPlugin) to support rectangular feature maps in gridAnchorPlugin.
  - Added [ROIAlignPlugin](plugin/roiAlignPlugin) to support the ONNX operator [RoiAlign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign). The ONNX parser will automatically route ROIAlign ops through the plugin.
  - Added Hopper support for the [BERTQKVToContextPlugin](plugin/bertQKVToContextPlugin) plugin.
  - Exposed the **use_int8_scale_max** attribute in the [BERTQKVToContextPlugin](plugin/bertQKVToContextPlugin) plugin to allow users to disable the by-default usage of INT8 scale factors to optimize softmax MAX reduction in versions 2 and 3 of the plugin.

- ONNX-TensorRT changes

  - Added support for operator [Reciprocal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal).

- Build containers

  - Updated default cuda versions to `11.8.0`.

- Tooling enhancements
  - Updated [onnx-graphsurgeon](tools/onnx-graphsurgeon) to v0.3.25.
  - Updated [Polygraphy](tools/Polygraphy) to v0.43.1.
  - Updated [polygraphy-extension-trtexec](tool/polygraphy-extension-trtexec) to v0.0.8.
  - Updated [Tensorflow Quantization Toolkit](tools/tensorflow-quantization) to v0.2.0.

## [22.08](https://github.com/NVIDIA/TensorRT/releases/tag/22.08) - 2022-08-16

Updated TensorRT version to 8.4.2 - see the [TensorRT 8.4.2 release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-4-2) for more information

### Changed

- Updated default protobuf version to 3.20.x
- Updated ONNX-TensorRT submodule version to `22.08` tag
- Updated `sampleIOFormats` and `sampleAlgorithmSelector` to use `ONNX` models over `Caffe`

### Fixes

- Fixed missing serialization member in custom `ClipPlugin` plugin used in `uff_custom_plugin` sample
- Fixed various Python import issues

### Added

- Added new DeBERTA demo
- Added version 2 for `disentangledAttentionPlugin` to support DeBERTA v2

### Removed

- None

## [22.07](https://github.com/NVIDIA/TensorRT/releases/tag/22.07) - 2022-07-21

### Added

- `polygraphy-trtexec-plugin` tool for Polygraphy
- Multi-profile support for demoBERT
- KV cache support for HF BART demo

### Changed

- Updated ONNX-GS to `v0.3.20`

### Removed

- None

## [8.4.1 GA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-4-1) - 2022-06-14

TensorRT OSS release corresponding to TensorRT 8.4.1.5 GA release.

- Updates since [TensorRT 8.2.1 GA release](https://github.com/NVIDIA/TensorRT/releases/tag/8.2.1).
- Please refer to the [TensorRT 8.4.1 GA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-4-1) for more information.

Key Features and Updates:

- Samples enhancements

  - Added [Detectron2 Mask R-CNN R50-FPN](samples/python/detectron2/README.md) python sample
  - Added a [quickstart guide](quickstart/deploy_to_triton) for NVidia Triton deployment workflow.
  - Added onnx export script for [sampleOnnxMnistCoordConvAC](samples/sampleOnnxMnistCoordConvAC)
  - Removed `sampleNMT`.
  - Removed usage of deprecated TensorRT APIs in samples.

- EfficientDet sample

  - Added support for EfficientDet Lite and AdvProp models.
  - Added dynamic batch support.
  - Added mixed precision engine builder.

- HuggingFace transformer demo

  - Added BART model.
  - Performance speedup of GPT-2 greedy search using GPU implementation.
  - Fixed GPT2 onnx export failure due to 2G file size limitation.
  - Extended Megatron LayerNorm plugins to support larger hidden sizes.
  - Added performance benchmarking mode.
  - Enable tf32 format by default.

- `demoBERT` enhancements

  - Add `--duration` flag to perf benchmarking script.
  - Fixed import of `nvinfer_plugins` library in demoBERT on Windows.

- Torch-QAT toolkit

  - `quant_bert.py` module removed. It is now upstreamed to [HuggingFace QDQBERT](https://huggingface.co/docs/transformers/model_doc/qdqbert).
  - Use axis0 as default for deconv.
  - [#1939](https://github.com/NVIDIA/TensorRT/issues/1939) - Fixed path in `classification_flow` example.

- Plugin enhancements

  - Added [Disentangled attention plugin](plugin/disentangledAttentionPlugin), `DisentangledAttention_TRT`, to support DeBERTa model.
  - Added [Multiscale deformable attention plugin](plugin/multiscaleDeformableAttnPlugin), `MultiscaleDeformableAttnPlugin_TRT`, to support DDETR model.
  - Added new plugins: [decodeBbox3DPlugin](plugin/decodeBbox3DPlugin), [pillarScatterPlugin](plugin/pillarScatterPlugin), and [voxelGeneratorPlugin](plugin/voxelGeneratorPlugin).
  - Refactored [EfficientNMS plugin](plugin/efficientNMSPlugin) to support [TF-TRT](https://github.com/tensorflow/tensorrt) and implicit batch mode.
  - `fp16` support for `pillarScatterPlugin`.

- Build containers

  - Updated default cuda versions to `11.6.2`.
  - [CentOS Linux 8 has reached End-of-Life](https://www.centos.org/centos-linux-eol/) on Dec 31, 2021. The corresponding container has been removed from TensorRT-OSS.
  - Install `devtoolset-8` for updated g++ versions in CentOS7 container.

- Tooling enhancements

  - Added [Tensorflow Quantization Toolkit](tools/tensorflow-quantization) v0.1.0 for Quantization-Aware-Training of Tensorflow 2.x Keras models.
  - Added [TensorRT Engine Explorer](tools/experimental/trt-engine-explorer/README.md) v0.1.2 for inspecting TensorRT engine plans and associated inference profiling data.
  - Updated [Polygraphy](tools/Polygraphy) to v0.38.0.
  - Updated [onnx-graphsurgeon](tools/onnx-graphsurgeon) to v0.3.19.

- `trtexec` enhancements
  - Added `--layerPrecisions` and `--layerOutputTypes` flags for specifying layer-wise precision and output type constraints.
  - Added `--memPoolSize` flag to specify the size of workspace as well as the DLA memory pools via a unified interface. Correspondingly the `--workspace` flag has been deprecated.
  - "End-To-End Host Latency" metric has been removed. Use the “Host Latency” metric instead. For more information, refer to [Benchmarking Network](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-benchmark) section in the TensorRT Developer Guide.
  - Use `enqueueV2()` instead of `enqueue()` when engine has explicit batch dimensions.

## [22.06](https://github.com/NVIDIA/TensorRT/releases/tag/22.06) - 2022-06-08

### Added

- None

### Changed

- Disentangled attention (DMHA) plugin refactored
- ONNX parser updated to 8.2GA

### Removed

- None

## [22.05](https://github.com/NVIDIA/TensorRT/releases/tag/22.05) - 2022-05-13

### Added

- Disentangled attention plugin for DeBERTa
- DMHA (multiscaleDeformableAttnPlugin) plugin for DDETR
- Performance benchmarking mode to HuggingFace demo

### Changed

- Updated base TensorRT version to 8.2.5.1
- Updated onnx-graphsurgeon v0.3.19 [CHANGELOG](tools/onnx-graphsurgeon/CHANGELOG.md)
- fp16 support for pillarScatterPlugin
- [#1939](https://github.com/NVIDIA/TensorRT/issues/i1939) - Fixed path in quantization `classification_flow`
- Fixed GPT2 onnx export failure due to 2G limitation
- Use axis0 as default for deconv in pytorch-quantization toolkit
- Updated onnx export script for CoordConvAC sample
- Install devtoolset-8 for updated g++ version in CentOS7 container

### Removed

- Usage of deprecated TensorRT APIs in samples removed
- `quant_bert.py` module removed from pytorch-quantization

## [22.04](https://github.com/NVIDIA/TensorRT/releases/tag/22.04) - 2022-04-13

### Added

- TensorRT Engine Explorer v0.1.0 [README](tools/experimental/trt-engine-explorer/README.md)
- Detectron 2 Mask R-CNN R50-FPN python [sample](samples/python/detectron2/README.md)
- Model export script for sampleOnnxMnistCoordConvAC

### Changed

- Updated base TensorRT version to 8.2.4.2
- Updated copyright headers with SPDX identifiers
- Updated onnx-graphsurgeon v0.3.17 [CHANGELOG](tools/onnx-graphsurgeon/CHANGELOG.md)
- `PyramidROIAlign` plugin refactor and bug fixes
- Fixed `MultilevelCropAndResize` crashes on Windows
- [#1583](https://github.com/NVIDIA/TensorRT/issues/1583) - sublicense ieee/half.h under Apache2
- Updated demo/BERT performance tables for rel-8.2
- [#1774](https://github.com/NVIDIA/TensorRT/issues/1774) Fix python hangs at IndexErrors when TF is imported after TensorRT
- Various bugfixes in demos - BERT, Tacotron2 and HuggingFace GPT/T5 notebooks
- Cleaned up sample READMEs

### Removed

- sampleNMT removed from samples

## [22.03](https://github.com/NVIDIA/TensorRT/releases/tag/22.03) - 2022-03-23

### Added

- EfficientDet sample enhancements
  - Added support for EfficientDet Lite and AdvProp models.
  - Added dynamic batch support.
  - Added mixed precision engine builder.

### Changed

- Better decoupling of HuggingFace demo tests

## [22.02](https://github.com/NVIDIA/TensorRT/releases/tag/22.02) - 2022-02-04

### Added

- New plugins: [decodeBbox3DPlugin](plugin/decodeBbox3DPlugin), [pillarScatterPlugin](plugin/pillarScatterPlugin), and [voxelGeneratorPlugin](plugin/voxelGeneratorPlugin)

### Changed

- Extend Megatron LayerNorm plugins to support larger hidden sizes
- Refactored EfficientNMS plugin for TFTRT and added implicit batch mode support
- Update base TensorRT version to 8.2.3.0
- GPT-2 greedy search speedup - now runs on GPU
- Updates to TensorRT developer tools
  - Polygraphy [v0.35.1](tools/Polygraphy/CHANGELOG.md#v0351-2022-01-14)
  - onnx-graphsurgeon [v0.3.15](tools/onnx-graphsurgeon/CHANGELOG.md#v0315-2022-01-18)
- Updated ONNX parser to v8.2.3.0
- Minor updates and bugfixes
  - Samples: TFOD, GPT-2, demo/BERT
  - Plugins: proposalPlugin, geluPlugin, bertQKVToContextPlugin, batchedNMS

### Removed

- Unused source file(s) in demo/BERT

## [8.2.1 GA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-2-1) - 2021-11-24

TensorRT OSS release corresponding to TensorRT 8.2.1.8 GA release.

- Updates since [TensorRT 8.2.0 EA release](https://github.com/NVIDIA/TensorRT/releases/tag/8.2.0-EA).
- Please refer to the [TensorRT 8.2.1 GA release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-2-1) for more information.

- ONNX parser [v8.2.1](https://github.com/onnx/onnx-tensorrt/releases/tag/release%2F8.2-GA)

  - Removed duplicate constant layer checks that caused some performance regressions
  - Fixed expand dynamic shape calculations
  - Added parser-side checks for `Scatter` layer support

- Sample updates

  - Added [Tensorflow Object Detection API converter samples](samples/python/tensorflow_object_detection_api), including Single Shot Detector, Faster R-CNN and Mask R-CNN models
  - Multiple enhancements in HuggingFace transformer demos
    - Added multi-batch support
    - Fixed resultant performance regression in batchsize=1
    - Fixed T5 large/T5-3B accuracy issues
    - Added [notebooks](demo/HuggingFace/notebooks) for T5 and GPT-2
    - Added CPU benchmarking option
  - Deprecated `kSTRICT_TYPES` (strict type constraints). Equivalent behaviour now achieved by setting `PREFER_PRECISION_CONSTRAINTS`, `DIRECT_IO`, and `REJECT_EMPTY_ALGORITHMS`
  - Removed `sampleMovieLens`
  - Renamed sampleReformatFreeIO to sampleIOFormats
  - Add `idleTime` option for samples to control qps
  - Specify default value for `precisionConstraints`
  - Fixed reporting of TensorRT build version in trtexec
  - Fixed `combineDescriptions` typo in trtexec/tracer.py
  - Fixed usages of of `kDIRECT_IO`

- Plugin updates

  - `EfficientNMS` plugin support extended to TF-TRT, and for clang builds.
  - Sanitize header definitions for BERT fused MHA plugin
  - Separate C++ and cu files in `splitPlugin` to avoid PTX generation (required for CUDA enhanced compatibility support)
  - Enable C++14 build for plugins

- ONNX tooling updates

  - [onnx-graphsurgeon](tools/onnx-graphsurgeon/CHANGELOG.md) upgraded to v0.3.14
  - [Polygraphy](tools/Polygraphy/CHANGELOG.md) upgraded to v0.33.2
  - [pytorch-quantization](tools/pytorch-quantization) toolkit upgraded to v2.1.2

- Build and container fixes

  - Add `SM86` target to default `GPU_ARCHS` for platforms with cuda-11.1+
  - Remove deprecated `SM_35` and add `SM_60` to default `GPU_ARCHS`
  - Skip CUB builds for cuda 11.0+ [#1455](https://github.com/NVIDIA/TensorRT/pull/1455)
  - Fixed cuda-10.2 container build failures in Ubuntu 20.04
  - Add native ARM server build container
  - Install devtoolset-8 for updated g++ version in CentOS7
  - Added a note on supporting c++14 builds for CentOS7
  - Fixed docker build for large UIDs [#1373](https://github.com/NVIDIA/TensorRT/issues/1373)
  - Updated README instructions for Jetpack builds

- demo enhancements

  - Updated Tacotron2 instructions and add CPU benchmarking
  - Fixed issues in demoBERT python notebook

- Documentation updates
  - Updated Python documentation for `add_reduce`, `add_top_k`, and `ISoftMaxLayer`
  - Renamed default GitHub branch to `main` and updated hyperlinks

## [8.2.0 EA](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#rel-8-2-0-EA) - 2021-10-05

### Added

- [Demo applications](demo/HuggingFace) showcasing TensorRT inference of [HuggingFace Transformers](https://huggingface.co/transformers).
  - Support is currently extended to GPT-2 and T5 models.
- Added support for the following ONNX operators:
  - `Einsum`
  - `IsNan`
  - `GatherND`
  - `Scatter`
  - `ScatterElements`
  - `ScatterND`
  - `Sign`
  - `Round`
- Added support for building TensorRT Python API on Windows.

### Updated

- Notable API updates in TensorRT 8.2.0.6 EA release. See [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) for details.
  - Added three new APIs, `IExecutionContext: getEnqueueEmitsProfile()`, `setEnqueueEmitsProfile()`, and `reportToProfiler()` which can be used to collect layer profiling info when the inference is launched as a CUDA graph.
  - Eliminated the global logger; each `Runtime`, `Builder` or `Refitter` now has its own logger.
  - Added new operators: `IAssertionLayer`, `IConditionLayer`, `IEinsumLayer`, `IIfConditionalBoundaryLayer`, `IIfConditionalOutputLayer`, `IIfConditionalInputLayer`, and `IScatterLayer`.
  - Added new `IGatherLayer` modes: `kELEMENT` and `kND`
  - Added new `ISliceLayer` modes: `kFILL`, `kCLAMP`, and `kREFLECT`
  - Added new `IUnaryLayer` operators: `kSIGN` and `kROUND`
  - Added new runtime class `IEngineInspector` that can be used to inspect the detailed information of an engine, including the layer parameters, the chosen tactics, the precision used, etc.
  - `ProfilingVerbosity` enums have been updated to show their functionality more explicitly.
- Updated TensorRT OSS container defaults to cuda 11.4
- CMake to target C++14 builds.
- Updated following ONNX operators:
  - `Gather` and `GatherElements` implementations to natively support negative indices
  - `Pad` layer to support ND padding, along with `edge` and `reflect` padding mode support
  - `If` layer with general performance improvements.

### Removed

- Removed `sampleMLP`.
- Several flags of trtexec have been deprecated:
  - `--explicitBatch` flag has been deprecated and has no effect. When the input model is in UFF or in Caffe prototxt format, the implicit batch dimension mode is used automatically; when the input model is in ONNX format, the explicit batch mode is used automatically.
  - `--explicitPrecision` flag has been deprecated and has no effect. When the input ONNX model contains Quantization/Dequantization nodes, TensorRT automatically uses explicit precision mode.
  - `--nvtxMode=[verbose|default|none]` has been deprecated in favor of `--profilingVerbosity=[detailed|layer_names_only|none]` to show its functionality more explicitly.

## [21.10](https://github.com/NVIDIA/TensorRT/releases/tag/21.10) - 2021-10-05

### Added

- Benchmark script for demoBERT-Megatron
- Dynamic Input Shape support for EfficientNMS plugin
- Support empty dimensions in ONNX
- INT32 and dynamic clips through elementwise in ONNX parser

### Changed

- Bump TensorRT version to 8.0.3.4
- Use static shape for only single batch single sequence input in demo/BERT
- Revert to using native FC layer in demo/BERT and FCPlugin only on older GPUs.
- Update demo/Tacotron2 for TensorRT 8.0
- Updates to TensorRT developer tools
  - Polygraphy [v0.33.0](tools/Polygraphy/CHANGELOG.md#v0330-2021-09-16)
    - Added various examples, a CLI User Guide and how-to guides.
    - Added experimental support for DLA.
    - Added a `data to-input` tool that can combine inputs/outputs created by `--save-inputs`/`--save-outputs`.
    - Added a `PluginRefRunner` which provides CPU reference implementations for TensorRT plugins
    - Made several performance improvements in the Polygraphy CUDA wrapper.
    - Removed the `to-json` tool which was used to convert Pickled data generated by Polygraphy 0.26.1 and older to JSON.
  - Bugfixes and documentation updates in pytorch-quantization toolkit.
- Bumped up package versions: tensorflow-gpu 2.5.1, pillow 8.3.2
- ONNX parser enhancements and bugfixes
  - Update ONNX submodule to v1.8.0
  - Update convDeconvMultiInput function to properly handle deconvs
  - Update RNN documentation
  - Update QDQ axis assertion
  - Fix bidirectional activation alpha and beta values
  - Fix opset10 `Resize`
  - Fix shape tensor unsqueeze
  - Mark BOOL tiles as unsupported
  - Remove unnecessary shape tensor checks

### Removed

- N/A

## [21.09](https://github.com/NVIDIA/TensorRT/releases/tag/21.09) - 2021-09-22

### Added

- Add `ONNX2TRT_VERSION` overwrite in CMake.

### Changed

- Updates to TensorRT developer tools
  - ONNX-GraphSurgeon [v0.3.12](tools/onnx-graphsurgeon/CHANGELOG.md#v0312-2021-08-24)
  - pytorch-quantization toolkit [v2.1.1](tools/pytorch-quantization)
- Fix assertion in EfficientNMSPlugin

### Removed

- N/A

## [21.08](https://github.com/NVIDIA/TensorRT/releases/tag/21.08) - 2021-08-05

### Added

- Add demoBERT and demoBERT-MT (sparsity) benchmark data for TensorRT 8.
- Added example python notebooks
  - [BERT - Q&A with TensorRT](demo/BERT/notebooks)
  - [EfficientNet - Object Detection with TensorRT](demo/EfficientDet/notebooks)

### Changed

- Updated samples and plugins directory structure
- Updates to TensorRT developer tools
  - Polygraphy [v0.31.1](tools/Polygraphy/CHANGELOG.md#v0311-2021-07-16)
  - ONNX-GraphSurgeon [v0.3.11](tools/onnx-graphsurgeon/CHANGELOG.md#v0311-2021-07-14)
  - pytorch-quantization toolkit [v2.1.1](tools/pytorch-quantization)
- README fix to update build command for native aarch64 builds.

### Removed

- N/A

## [21.07](https://github.com/NVIDIA/TensorRT/releases/tag/21.07) - 2021-07-21

Identical to the TensorRT-OSS [8.0.1](https://github.com/NVIDIA/TensorRT/releases/tag/8.0.1) Release.

## [8.0.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/#tensorrt-8) - 2021-07-02

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
- Added TensorRT [QuickStart Guide](https://github.com/NVIDIA/TensorRT/tree/main/quickstart).
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
  - Fix issues found on Jetson NANO

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

- API documentation for [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/docs)

### Changed

- Support for SM86 in [demoBERT](https://github.com/NVIDIA/TensorRT/tree/main/demo/BERT)
- Updated NGC checkpoint URLs for [demoBERT](https://github.com/NVIDIA/TensorRT/tree/main/demo/BERT) and [Tacotron2](https://github.com/NVIDIA/TensorRT/tree/main/demo/Tacotron2).

### Removed

- N/A

## [20.10](https://github.com/NVIDIA/TensorRT/releases/tag/20.10) - 2020-10-22

### Added

- [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) v0.20.13 - Deep Learning Inference Prototyping and Debugging Toolkit
- [PyTorch-Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) v2.0.0
- Updated BERT plugins for [variable sequence length inputs](https://github.com/NVIDIA/TensorRT/tree/main/demo/BERT#variable-sequence-length)
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
