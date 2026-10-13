---
name: Report a TensorRT issue
about: The more information you share, the more feedback we can provide.
title: 'XXX failure of TensorRT X.Y when running XXX on GPU XXX'
labels: ''
assignees: ''

---

## Description

<!--
  A clear and concise description of the issue.

  For example: I tried to run model ABC on GPU, but it fails with the error below (share a 2-3 line error log).
-->


## Environment

<!-- Please share any setup information you know. This will help us to understand and address your case. -->

**TensorRT Version**:

**NVIDIA GPU**:

**NVIDIA Driver Version**:

**CUDA Version**:

**CUDNN Version**:


Operating System:

Python Version (if applicable):

Tensorflow Version (if applicable):

PyTorch Version (if applicable):

Baremetal or Container (if so, version):


## Relevant Files

<!-- Please include links to any models, data, files, or scripts necessary to reproduce your issue. (Github repo, Google Drive/Dropbox, etc.) -->

**Model link**:


## Steps To Reproduce

<!--
  Craft a minimal bug report following this guide - https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports

  Please include:
  * Exact steps/commands to build your repro
  * Exact steps/commands to run your repro
  * Full traceback of errors encountered
-->

**Commands or scripts**:

**Have you tried [the latest release](https://developer.nvidia.com/tensorrt)?**:

**Attach the captured .json and .bin files from [TensorRT's API Capture tool](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capture-replay.html) if you're on an x86_64 Unix system**

**Can this model run on other frameworks?** For example run ONNX model with ONNXRuntime (`polygraphy run <model.onnx> --onnxrt`):
