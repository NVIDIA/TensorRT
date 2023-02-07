# Introduction

This demo application ("demoDiffusion") showcases the acceleration of [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) pipeline using TensorRT plugins.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/8.5 --single-branch
cd TensorRT
git submodule update --init --recursive
```

### Launch TensorRT NGC container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/tensorrt:22.10-py3 /bin/bash
```

### (Optional) Install latest TensorRT release

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorrt
```
> NOTE: Alternatively, you can download and install TensorRT packages from [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt).

### Build TensorRT plugins library

Build TensorRT Plugins library using the [TensorRT OSS build instructions](https://github.com/NVIDIA/TensorRT/blob/main/README.md#building-tensorrt-oss).

```bash
export TRT_OSSPATH=/workspace

cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so"
```

### Install required packages

```bash
cd $TRT_OSSPATH/demo/Diffusion
pip3 install -r requirements.txt

# Create output directories
mkdir -p onnx engine output
```

> NOTE: demoDiffusion has been tested on systems with NVIDIA A100, RTX3090, and RTX4090 GPUs, and the following software configuration.
```
cuda-python         11.8.1
diffusers           0.7.2
onnx                1.12.0
onnx-graphsurgeon   0.3.25
onnxruntime         1.13.1
polygraphy          0.43.1
tensorrt            8.5.1.7
tokenizers          0.13.2
torch               1.12.0+cu116
transformers        4.24.0
```

> NOTE: optionally install HuggingFace [accelerate](https://pypi.org/project/accelerate/) package for faster and less memory-intense model loading.


# Running demoDiffusion

### Review usage instructions

```bash
python3 demo-diffusion.py --help
```

### HuggingFace user access token

To download the model checkpoints for the Stable Diffusion pipeline, you will need a `read` access token. See [instructions](https://huggingface.co/docs/hub/security-tokens).

```bash
export HF_TOKEN=<your access token>
```

### Generate an image guided by a single text prompt

```bash
LD_PRELOAD=${PLUGIN_LIBS} python3 demo-diffusion.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN -v
```

# Restrictions
- Upto 16 simultaneous prompts (maximum batch size) per inference.
- For generating images of dynamic shapes without rebuilding the engines, use `--force-dynamic-shape`.
- Supports images sizes between 256x256 and 1024x1024.
