# Introduction

This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/8.6 --single-branch
cd TensorRT
```

### Launch TensorRT NGC container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.02-py3 /bin/bash
```

### (Optional) Install latest TensorRT release

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorrt
```
> NOTE: Alternatively, you can download and install TensorRT packages from [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt).

### Install required packages

```bash
export TRT_OSSPATH=/workspace
cd $TRT_OSSPATH/demo/Diffusion
pip3 install -r requirements.txt

# Create output directories
mkdir -p onnx engine output
```

> NOTE: demoDiffusion has been tested on systems with NVIDIA A100, RTX3090, and RTX4090 GPUs, and the following software configuration.
```
diffusers           0.14.0
onnx                1.13.1
onnx-graphsurgeon   0.3.26
onnxruntime         1.14.1
polygraphy          0.44.2
tensorrt            8.6.0.10
tokenizers          0.13.2
torch               1.13.0
transformers        4.26.1
```

> NOTE: optionally install HuggingFace [accelerate](https://pypi.org/project/accelerate/) package for faster and less memory-intense model loading.


# Running demoDiffusion

### Review usage instructions for the supported pipelines

```bash
python3 demo_txt2img.py --help
python3 demo_img2img.py --help
python3 demo_inpainting.py --help
```

### HuggingFace user access token

To download the model checkpoints for the Stable Diffusion pipeline, you will need a `read` access token. See [instructions](https://huggingface.co/docs/hub/security-tokens).

```bash
export HF_TOKEN=<your access token>
```

### Generate an image guided by a single text prompt

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN -v
```

### Generate an image guided by an image and single text prompt

```bash
python3 demo_img2img.py "photorealistic new zealand hills" --hf-token=$HF_TOKEN -v
```

Use `--input-image=<path to image>` to specify your image. Otherwise the example image will be downloaded from the Internet.

### Generate an inpainted image guided by an image, mask and single text prompt

```bash
python3 demo_inpaint.py "a mecha robot sitting on a bench" --hf-token=$HF_TOKEN -v
```

Use `--input-image=<path to image>` and `--mask-image=<path to mask>` to specify your inputs. They must have the same dimensions. Otherwise the example image and mask will be downloaded from the Internet.

### Input arguments
- One can set schdeuler using `--scheduler=EulerA`. Note that some schedulers are not available for some pipelines or version.
- To accelerate engine building time one can use `--timing-cache=<path to cache file>`. This cache file will be created if does not exist. Note, that it may influence the performance if the cache file created on the other hardware is used. It is suggested to use this flag only during development. To achieve the best perfromance during deployment, please, build engines without timing cache.
- To switch between versions or pipelines one needs either to clear onnx and engine dirs, or to specify `--force-onnx-export --force-onnx-optimize --force-engine-build` or to create new dirs and to specify `--onnx-dir=<new onnx dir> --engine-dir=<new engine dir>`.

