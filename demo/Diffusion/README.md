# Introduction

This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion and ControlNet pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/9.0 --single-branch
cd TensorRT
```

### Launch TensorRT NGC container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.06-py3 /bin/bash
```

### Install latest TensorRT release

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorrt
```

Minimum required version is TensorRT 8.6.0. Check your installed version using:
`python3 -c 'import tensorrt;print(tensorrt.__version__)'`

> NOTE: Alternatively, you can download and install TensorRT packages from [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt).

### Install required packages

```bash
export TRT_OSSPATH=/workspace
cd $TRT_OSSPATH/demo/Diffusion
pip3 install -r requirements.txt

```

> NOTE: demoDiffusion has been tested on systems with NVIDIA A100, RTX3090, and RTX4090 GPUs, and the following software configuration.
```
diffusers           0.14.0
onnx                1.13.1
onnx-graphsurgeon   0.3.26
onnxruntime         1.14.1
polygraphy          0.47.1
tensorrt            8.6.1.6
tokenizers          0.13.2
torch               2.1.0
transformers        4.26.1
controlnet-aux      0.0.6
```

> NOTE: optionally install HuggingFace [accelerate](https://pypi.org/project/accelerate/) package for faster and less memory-intense model loading.


# Running demoDiffusion

### Review usage instructions for the supported pipelines

```bash
python3 demo_txt2img.py --help
python3 demo_img2img.py --help
python3 demo_inpaint.py --help
python3 demo_controlnet.py --help
```

### HuggingFace user access token

To download the model checkpoints for the Stable Diffusion pipeline, you will need a `read` access token. See [instructions](https://huggingface.co/docs/hub/security-tokens).

```bash
export HF_TOKEN=<your access token>
```

### Generate an image with Stable Diffusion guided by a single text prompt

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN
```

### Generate an image guided by an image and single text prompt

```bash
python3 demo_img2img.py "photorealistic new zealand hills" --hf-token=$HF_TOKEN
```

Use `--input-image=<path to image(s)>` to specify your image. Otherwise the example image will be downloaded from the Internet.

### Generate an inpainted image guided by an image, mask and single text prompt

```bash

python3 demo_inpaint.py "a mecha robot sitting on a bench" --hf-token=$HF_TOKEN --version=1.5 --onnx-dir=onnx-1.5 --engine-dir=engine-1.5
```

Use `--input-image=<path to image(s)>` and `--mask-image=<path to mask>` to specify your inputs. They must have the same dimensions. Otherwise the example image and mask will be downloaded from the Internet.

### Generate an image with ControlNet guided by image(s) and text prompt(s)

```bash

python3 demo_controlnet.py "bag" --hf-token=$HF_TOKEN --version=1.5 --denoising-steps 20 --controlnet-type scribble
```


Use `--input-image=<path to image(s)>` to specify your image inputs. If not specified, the image will be downloaded and preprocessed depending on the `--controlnet-type` input. If the images are provided using the `--input-image` argument, they are expected to be preprocessed for the appropriate controlnet type.
The ControlNet type can be specified using the `--controlnet-type` argument. Choose from `canny depth hed mlsd normal openpose scribble seg`. 

#### ControlNet Samples:

<img src="https://drive.google.com/uc?export=view&id=17ub3MVSQHp26ty-wioNX6iQQ-nAveYSV" alt= “” width="800" height="400">


#### Combining Multiple Conditionings

Multiple ControlNet types can also be specified to combine the conditionings. While specifying multiple conditionings, controlnet scales should also be provided. The scales signify the importance of each conditioning in relation with the other. For example, to condition using `openpose` and `canny` with scales of 1.0 and 0.8 respectively, the arguments provided would be `--controlnet-type openpose canny` and `--controlnet-scale 1.0 0.8`. Note that the number of controlnet scales provided should match the number of controlnet types.

### Generate an image with Stable Diffusion XL guided by a single text prompt

Run the below command to generate an image with Stable Diffusion XL

```bash
python3 demo_txt2img_xl.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version=xl-1.0
```


### Universal input arguments
- One can set schdeuler using `--scheduler=EulerA`. Note that some schedulers are not available for some pipelines or version.
- To accelerate engine building time one can use `--timing-cache=<path to cache file>`. This cache file will be created if does not exist. Note, that it may influence the performance if the cache file created on the other hardware is used. It is suggested to use this flag only during development. To achieve the best perfromance during deployment, please, build engines without timing cache.
- To switch between versions or pipelines one needs either to clear onnx and engine dirs, or to specify `--force-onnx-export --force-onnx-optimize --force-engine-build` or to create new dirs and to specify `--onnx-dir=<new onnx dir> --engine-dir=<new engine dir>`.
- Inference performance can be improved by enabling [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) using `--use-cuda-graph`. Enabling CUDA graphs requires fixed input shapes, so this flag must be combined with `--build-static-batch` and cannot be combined with `--build-dynamic-shape`.
