# Introduction

This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion and ControlNet pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/9.1 --single-branch
cd TensorRT
```

### Launch NVIDIA pytorch container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.07-py3 /bin/bash
```

### Install latest TensorRT release

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt
```

> NOTE: TensorRT 9.0 is only available as a pre-release

Check your installed version using:
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
diffusers           0.19.3
onnx                1.14.0
onnx-graphsurgeon   0.3.26
onnxruntime         1.15.1
polygraphy          0.47.1
tensorrt            9.1.0.4
tokenizers          0.13.2
torch               2.1.0
transformers        4.31.0
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
python3 demo_txt2img_xl.py --help
```

### HuggingFace user access token

To download model checkpoints for the Stable Diffusion pipelines, obtain a `read` access token to HuggingFace Hub. See [instructions](https://huggingface.co/docs/hub/security-tokens).

```bash
export HF_TOKEN=<your access token>
```

### Generate an image guided by a text prompt

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN
```

### Generate an image guided by an initial image and a text prompt

```bash
wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg -O sketch-mountains-input.jpg

python3 demo_img2img.py "A fantasy landscape, trending on artstation" --hf-token=$HF_TOKEN --input-image=sketch-mountains-input.jpg
```

### Generate an inpainted image guided by an image, mask and a text prompt

```bash
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png -O dog-on-bench.png
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png -O dog-mask.png

python3 demo_inpaint.py "a mecha robot sitting on a bench" --hf-token=$HF_TOKEN --input-image=dog-on-bench.png --mask-image=dog-mask.png
```

> NOTE: inpainting is only supported in versions `1.5` and `2.0`.

### Generate an image with ControlNet guided by image(s) and text prompt(s)

```bash
python3 demo_controlnet.py "Stormtrooper's lecture in beautiful lecture hall" --controlnet-type depth --hf-token=$HF_TOKEN --denoising-steps 20 --onnx-dir=onnx-cnet-depth --engine-dir=engine-cnet-depth
```

> NOTE: `--input-image` must be a pre-processed image corresponding to `--controlnet-type`. If unspecified, a sample image will be downloaded. Supported controlnet types include: `canny`, `depth`, `hed`, `mlsd`, `normal`, `openpose`, `scribble`, and `seg`.

Examples:
<img src="https://drive.google.com/uc?export=view&id=17ub3MVSQHp26ty-wioNX6iQQ-nAveYSV" alt= “” width="800" height="400">

#### Combining multiple conditionings

Multiple ControlNet types can also be specified to combine the conditionings. While specifying multiple conditionings, controlnet scales should also be provided. The scales signify the importance of each conditioning in relation with the other. For example, to condition using `openpose` and `canny` with scales of 1.0 and 0.8 respectively, the arguments provided would be `--controlnet-type openpose canny` and `--controlnet-scale 1.0 0.8`. Note that the number of controlnet scales provided should match the number of controlnet types.


### Generate an image with Stable Diffusion XL guided by a single text prompt

Run the below command to generate an image with Stable Diffusion XL

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --hf-token=$HF_TOKEN --version=xl-1.0
```

The optional refiner model may be enabled by specifying `--enable-refiner` and separate directories for storing refiner onnx and engine files using `--onnx-refiner-dir` and `--engine-refiner-dir` respectively.

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --hf-token=$HF_TOKEN --version=xl-1.0 --enable-refiner --onnx-refiner-dir=onnx-refiner --engine-refiner-dir=engine-refiner
```

### Generate an image guided by a text prompt, and using specified LoRA model weight updates

```bash
python3 demo_txt2img.py "A pokemon with green eyes and red legs." --hf-token=$HF_TOKEN --lora-weights="sayakpaul/sd-model-finetuned-lora-t4" --lora-scale=1.0
```

### Common usage options
- Noise scheduler can be set using `--scheduler=<scheduler>`. Note that some schedulers are not available for certain pipelines or SD versions.
- To accelerate engine building time use `--timing-cache=<path to cache file>`. The cache file will be created if it does not already exist. Note that performance may degrade if cache files are used across multiple GPU targets. It is recommended to use timing caches only during development. To achieve the best perfromance in deployment, please build engines without timing cache.
- To switch between versions or pipelines one needs either to clear onnx and engine dirs, or to specify `--force-onnx-export --force-onnx-optimize --force-engine-build` or to create new dirs and to specify `--onnx-dir=<new onnx dir> --engine-dir=<new engine dir>`.
- Inference performance can be improved by enabling [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) using `--use-cuda-graph`. Enabling CUDA graphs requires fixed input shapes, so this flag must be combined with `--build-static-batch` and cannot be combined with `--build-dynamic-shape`.



