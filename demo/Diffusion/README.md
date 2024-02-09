# Introduction

This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion and ControlNet pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/9.3 --single-branch
cd TensorRT
```

### Launch NVIDIA pytorch container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.12-py3 /bin/bash
```

### Install latest TensorRT release

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt
```

> NOTE: TensorRT 9.x is only available as a pre-release

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
diffusers           0.23.1
onnx                1.14.0
onnx-graphsurgeon   0.3.26
onnxruntime         1.15.1
polygraphy          0.49.7
tensorrt            9.3.0.1
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

### Text-to-Image using SD 1.5

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom"
```

### Image-to-Image using SD 1.5

```bash
wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg -O sketch-mountains-input.jpg

python3 demo_img2img.py "A fantasy landscape, trending on artstation" --input-image sketch-mountains-input.jpg
```

### Inpainting
Generate an inpainted image guided by an image, mask and text prompt.

```bash
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png -O dog-on-bench.png
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png -O dog-mask.png

python3 demo_inpaint.py "a mecha robot sitting on a bench" --input-image dog-on-bench.png --mask-image dog-mask.png
```

> NOTE: inpainting is only supported in versions `1.5` and `2.0`.

### Text-to-Image using SD 1.5 + ControlNet
Generate an image guided by conditioning control input(s) and text prompt.

```bash
python3 demo_controlnet.py "Stormtrooper's lecture in beautiful lecture hall" --controlnet-type depth --denoising-steps 20 --onnx-dir onnx-cnet-depth --engine-dir engine-cnet-depth
```

> NOTE: `--input-image` must be a pre-processed image corresponding to `--controlnet-type`. If unspecified, a sample image will be downloaded. Supported controlnet types include: `canny`, `depth`, `hed`, `mlsd`, `normal`, `openpose`, `scribble`, and `seg`.

Examples:
<img src="https://drive.google.com/uc?export=view&id=17ub3MVSQHp26ty-wioNX6iQQ-nAveYSV" alt= “” width="800" height="400">

#### Combining multiple conditionings

Multiple ControlNet types can also be specified to combine the conditionings. While specifying multiple conditionings, controlnet scales should also be provided. The scales signify the importance of each conditioning in relation with the other. For example, to condition using `openpose` and `canny` with scales of 1.0 and 0.8 respectively, the arguments provided would be `--controlnet-type openpose canny` and `--controlnet-scale 1.0 0.8`. Note that the number of controlnet scales provided should match the number of controlnet types.


### Text-to-image using SDXL (Stable Diffusion XL)

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl
```

The optional refiner model may be enabled by specifying `--enable-refiner` and separate directories for storing refiner onnx and engine files using `--onnx-refiner-dir` and `--engine-refiner-dir` respectively.

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --enable-refiner --onnx-refiner-dir onnx-sdxl-refiner --engine-refiner-dir engine-sdxl-refiner
```

### Text-to-image using refittable SDXL + LoRA weights
Since we are building a refittable engine in this example, LoRA weights are not fused into the plan file and can be refitted on-the-fly during inference.
```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-1.0 --lora-path "ostris/embroidery_style_lora_sdxl" --lora-scale 1.0 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --build-enable-refit
```

It is also possible to combine multiple LoRAs.
```bash
python3 demo_txt2img_xl.py "Picture of a rustic Italian village with Olive trees and mountains" --version=xl-1.0 --lora-path "ostris/crayon_style_lora_sdxl" "ostris/watercolor_style_lora_sdxl" --lora-scale 0.3 0.7 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --build-enable-refit
```

### Faster Text-to-image using SDXL & INT8 quantization using AMMO

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --int8 --quantization-level 3
```

Note that the calibration process can be quite time-consuming, and will be repeated if `--quantization-level`, `--denoising-steps`, or `--onnx-dir` is changed.

### Faster Text-to-Image using SDXL + LCM (Latent Consistency Model) LoRA weights
[LCM-LoRA](https://arxiv.org/abs/2311.05556) produces good quality images in 4 to 8 denoising steps instead of 30+ needed base model. Note that we use LCM scheduler and disable classifier-free-guidance by setting `--guidance-scale` to 0.
LoRA weights are fused into the ONNX and finalized TensorRT plan files in this example.
```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-1.0 --lora-path "latent-consistency/lcm-lora-sdxl" --lora-scale 1.0 --onnx-dir onnx-sdxl-lcm-nocfg --engine-dir engine-sdxl-lcm-nocfg --denoising-steps 4 --scheduler LCM --guidance-scale 0.0
```
### Faster Text-to-Image using SDXL Turbo
Even faster image generation than LCM, producing coherent images in just 1 step. Note: SDXL Turbo works best for 512x512 resolution, EulerA scheduler and classifier-free-guidance disabled.
```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-turbo --onnx-dir onnx-sdxl-turbo --engine-dir engine-sdxl-turbo --denoising-steps 1 --scheduler EulerA --guidance-scale 0.0 --width 512 --height 512
```

## Configuration options
- Noise scheduler can be set using `--scheduler <scheduler>`. Note: not all schedulers are available for every version.
- To accelerate engine building time use `--timing-cache <path to cache file>`. The cache file will be created if it does not already exist. Note that performance may degrade if cache files are used across multiple GPU targets. It is recommended to use timing caches only during development. To achieve the best perfromance in deployment, please build engines without timing cache.
- Specify new directories for storing onnx and engine files when switching between versions, LoRAs, ControlNets, etc. This can be done using `--onnx-dir <new onnx dir>` and `--engine-dir <new engine dir>`.
- Inference performance can be improved by enabling [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) using `--use-cuda-graph`. Enabling CUDA graphs requires fixed input shapes, so this flag must be combined with `--build-static-batch` and cannot be combined with `--build-dynamic-shape`.

