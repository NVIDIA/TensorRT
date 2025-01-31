# Introduction

This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion and ControlNet pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/10.8 --single-branch
cd TensorRT
```

### Launch NVIDIA pytorch container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:24.10-py3 /bin/bash
```

NOTE: The demo supports CUDA>=11.8

### Install latest TensorRT release

```bash
python3 -m pip install --upgrade pip
pip install --pre tensorrt-cu12
```

Check your installed version using:
`python3 -c 'import tensorrt;print(tensorrt.__version__)'`

> NOTE: Alternatively, you can download and install TensorRT packages from [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt).

### Install required packages

```bash
export TRT_OSSPATH=/workspace
cd $TRT_OSSPATH/demo/Diffusion
pip3 install -r requirements.txt
```

> NOTE: demoDiffusion has been tested on systems with NVIDIA H100, A100, L40, T4, and RTX4090 GPUs, and the following software configuration.
```
diffusers           0.31.0
onnx                1.15.0
onnx-graphsurgeon   0.5.2
onnxruntime         1.16.3
polygraphy          0.49.9
tensorrt            10.8.0.31
tokenizers          0.13.3
torch               2.2.0
transformers        4.42.2
controlnet-aux      0.0.6
nvidia-modelopt     0.15.1
```

# Running demoDiffusion

### Review usage instructions for the supported pipelines

```bash
python3 demo_txt2img.py --help
python3 demo_img2img.py --help
python3 demo_inpaint.py --help
python3 demo_controlnet.py --help
python3 demo_txt2img_xl.py --help
python3 demo_txt2img_flux.py --help
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

### Faster Text-to-image using SD1.5 or SD2.1 INT8 & FP8 quantization using ModelOpt

Run the below command to generate an image with SD1.5 or SD2.1 in INT8

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --int8
```

Run the below command to generate an image with SD1.5 or SD2.1 in FP8. (FP8 is only supported on Hopper and Ada.)

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --fp8
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
python3 demo_txt2img_xl.py "Picture of a rustic Italian village with Olive trees and mountains" --version=xl-1.0 --lora-path "ostris/crayon_style_lora_sdxl" "ostris/watercolor_style_lora_sdxl" --lora-weight 0.3 0.7 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --build-enable-refit
```

### Faster Text-to-image using SDXL INT8 & FP8 quantization using ModelOpt

Run the below command to generate an image with Stable Diffusion XL in INT8

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --int8
```

Run the below command to generate an image with Stable Diffusion XL in FP8. (FP8 is only supported on Hopper and Ada.)

```bash
python3 demo_txt2img_xl.py "a photo of an astronaut riding a horse on mars" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --fp8
```

> Note that INT8 & FP8 quantization is only supported for SDXL, SD1.5, SD2.1 and SD2.1-base, and won't work with LoRA weights. FP8 quantization is only supported on Hopper and Ada. Some prompts may produce better inputs with fewer denoising steps (e.g. `--denoising-steps 20`) but this will repeat the calibration, ONNX export, and engine building processes for the U-Net.

For step-by-step tutorials to run INT8 & FP8 inference on stable diffusion models, please refer to examples in [TensorRT ModelOpt diffusers sample](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/diffusers).

### Faster Text-to-Image using SDXL + LCM (Latent Consistency Model) LoRA weights

[LCM-LoRA](https://arxiv.org/abs/2311.05556) produces good quality images in 4 to 8 denoising steps instead of 30+ needed base model. Note that we use LCM scheduler and disable classifier-free-guidance by setting `--guidance-scale` to 0.
LoRA weights are fused into the ONNX and finalized TensorRT plan files in this example.
```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-1.0 --lora-path "latent-consistency/lcm-lora-sdxl" --lora-weight 1.0 --onnx-dir onnx-sdxl-lcm-nocfg --engine-dir engine-sdxl-lcm-nocfg --denoising-steps 4 --scheduler LCM --guidance-scale 0.0
```
### Faster Text-to-Image using SDXL Turbo
Even faster image generation than LCM, producing coherent images in just 1 step. Note: SDXL Turbo works best for 512x512 resolution, EulerA scheduler and classifier-free-guidance disabled.
```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-turbo --onnx-dir onnx-sdxl-turbo --engine-dir engine-sdxl-turbo --denoising-steps 1 --scheduler EulerA --guidance-scale 0.0 --width 512 --height 512
```

### Generate an image guided by a text prompt using Stable Diffusion 3

Run the command below to generate an image using Stable Diffusion 3

```bash
python3 demo_txt2img_sd3.py "A vibrant street wall covered in colorful graffiti, the centerpiece spells \"SD3 MEDIUM\", in a storm of colors" --version sd3 --hf-token=$HF_TOKEN
```

You can also specify an input image conditioning as shown below

```bash
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png -O dog-on-bench.png

python3 demo_txt2img_sd3.py "dog wearing a sweater and a blue collar" --version sd3 --input-image dog-on-bench.png --hf-token=$HF_TOKEN
```

Note that a denosing-percentage is applied to the number of denoising-steps when an input image conditioning is provided. Its default value is set to 0.6. This parameter can be updated using `--denoising-percentage`

### Generate a video guided by an initial image using Stable Video Diffusion

Download the pre-exported ONNX model

```bash
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1-tensorrt onnx-svd-xt-1-1
cd onnx-svd-xt-1-1 && git lfs pull && cd ..
```

SVD-XT-1.1 (25 frames at resolution 576x1024)
```bash
python3 demo_img2vid.py --version svd-xt-1.1 --onnx-dir onnx-svd-xt-1-1 --engine-dir engine-svd-xt-1-1 --hf-token=$HF_TOKEN
```


Run the command below to generate a video in FP8.

```bash
python3 demo_img2vid.py --version svd-xt-1.1 --onnx-dir onnx-svd-xt-1-1 --engine-dir engine-svd-xt-1-1 --hf-token=$HF_TOKEN --fp8
```

> NOTE: There is a bug in HuggingFace, you can workaround with following this [PR](https://github.com/huggingface/diffusers/pull/6562/files)

```
if torch.is_tensor(num_frames):
    num_frames = num_frames.item()
emb = emb.repeat_interleave(num_frames, dim=0)
```

You may also specify a custom conditioning image using `--input-image`:
```bash
python3 demo_img2vid.py --version svd-xt-1.1 --onnx-dir onnx-svd-xt-1-1 --engine-dir engine-svd-xt-1-1 --input-image https://www.hdcarwallpapers.com/walls/2018_chevrolet_camaro_zl1_nascar_race_car_2-HD.jpg --hf-token=$HF_TOKEN
```

NOTE: The min and max guidance scales are configured using --min-guidance-scale and --max-guidance-scale respectively.

### Generate an image guided by a text prompt using Stable Cascade

Run the below command to generate an image using Stable Cascade
```bash
python3 demo_stable_cascade.py --onnx-opset=16 "Anthropomorphic cat dressed as a pilot" --onnx-dir onnx-sc --engine-dir engine-sc
```

The lite versions of the models are also supported using the command below
```bash
python3 demo_stable_cascade.py --onnx-opset=16 "Anthropomorphic cat dressed as a pilot" --onnx-dir onnx-sc-lite --engine-dir engine-sc-lite --lite
```

> NOTE: The pipeline is only enabled for the BF16 model weights

> NOTE: The pipeline only supports ONNX export using Opset 16.

> NOTE: The denoising steps and guidance scale for the Prior and Decoder models are configured using --prior-denoising-steps, --prior-guidance-scale, --decoder-denoising-steps, and --decoder-guidance-scale respectively.

### Generate an image guided by a text prompt using Flux

Run the below command to generate an image with FLUX.1 Dev in FP16.

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN
```

Run the below command to generate an image with FLUX.1 Dev in BF16.

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --bf16
```

Run the below command to generate an image with FLUX.1 Dev in FP8. (FP8 is only supported on Hopper and Ada.)

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --fp8
```

Run the below command to generate an image with FLUX.1 Schnell in FP16.

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell"
```

Run the below command to generate an image with FLUX.1 Schnell in BF16.

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --bf16
```

Run the below command to generate an image with FLUX.1 Schnell in FP8. (FP8 is only supported on Hopper and Ada.)

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --fp8
```

NOTE: Running the FLUX.1 Dev or FLUX.1 Schnell pipeline requires 48GB or 24GB of GPU memory or higher, respectively.

### Generate an image guided by an initial image and a text prompt using Flux

```bash
wget "https://miro.medium.com/v2/resize:fit:640/format:webp/1*iD8mUonHMgnlP0qrSx3qPg.png" -O yellow.png

python3 demo_img2img_flux.py "A home with 2 floors and windows. The front door is purple" --hf-token=$HF_TOKEN --input-image yellow.png --image-strength 0.95 --bf16
```

### Generate an image guided by a text prompt and a control image using Flux ControlNet

Download the control image below
```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png
```

Run the Depth ControlNet pipeline

```bash
python3 demo_img2img_flux.py "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts." --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --bf16 --denoising-steps 30 --onnx-dir onnx_depth/ --engine-dir engine_depth/
```

Run the Canny ControlNet pipeline

```bash
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --bf16 --onnx-dir onnx_canny/ --engine-dir engine_canny/
```

### Generate an image guided by a text prompt and a control image using Flux ControlNet in FP8

FP8 ControlNet pipelines require downloading a calibration dataset and providing the path. You can use the datasets provided by Black Forest Labs here: [depth](https://drive.google.com/file/d/1DFfhOSrTlKfvBFLcD2vAALwwH4jSGdGk/view) | [canny](https://drive.google.com/file/d/1dRoxOL-vy3tSAesyqBSJoUWsbkMwv3en/view)

You can use the `--calibraton-dataset` flag to specify the path, which is set to `./{depth/canny}-eval/benchmark` by default if not provided. Note that the dataset should have `inputs/` and `prompts/` underneath the provided path, matching the format of the BFL dataset.

Run the Depth ControlNet pipeline in FP8. (FP8 is only supported on Hopper and Ada.)

```bash
python3 demo_img2img_flux.py "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts." --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --fp8 --denoising-steps 30 --onnx-dir onnx_depth/ --engine-dir engine_depth/
```

Run the Canny ControlNet pipeline in FP8 using a custom dataset path. (FP8 is only supported on Hopper and Ada.)

```bash
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --fp8 --onnx-dir onnx_canny/ --engine-dir engine_canny/ --calibration-dataset {custom/dataset/path}
```

### Exporting ONNX models without running TRT build or inference.
This is useful for exporting ONNX files using a device with a larger vram and building TRT engines on another device.
```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --onnx-export-only
```

### Use separate directories for individual ONNX models
The directories specified in `--model-onnx-dirs` will override the directory set in `--onnx-dir`. Unspecified models will continue to use the directory set in `--onnx-dir`.
Suppose the model storage locations are as following:
* transformer model ONNX files are saved at `./onnx_folder_1/transformer` and `./onnx_folder_1/transformer.opt`. 
* vae model ONNX files are saved in `./onnx_folder_2/vae` and `./onnx_folder_2/vae.opt`.
* Other models (t5 and clip) are still under `./onnx/`.

The corresponding command to run the pipeline:
```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --onnx-dir=onnx --model-onnx-dirs=transformer:onnx_folder_1,vae:onnx_folder_2
```

## Configuration options
- Noise scheduler can be set using `--scheduler <scheduler>`. Note: not all schedulers are available for every version.
- To accelerate engine building time use `--timing-cache <path to cache file>`. The cache file will be created if it does not already exist. Note that performance may degrade if cache files are used across multiple GPU targets. It is recommended to use timing caches only during development. To achieve the best perfromance in deployment, please build engines without timing cache.
- Specify new directories for storing onnx and engine files when switching between versions, LoRAs, ControlNets, etc. This can be done using `--onnx-dir <new onnx dir>` and `--engine-dir <new engine dir>`.
- Inference performance can be improved by enabling [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) using `--use-cuda-graph`. Enabling CUDA graphs requires fixed input shapes, so this flag must be combined with `--build-static-batch` and cannot be combined with `--build-dynamic-shape`.


