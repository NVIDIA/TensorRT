# Introduction

This demo application ("demoDiffusion") showcases the acceleration of Stable Diffusion and ControlNet pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/10.14 --single-branch
cd TensorRT
```

### Launch NVIDIA pytorch container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:25.08-py3 /bin/bash
```

NOTE: The demo supports CUDA>=12.0

### Install the required packages

To install dependencies for modern pipelines (SDXL and later):

```bash
source setup.sh
```

To install dependencies for legacy pipelines (SD 1.5/2.1):

```bash
REQUIREMENTS_FILE=requirements_legacy.txt source setup.sh
```

Check your installed version using:
`python3 -c 'import tensorrt;print(tensorrt.__version__)'`

> NOTE: Alternatively, you can download and install TensorRT packages from [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt).


> NOTE: demoDiffusion has been tested on systems with NVIDIA H100, A100, L40, T4, and RTX4090 GPUs, and the following software configuration.

```
diffusers           0.35.0
onnx                1.18.0
onnx-graphsurgeon   0.5.2
onnxruntime         1.19.2
polygraphy          0.49.22
tensorrt            10.14.1.48
tokenizers          0.13.3
torch               2.8.0a0+5228986c39.nv25.6
transformers        4.52.4
controlnet-aux      0.0.6
nvidia-modelopt     0.31.0
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

### Generate an image with Stable Diffusion XL with ControlNet guided by an image and a text prompt

```bash
python3 demo_controlnet.py "A beautiful bird with rainbow colors" --controlnet-type canny --hf-token=$HF_TOKEN --denoising-steps 20 --onnx-dir=onnx-cnet --engine-dir=engine-cnet --version xl-1.0
```

> NOTE: Currently only `--controlnet-type canny` is supported. `--input-image` must be a pre-processed image corresponding to `--controlnet-type canny`. If unspecified, a sample image will be downloaded.

> NOTE: FP8 quantization (`--fp8`) is supported.

### Generate an image guided by a text prompt, and using specified LoRA model weight updates

```bash
# FP16
python3 demo_txt2img_xl.py "Picture of a rustic Italian village with Olive trees and mountains" --version=xl-1.0 --lora-path "ostris/crayon_style_lora_sdxl" "ostris/watercolor_style_lora_sdxl" --lora-weight 0.3 0.7 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --build-enable-refit

# FP8
python3 demo_txt2img_xl.py "Picture of a rustic Italian village with Olive trees and mountains" --version=xl-1.0 --lora-path "ostris/crayon_style_lora_sdxl" "ostris/watercolor_style_lora_sdxl" --lora-weight 0.3 0.7 --onnx-dir onnx-sdxl-lora --engine-dir engine-sdxl-lora --fp8
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

### Generate an image guided by a text prompt using Stable Diffusion 3 and its variants

Run the command below to generate an image using Stable Diffusion 3 and Stable Diffusion 3.5

```bash
# Stable Diffusion 3
python3 demo_txt2img_sd3.py "A vibrant street wall covered in colorful graffiti, the centerpiece spells \"SD3 MEDIUM\", in a storm of colors" --version sd3 --hf-token=$HF_TOKEN

# Stable Diffusion 3.5-medium
python3 demo_txt2img_sd35.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version=3.5-medium --denoising-steps=30 --guidance-scale 3.5 --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# Stable Diffusion 3.5-large
python3 demo_txt2img_sd35.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version=3.5-large --denoising-steps=30 --guidance-scale 3.5 --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# Stable Diffusion 3.5-large FP8
python3 demo_txt2img_sd35.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version=3.5-large --denoising-steps=30 --guidance-scale 3.5 --hf-token=$HF_TOKEN --fp8 --download-onnx-models --onnx-dir onnx_35_fp8/ --engine-dir engine_35_fp8/
```

You can also specify an input image conditioning as shown below

```bash
wget https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png -O dog-on-bench.png

# Stable Diffusion 3
python3 demo_txt2img_sd3.py "dog wearing a sweater and a blue collar" --version sd3 --input-image dog-on-bench.png --hf-token=$HF_TOKEN
```

Note that a denosing-percentage is applied to the number of denoising-steps when an input image conditioning is provided. Its default value is set to 0.6. This parameter can be updated using `--denoising-percentage`

### Generate an image with Stable Diffusion v3.5-large with ControlNet guided by an image and a text prompt

```bash
# Depth BF16
python3 demo_controlnet_sd35.py "a photo of a man" --controlnet-type depth --hf-token=$HF_TOKEN --denoising-steps 40 --guidance-scale 4.5 --bf16 --download-onnx-models

# Depth FP8
python3 demo_controlnet_sd35.py "a photo of a man" --version=3.5-large --fp8 --controlnet-type depth --download-onnx-models --denoising-steps=40 --guidance-scale 4.5 --hf-token=$HF_TOKEN

# Canny BF16
python3 demo_controlnet_sd35.py "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms" --controlnet-type canny --hf-token=$HF_TOKEN --denoising-steps 60 --guidance-scale 3.5 --bf16 --download-onnx-models

# Canny FP8
python3 demo_controlnet_sd35.py "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms" --version=3.5-large --fp8 --controlnet-type canny --hf-token=$HF_TOKEN --denoising-steps 60 --guidance-scale 3.5 --download-onnx-models

# Blur
python3 demo_controlnet_sd35.py "generated ai art, a tiny, lost rubber ducky in an action shot close-up, surfing the humongous waves, inside the tube, in the style of Kelly Slater" --controlnet-type blur --hf-token=$HF_TOKEN --denoising-steps 60 --guidance-scale 3.5 --bf16 --download-onnx-models
```

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

### Generating Images with Flux

#### 1. Generate an Image from a Text Prompt

##### Run Flux.1-Dev

NOTE: Pass `--download-onnx-models` to avoid native ONNX export and download the ONNX models from [Black Forest Labs' collection](https://huggingface.co/collections/black-forest-labs/flux1-onnx-679d06b7579583bd84c8ef83). It is only supported for BF16, FP8, and FP4 pipelines.

```bash
# FP16 (requires >48GB VRAM for native export)
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN

# BF16
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# FP8
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --quantization-level 4 --fp8 --download-onnx-models

# FP4
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --fp4 --download-onnx-models
```

##### Run Flux.1-Schnell

```bash
# FP16 (requires >48GB VRAM for native export)
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell"

# BF16
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --bf16 --download-onnx-models

# FP8
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --quantization-level 4 --fp8 --download-onnx-models

# FP4
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version="flux.1-schnell" --fp4 --download-onnx-models
```

---

#### 2. Generate an Image from an Initial Image + Text Prompt

Download an example input image:

```bash
wget "https://miro.medium.com/v2/resize:fit:640/format:webp/1*iD8mUonHMgnlP0qrSx3qPg.png" -O yellow.png
```

Run the image-to-image pipeline:

```bash
python3 demo_img2img_flux.py "A home with 2 floors and windows. The front door is purple" --hf-token=$HF_TOKEN --input-image yellow.png --image-strength 0.95 --bf16 --onnx-dir onnx-flux-dev/bf16 --engine-dir engine-flux-dev/
```

---

#### 3. Generate an Image Using Flux ControlNet

##### Download the Control Image

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png
```

##### Calibration Data for native ONNX export (FP8 Pipeline)

FP8 ControlNet pipelines require downloading a calibration dataset and providing the path. You can use the datasets provided by Black Forest Labs here: [depth](https://drive.google.com/file/d/1DFfhOSrTlKfvBFLcD2vAALwwH4jSGdGk/view) | [canny](https://drive.google.com/file/d/1dRoxOL-vy3tSAesyqBSJoUWsbkMwv3en/view)

You can use the `--calibraton-dataset` flag to specify the path, which is set to `./{depth/canny}-eval/benchmark` by default if not provided. Note that the dataset should have `inputs/` and `prompts/` underneath the provided path, matching the format of the BFL dataset.

##### Depth ControlNet

```bash
# BF16
python3 demo_img2img_flux.py "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts." --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --bf16 --denoising-steps 30  --download-onnx-models

# FP8 using pre-exported ONNX models
python3 demo_img2img_flux.py "A robot made of exotic candies" --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --fp8 --denoising-steps 30 --download-onnx-models --build-static-batch --quantization-level 4

# FP8 using native ONNX export
rm -rf onnx/* engine/* && python3 demo_img2img_flux.py "A robot made of exotic candies" --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --quantization-level 4 --fp8 --denoising-steps 30

# FP4
python3 demo_img2img_flux.py "A robot made of exotic candies" --version="flux.1-dev-depth" --hf-token=$HF_TOKEN --guidance-scale 10 --control-image robot.png --fp4 --denoising-steps 30 --download-onnx-models --build-static-batch
```

##### Canny ControlNet

```bash
# BF16
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --bf16 --denoising-steps 30 --download-onnx-models

# FP8 using pre-exported ONNX models
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --fp8 --denoising-steps 30 --download-onnx-models --build-static-batch --quantization-level 4

# FP8 using native ONNX export
rm -rf onnx/* engine/* && python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --quantization-level 4 --fp8 --denoising-steps 30 --calibration-dataset {custom/dataset/path}

# FP4
python3 demo_img2img_flux.py "a robot made out of gold" --version="flux.1-dev-canny" --hf-token=$HF_TOKEN --guidance-scale 30 --control-image robot.png --fp4 --denoising-steps 30 --download-onnx-models --build-static-batch
```

#### 4. Generate an Image Using Flux LoRA

FLUX supports loading LoRA for Flux.1-Dev and Flux.1-Schnell. Make sure the target lora is compatible with the transformer model. Below is an example of using a [water color Flux LoRA](https://huggingface.co/SebastianBodza/flux_lora_aquarel_watercolor)

```bash
# FP16
python3 demo_txt2img_flux.py "A painting of a barista creating an intricate latte art design, with the 'Coffee Creations' logo skillfully formed within the latte foam. In a watercolor style, AQUACOLTOK. White background." --hf-token=$HF_TOKEN --lora-path "SebastianBodza/flux_lora_aquarel_watercolor" --lora-weight 1.0 --onnx-dir=onnx-flux-lora --engine-dir=engine-flux-lora

# FP8
python3 demo_txt2img_flux.py "A painting of a barista creating an intricate latte art design, with the 'Coffee Creations' logo skillfully formed within the latte foam. In a watercolor style, AQUACOLTOK. White background." --hf-token=$HF_TOKEN --lora-path "SebastianBodza/flux_lora_aquarel_watercolor" --lora-weight 1.0 --onnx-dir=onnx-flux-lora --engine-dir=engine-flux-lora --fp8
```

#### 5. Edit an Image using Flux Kontext

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png

# BF16
python3 demo_img2img_flux.py "Add a hat to the cat" --version="flux.1-kontext-dev" --hf-token=$HF_TOKEN --guidance-scale 2.5 --kontext-image cat.png --denoising-steps 28 --bf16 --onnx-dir onnx-kontext --engine-dir engine-kontext --download-onnx-models

# FP8
python3 demo_img2img_flux.py "Add a hat to the cat" --version="flux.1-kontext-dev" --hf-token=$HF_TOKEN --guidance-scale 2.5 --kontext-image cat.png --denoising-steps 28 --fp8 --onnx-dir onnx-kontext-fp8 --engine-dir engine-kontext-fp8 --download-onnx-models --quantization-level 4

# FP4
python3 demo_img2img_flux.py "Add a hat to the cat" --version="flux.1-kontext-dev" --hf-token=$HF_TOKEN --guidance-scale 2.5 --kontext-image cat.png --denoising-steps 28 --fp4 --onnx-dir onnx-kontext-fp4 --engine-dir engine-kontext-fp4 --download-onnx-models
```
---

#### 5. Export ONNX Models Only (Skip Inference)

Use the `--onnx-export-only` flag to export ONNX models on a higher-VRAM device. The exported ONNX models can be used on a device with lower VRAM for the engine build and inference steps.

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --onnx-export-only
```

---

#### 6. Running Flux on GPUs with Limited Memory

##### Optimization Flags

- `--low-vram`: Enables model-offloading for reduced VRAM usage.
- `--ws`: Enables weight streaming in TensorRT engines.
- `--t5-ws-percentage` and `--transformer-ws-percentage`: Set runtime weight streaming budgets.
- `--build-static-batch`: Build all engines using static batch sizes to lower the required activation memory. This will limit supported batch size of these engines for inference to the value specified by `--batch-size`.

##### FLUX VRAM Requirements Table

Memory usage captured below excludes the ONNX export step, and assumes use of the `--build-static-batch` flag to reduce activation VRAM usage. Users can either use [pre-exported ONNX models](README.md#download-pre-exported-models-recommended-for-48gb-vram) or export the models separately on a higher-VRAM device using [--onnx-export-only](README.md#4-export-onnx-models-only-skip-inference).

| Precision | Default VRAM Usage | With `--low-vram` |
| --------- | ------------------ | ----------------- |
| FP16      | 39.3 GB            | 23.9 GB           |
| BF16      | 35.7 GB            | 23.9 GB           |
| FP8       | 24.6 GB            | 14.9 GB           |
| FP4       | 21.67 GB           | 11.1 GB           |

NOTE: The FP8 and FP4 Pipelines are supported on Hopper/Ada/Blackwell devices only. The FP4 pipeline is most performant on Blackwell devices.


### Run Cosmos2 World Foundation Models

Select the prompts and export them as below

```bash
export PROMPT="A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."

export NEGATIVE_PROMPT="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
```

#### 1. Generate an Image from a Text Prompt

##### Run Cosmos-Predict2-2B-Text2Image

```bash
# BF16
python3 demo_txt2image_cosmos.py "$PROMPT" --negative-prompt="$NEGATIVE_PROMPT" --hf-token=$HF_TOKEN
```

#### 2. Generate a Video guided by an Initial Video Conditioning and a Text Prompt

##### Run Cosmos-Predict2-2B-Video2World (only PyTorch backend enabled)

```bash
# BF16
python3 demo_vid2world_cosmos.py "$PROMPT" --negative-prompt="$NEGATIVE_PROMPT" --hf-token=$HF_TOKEN
```


### Specify Custom Paths for ONNX models and TensorRT engines (FLUX, Stable Diffusion 3.5 and Cosmos only)

Custom override paths to pre-exported ONNX model files can be provided using `--custom-onnx-paths`. These ONNX models are directly used to build TRT engines without further optimization on the ONNX graphs. Paths should be a comma-separated list of <model_name>:<path> pairs. For example: `--custom-onnx-paths=transformer:/path/to/transformer.onnx,vae:/path/to/vae.onnx`. Call <PipelineClass>.get_model_names(...) for the list of supported model names.

Custom override paths to pre-built engine files can be provided using `--custom-engine-paths`. Paths should be a comma-separated list of <model_name>:<path> pairs. For example: `--custom-onnx-paths=transformer:/path/to/transformer.plan,vae:/path/to/vae.plan`.

## Configuration options

- Noise scheduler can be set using `--scheduler <scheduler>`. Note: not all schedulers are available for every version.
- To accelerate engine building time use `--timing-cache <path to cache file>`. The cache file will be created if it does not already exist. Note that performance may degrade if cache files are used across multiple GPU targets. It is recommended to use timing caches only during development. To achieve the best perfromance in deployment, please build engines without timing cache.
- Specify new directories for storing onnx and engine files when switching between versions, LoRAs, ControlNets, etc. This can be done using `--onnx-dir <new onnx dir>` and `--engine-dir <new engine dir>`.
- Inference performance can be improved by enabling [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) using `--use-cuda-graph`. Enabling CUDA graphs requires fixed input shapes, so this flag must be combined with `--build-static-batch` and cannot be combined with `--build-dynamic-shape`.

