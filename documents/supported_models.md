# TensorRT Supported Model List

This verified model matrix pairs with [`import_workflows.md`](./import_workflows.md). For each model family, it lists the dtype(s) used during validation.

## Scope & Reading Guide

TensorRT is a general-purpose neural-network graph execution engine, not a model zoo. In principle **any NN architecture** can run on TensorRT as long as it is expressible through the workflows described in the [Import Workflows Guide](./import_workflows.md). The [Custom Plugin](./import_workflows.md#adding-a-custom-operator--plugin) section covers the escape hatch for ops TensorRT does not yet implement natively.

The table below is **not** an exhaustive support list. It is the subset of models NVIDIA has verified and benchmarked; we publish it so you know which configurations have a known-good baseline and where the current rough edges are. If your model is not listed, the expectation is still that it works — please file an issue if it does not.

### Reading the Tables

- **Dtype** lists the precision used for the verified baseline. Other precisions may also work.
- Component-split models (diffusion pipelines, speech models with encoder/decoder) list one row per validated component.

## Table of Contents

- [LLMs / Text Generation](#llms--text-generation)
- [Encoder-only NLP (BERT family, embeddings)](#encoder-only-nlp-bert-family-embeddings)
- [Vision Classification & Embeddings](#vision-classification--embeddings)
- [Speech / Audio](#speech--audio)
- [Diffusion Models](#diffusion-models)
- [Multimodal](#multimodal)
- [Legacy / TRT Sample Models](#legacy--trt-sample-models)
- [Requesting New Model Coverage](#requesting-new-model-coverage)

---

## LLMs / Text Generation

> **Preferred path for LLM generation:** [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (KV-cache, paged attention, FP8/INT4, speculative decoding, tensor/pipeline parallelism). For production LLM serving, use TensorRT-LLM.

| Model                           | Dtype    |
|---------------------------------|----------|
| `meta-llama/Llama-3.1-8B`       | bfloat16 |
| `meta-llama/Llama-3.2-1B`       | bfloat16 |
| `Qwen/Qwen3-0.6B`               | bfloat16 |
| `deepseek-ai/Janus-Pro-7B`      | bfloat16 |

> For TensorRT-LLM's own coverage, see the [TensorRT-LLM model support matrix](https://github.com/NVIDIA/TensorRT-LLM#model-zoo).

---

## Encoder-only NLP (BERT family, embeddings)

| Model                                              | Dtype   |
|----------------------------------------------------|---------|
| `google-bert/bert-base-uncased`                    | float32 |
| `google-bert/bert-base-multilingual-cased`         | float16 |
| `FacebookAI/roberta-base`                          | float32 |
| `FacebookAI/roberta-large`                         | float32 |
| `FacebookAI/xlm-roberta-base`                      | float32 |
| `distilbert/distilbert-base-uncased`               | float32 |
| `sentence-transformers/all-MiniLM-L6-v2`           | float32 |
| `sentence-transformers/all-mpnet-base-v2`          | float32 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | float32 |
| `BAAI/bge-base-en-v1.5`                            | float32 |
| `nlpaueb/legal-bert-base-uncased`                  | float32 |

---

## Vision Classification & Embeddings

| Model                                       | Dtype   |
|---------------------------------------------|---------|
| `torchvision/resnet50`                      | float32 |
| `timm/mobilenetv3_small_100.lamb_in1k`      | float32 |
| `trpakov/vit-face-expression`               | float32 |
| `openai/clip-vit-base-patch32`              | float32 |
| `openai/clip-vit-large-patch14`             | float32 |
| `facebook/dinov2-base`                      | float32 |
| `Falconsai/nsfw_image_detection`            | float32 |
| `dima806/fairface_age_image_detection`      | float32 |

---

## Speech / Audio

| Model (Component)                                 | Dtype   |
|---------------------------------------------------|---------|
| `openai/whisper-large-v3-turbo` (Encoder)         | float32 |
| `openai/whisper-large-v3-turbo` (Decoder)         | float32 |
| `openai/whisper-large-v3` (Encoder)               | float32 |
| `openai/whisper-large-v3` (Decoder)               | float32 |
| `laion/clap-htsat-fused`                          | float32 |
| `sesame/csm-1b` (Backbone)                        | float32 |
| `neuphonic/neutts-air`                            | float32 |
| `LiquidAI/LFM2-Audio-1.5B`                        | float32 |

---

## Diffusion Models

Diffusion pipelines are evaluated per component (Text Encoder / UNet or DiT / VAE) because TRT does not ingest the pipeline object directly.

| Pipeline (Component)                                       | Dtype    |
|------------------------------------------------------------|----------|
| `stabilityai/sd-turbo`                                     | float16  |
| `stabilityai/sdxl-turbo` (UNet)                            | float16  |
| `stabilityai/sdxl-turbo` (VAE / Text Encoders)             | mixed    |
| `stabilityai/stable-diffusion-xl-base-1.0`                 | float16  |
| `CompVis/stable-diffusion-v1-4`                            | float16  |
| `stable-diffusion-v1-5/stable-diffusion-v1-5`              | float16  |
| `stabilityai/stable-diffusion-2-1`                         | float16  |
| `playgroundai/playground-v2.5-1024px-aesthetic`            | float16  |
| `dataautogpt3/ProteusV0.3`                                 | float16  |
| `black-forest-labs/FLUX.2-dev` (Text Encoder)              | bfloat16 |
| `black-forest-labs/FLUX.2-dev` (DiT)                       | bfloat16 |
| `black-forest-labs/FLUX.2-dev` (VAE)                       | float16  |
| `black-forest-labs/FLUX.1-schnell` (DiT / TextEnc / VAE)   | mixed    |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (Text Encoder)          | float16  |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (VAE)                   | float16  |
| `Qwen/Qwen-Image` (Text Encoder)                           | bfloat16 |
| `Qwen/Qwen-Image` (DiT / VAE)                              | bfloat16 |
| `stabilityai/stable-diffusion-3-medium-diffusers`          | bfloat16 |
| `stabilityai/stable-diffusion-3.5-medium` / `3.5-large`    | mixed    |
| `HiDream-ai/HiDream-I1-Full`                               | bfloat16 |
| `stabilityai/stable-video-diffusion-img2vid-xt`            | float16  |

---

## Multimodal

| Model                             | Dtype    |
|-----------------------------------|----------|
| `openai/clip-vit-base-patch32`    | float32  |
| `deepseek-ai/Janus-Pro-7B`        | bfloat16 |
| `Datadog/Toto-Open-Base-1.0`      | float32  |

---

## Legacy / TRT Sample Models

TensorRT ships hand-validated C++/Python samples for these classic architectures and workflows:

- MNIST digit classifiers, model parsing, dynamic-shape, plugin, and safe-runtime samples — see `samples/` in this repo.

---

## Requesting New Model Coverage

File a GitHub issue with:

1. The Hugging Face ID or model source URL.
2. The target dtype (fp32 / fp16 / bf16 / fp8 / int8 / int4).
3. Any framework-level working example (helps us reproduce quickly).

The maintainers will benchmark the model and extend this table — no external contributor action needed for the benchmark step.
