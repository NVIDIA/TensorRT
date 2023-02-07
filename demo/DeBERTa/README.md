# DeBERTa Model Inference with TensorRT Disentangled Attention Optimizations

- [DeBERTa Model Inference with TensorRT Disentangled Attention Optimizations](#deberta-model-inference-with-tensorrt-disentangled-attention-optimizations)
  - [Background](#background)
  - [Performance Benchmark](#performance-benchmark)
  - [Environment Setup](#environment-setup)
  - [Step 1: PyTorch model to ONNX model](#step-1-pytorch-model-to-onnx-model)
  - [Step 2: Modify ONNX model for TensorRT engine building](#step-2-modify-onnx-model-for-tensorrt-engine-building)
  - [Step 3: Model inference with TensorRT (using Python TensorRT API or `trtexec`)](#step-3-model-inference-with-tensorrt-using-python-tensorrt-api-or-trtexec)
  - [Optional Step: Correctness check of model with and without plugin](#optional-step-correctness-check-of-model-with-and-without-plugin)
  - [Optional Step: Model inference with ONNX Runtime and TensorRT Execution Provider (Python API)](#optional-step-model-inference-with-onnx-runtime-and-tensorrt-execution-provider-python-api)

***

## Background
A performance gap has been observed between Google's [BERT](https://arxiv.org/abs/1810.04805) design and Microsoft's [DeBERTa](https://arxiv.org/abs/2006.03654) design. The main reason of the gap is the disentangled attention design in DeBERTa triples the attention computation over BERT's regular attention. In addition to the extra matrix multiplications, the disentangled attention design also involves indirect memory accesses during the gather operations. In this regard, a [TensorRT plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin/disentangledAttentionPlugin) has been implemented to optimize DeBERTa's disentangled attention module, which is built-in since TensorRT 8.4 GA Update 2 (8.4.3) release.

This DeBERTa demo includes code and scripts for (i) exporting ONNX model from PyTorch, (ii) modifying ONNX model by inserting the plugin nodes, (iii) model inference options with TensorRT `trtexec` executable, TensorRT Python API, or ONNX Runtime with TensorRT execution provider, and (iv) measuring the correctness and performance of the optimized model.

The demo works with the [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta_v2) of DeBERTa.

## Performance Benchmark
Experiments of inference performance are measured on NVIDIA A100-80GB, T4, and V100-16GB, using TensorRT 8.4 GA Update 2 (8.4.3) + CUDA 11.6 and TensorRT 8.2 GA Update 3 (8.2.4) + CUDA 11.4. The application-based model configuration is: sequence length = 512/1024/2048, hidden size = 384, intermediate size = 1536, number of layers = 12, number of attention heads = 6, maximum relative distance = 256, vocabulary size = 128K, batch size = 1, with randomly initialized weights. Numbers are average latency of 100 inference runs. Speedup numbers are fastest TensorRT number over the PyTorch baseline.

| Sequence Length | Model Latency (ms)             |           | TensorRT 8.4 GA Update 2 (8.4.3), CUDA 11.6 |           |           | TensorRT 8.2 GA Update 3 (8.2.4), CUDA 11.4 |           |
| :-------------- | :----------------------------- | :-------: | :-----------------------------------------: | :-------: | :-------: | :-----------------------------------------: | :-------: |
|                 |                                | A100-80GB |                     T4                      | V100-16GB | A100-80GB |                     T4                      | V100-16GB |
| 512             | PyTorch (FP32/TF32)            |   23.7    |                    35.6                     |   53.1    |   24.6    |                    34.0                     |   53.1    |
|                 | PyTorch (FP16)                 |   21.5    |                    22.7                     |   43.5    |   22.6    |                    20.6                     |   43.5    |
|                 | TensorRT (FP32/TF32)           |    3.9    |                    12.4                     |    7.2    |    4.3    |                    12.3                     |    5.6    |
|                 | TensorRT (FP16)                |  **1.6**  |                     6.2                     |  **3.9**  |  **1.8**  |                     6.1                     |  **3.1**  |
|                 | TensorRT w/ plugin (FP32/TF32) |    4.3    |                    12.9                     |    6.9    |    4.8    |                    13.1                     |    6.9    |
|                 | TensorRT w/ plugin (FP16)      |    1.9    |                   **5.6**                   |    4.0    |    2.1    |                   **6.0**                   |    4.2    |
|                 | **Speedup**                    | **14.8**  |                   **6.4**                   | **13.6** | **13.7**  |                   **5.7**                   | **17.1**  |
|                 |                                |           |                                             |           |           |                                             |           |
| 1024            | PyTorch (FP32/TF32)            |   35.7    |                    82.8                     |   65.4    |   35.8    |                    83.7                     |   65.4    |
|                 | PyTorch (FP16)                 |   35.1    |                    53.8                     |   59.4    |   35.6    |                    52.9                     |   59.4    |
|                 | TensorRT (FP32/TF32)           |    8.3    |                    31.3                     |   15.7    |    8.8    |                    34.7                     |   12.8    |
|                 | TensorRT (FP16)                |    3.8    |                    16.3                     |    8.3    |    3.7    |                    18.4                     |    7.4    |
|                 | TensorRT w/ plugin (FP32/TF32) |    7.9    |                    31.1                     |   14.3    |    9.0    |                    32.8                     |   13.5    |
|                 | TensorRT w/ plugin (FP16)      |  **2.8**  |                  **12.4**                   |  **7.3**  |  **3.3**  |                  **13.6**                   |  **6.8**  |
|                 | **Speedup**                    | **12.8**  |                   **6.7**                   |  **9.0**  | **10.8**  |                   **6.2**                   |  **9.6**  |
|                 |                                |           |                                             |           |           |                                             |           |
| 2048            | PyTorch (FP32/TF32)            |   84.8    |                    261.3                    |   236.3   |   84.8    |                    263.1                    |   236.3   |
|                 | PyTorch (FP16)                 |   79.4    |                    178.2                    |   205.5   |   76.0    |                    181.8                    |   205.5   |
|                 | TensorRT (FP32/TF32)           |   22.2    |                    109.1                    |   39.1    |   22.6    |                    109.8                    |   35.0    |
|                 | TensorRT (FP16)                |   10.2    |                    56.2                     |   23.5    |   10.8    |                    62.9                     |   21.0    |
|                 | TensorRT w/ plugin (FP32/TF32) |   20.4    |                    96.7                     |   38.6    |   21.2    |                    95.7                     |   35.6    |
|                 | TensorRT w/ plugin (FP16)      |  **7.6**  |                  **44.1**                   | **21.0**  |  **8.3**  |                  **44.6**                   | **18.1**  |
|                 | **Speedup**                    | **11.2**  |                   **5.9**                   | **11.3**  | **10.2**  |                   **5.9**                   | **13.1**  |

In addition, a pre-trained model `microsoft/deberta-v3-xsmall` was tested, which configuration is similar to sequencen length = 512 model above. And `microsoft/deberta-v3-large` (sequence length = 512) performance on TensorRT 8.4 GA Update 2 (8.4.3) is also added from recent experiments.

| Variant             | Model Latency (ms)             |           | TensorRT 8.4 GA Update 2 (8.4.3), CUDA 11.6 |           |           | TensorRT 8.2 GA Update 3 (8.2.4), CUDA 11.4 |           |
| :------------------ | :----------------------------- | :-------: | :-----------------------------------------: | :-------: | :-------: | :-----------------------------------------: | :-------: |
|                     |                                | A100-80GB |                     T4                      | V100-16GB | A100-80GB |                     T4                      | V100-16GB |
| `deberta-v3-xsmall` | PyTorch (FP32/TF32)            |   30.1    |                    40.6                     |   57.1    |   28.9    |                    39.2                     |   57.1    |
|                     | PyTorch (FP16)                 |   27.6    |                    26.3                     |   47.9    |   26.3    |                    24.8                     |   47.9    |
|                     | TensorRT (FP32/TF32)           |    4.1    |                    12.4                     |    7.6    |    4.4    |                    12.8                     |    5.8    |
|                     | TensorRT (FP16)                |    1.8    |                     6.2                     |  **3.7**  |  **1.9**  |                     6.4                     |  **3.1**  |
|                     | TensorRT w/ plugin (FP32/TF32) |    4.3    |                    12.9                     |    7.7    |    4.8    |                    13.6                     |    6.9    |
|                     | TensorRT w/ plugin (FP16)      |  **1.8**  |                   **5.6**                   |    4.7    |    2.1    |                   **6.0**                   |    4.1    |
|                     | **Speedup**                    | **16.7**  |                   **7.3**                   | **15.4**  | **15.2**  |                   **6.5**                   | **18.4**  |
|                     |                                |           |                                             |           |           |                                             |           |
| `deberta-v3-large`  | PyTorch (FP32/TF32)            |   51.6    |                    40.6                     |   100.0   |     -     |                      -                      |     -     |
|                     | PyTorch (FP16)                 |   52.6    |                    26.3                     |   96.5    |     -     |                      -                      |     -     |
|                     | TensorRT (FP32/TF32)           |   31.1    |                    32.3                     |   43.2    |     -     |                      -                      |     -     |
|                     | TensorRT (FP16)                |    7.8    |                    16.0                     | **13.9**  |     -     |                      -                      |     -     |
|                     | TensorRT w/ plugin (FP32/TF32) |   30.9    |                    32.8                     |   44.9    |     -     |                      -                      |     -     |
|                     | TensorRT w/ plugin (FP16)      |  **7.3**  |                  **13.5**                   |   14.4    |     -     |                      -                      |     -     |
|                     | **Speedup**                    |  **7.1**  |                   **3.0**                   |  **7.2**  |     -     |                      -                      |     -     |

Note that the performance gap between BERT's self-attention and DeBERTa's disentangled attention mainly comes from the additional `Gather` and `Transpose` operations in the attention design, and such gap becomes most significant when the maximum input sequence length is long (e.g., 2048). The fastest inference times are labeled as bold in the table above. In summary, for short sequence length applications, regular TensorRT inference might be sufficient, while for longer sequence length applications, the plugin optimizations are preferred and can be utilized to further improve the inference latency. Also, to get maximum speedup, using FP16 precision for inference is recommended.

## Environment Setup
It is recommended to use docker for reproducing the following steps. Follow the setup steps in TensorRT OSS [README](https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment) to build and launch the container and build OSS:

**Example: Ubuntu 20.04 on x86-64 with cuda-11.6.2 (default)**
```bash
# Download this TensorRT OSS repo
git clone -b main https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive

## at root of TensorRT OSS
# build container
./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda11.6

# launch container
./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda11.6 --gpus all

## now inside container
# build OSS (only required for pre-8.4.3 TensorRT versions)
cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)

# polygraphy bin location & trtexec bin location
export PATH="~/.local/bin":"${TRT_OSSPATH}/build/out":$PATH 

# navigate to the demo folder install additional dependencies (note PyTorch 1.11.0 is recommended for onnx to export properly)
cd $TRT_OSSPATH/demo/DeBERTa
pip install -r requirements.txt
```
> NOTE:
1. `sudo` password for Ubuntu build containers is 'nvidia'.
2. The DeBERTa plugin is only built-in after TensorRT 8.4 GA Update 2 (8.4.3) release. For pre-8.4.3 versions, you need to build TensorRT OSS from source and link the shared libraries with TensorRT build.
3. For ONNX Runtime deployment, the associated changes for the plugin are only built-in after 1.12 release. For pre-1.12 versions, you need to [build ONNX Runtime from source with TensorRT execution provider](https://onnxruntime.ai/docs/build/eps.html#tensorrt).
4. TensorRT OSS docker container is designed for use cases when you need to build OSS repository from source. After TensorRT 8.4 GA Update 2 (8.4.3) release, the most convenient way is to use the corresponding `22.08-py3` docker image at [TensorRT NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt) when it is released, without the need to build from dockerfile. But for now, please follow the instructions above to use the OSS docker image.

## Step 1: PyTorch model to ONNX model
```bash
python deberta_pytorch2onnx.py --filename ./test/deberta.onnx [--variant microsoft/deberta-v3-xsmall] [--seq-len 2048]
```

This will export the DeBERTa model from HuggingFace's DeBERTa-V2 implementation into ONNX format, with user given file name. Optionally, specify DeBERTa variant to export, such as `--variant microsoft/deberta-v3-xsmall`, or specify the maximum sequence length configuration for testing, such as `--seq-len 2048`. Models specified by `--variant` are with pre-trained weights, while models specified by `--seq-len` are with randomly initialized weights. Note that `--variant` and `--seq-len` cannot be used together because pre-trained models have pre-defined max sequence length.

## Step 2: Modify ONNX model for TensorRT engine building
```bash
python deberta_onnx_modify.py ./test/deberta.onnx # generates original TRT-compatible model, `*_original.onnx`

python deberta_onnx_modify.py ./test/deberta.onnx --plugin # generates TRT-compatible model with plugin nodes, `*_plugin.onnx`
```

The original HuggingFace implementation has uint8 Cast operations that TensorRT doesn't support, which needs to be removed from the ONNX model. After this step, the ONNX model can run in TensorRT. Without passing any flags, the script will save the model with Cast nodes removed, by default named with `_original` suffix.

Further, to use the DeBERTa plugin optimizations, the disentangled attention subgraph needs to be replaced by node named `DisentangledAttention_TRT`. By passing `--plugin` flag, the script will save the model with Cast nodes removed and plugin nodes replaced, by default named with `_plugin` suffix.

The benefits of the DeBERTa plugin optimizations can be demonstrated by comparing the latency of original model and plugin model.

## Step 3: Model inference with TensorRT (using Python TensorRT API or `trtexec`)
```bash
# build and test the original DeBERTa model (baseline)
python deberta_tensorrt_inference.py --onnx=./test/deberta_original.onnx --build fp16 --test fp16

# build and test the optimized DeBERTa model with plugin
python deberta_tensorrt_inference.py --onnx=./test/deberta_plugin.onnx --build fp16 --test fp16
```

This will build and test the original and optimized DeBERTa models. `--build` to build the engine from ONNX model, and `--test` to measure the optimized latency. TensorRT engine of different precisions (`--fp32/--tf32/--fp16`) can be built. Engine files are saved as `**/[Model name]_[GPU name]_[Precision].engine`.

> NOTE:
1. To get maximum speedup, using `--fp16` is recommended. Also, it was observed the plugin optimizations demonstrate more speedup when the input sequence length is long, such as 2048.
2. TF32 is only effective on Ampere and later devices. 
3. TensorRT optimization profile in `deberta_tensorrt_inference.py` is set by default for batch size of 1 usage. For batch usage, the optional and maximum profile should be changed.
4. TensorRT engines are specific to the exact GPU device they were built on, as well as the platform and the TensorRT version. On the same machine, building is needed only once and the engine can be used for repeated testing.

For `trtexec` test, it is recommended to used the [TensorRT NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt) where the executable is built-in. The DeBERTa plugin support is available since docker image `22.08-py3`, which contains TensorRT 8.4 GA Update 2 (8.4.3) pre-installed. For earlier images before `22.08`, skip this `trtexec` test.

`trtexec` command for sequence length = 2048 model is given as an example:
```bash
trtexec --onnx=./test/deberta_plugin.onnx --workspace=4096 --explicitBatch --optShapes=input_ids:1x2048,attention_mask:1x2048 --iterations=10 --warmUp=10 --noDataTransfers --fp16
```

## Optional Step: Correctness check of model with and without plugin
```bash
# prepare the ONNX models with intermediate output nodes (this will save two new onnx models with suffix `*_correctness_check_original.onnx` and `*_correctness_check_plugin.onnx`)
python deberta_onnx_modify.py ./test/deberta.onnx --correctness-check

# build the ONNX models with intermediate outputs for comparison
python deberta_tensorrt_inference.py --onnx=./test/deberta_correctness_check_original.onnx --build fp16
python deberta_tensorrt_inference.py --onnx=./test/deberta_correctness_check_plugin.onnx --build fp16

# run correctness check (specify the root model name with --onnx)
python deberta_tensorrt_inference.py --onnx=./test/deberta --correctness-check fp16
```

Correctness check requires intermediate outputs from the model, thus it is necessary to modify the ONNX graph and add intermediate output nodes. The correctness check was added at the location of plugin outputs in each layer. The metrics are average and maximum of the element-wise absolute error. Note that for FP16 precision with 10 significance bits, absolute error in the order of 1e-2 and 1e-3 is expected, and for FP32 precision with 23 significance bits, 1e-6 to 1e-7 is expected. Refer to [Machine Epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) for details. 

Example output for FP16 correctness check between original DeBERTa model and plugin optimized DeBERTa model:
```bash
[Layer 0 Element-wise Check] Avgerage absolute error: 2.152580e-03, Maximum absolute error: 3.010559e-02. 1e-2~1e-3 expected for FP16 (10 significance bits) and 1e-6~1e-7 expected for FP32 (23 significance bits)
...
...
[Layer 12 Element-wise Check] Avgerage absolute error: 6.198883e-05, Maximum absolute error: 1.220703e-04. 1e-2~1e-3 expected for FP16 (10 significance bits) and 1e-6~1e-7 expected for FP32 (23 significance bits)
```

## Optional Step: Model inference with ONNX Runtime and TensorRT Execution Provider (Python API)
```bash
python deberta_ort_inference.py --onnx=./test/deberta_original.onnx --test fp16

python deberta_ort_inference.py --onnx=./test/deberta_plugin.onnx --test fp16

python deberta_ort_inference.py --onnx=./test/deberta --correctness-check fp16 # for correctness check
```

In addition to TensorRT inference, [ONNX Runtime](https://github.com/microsoft/onnxruntime) with TensorRT Execution Provider can also be used as the inference framework. The DeBERTa TensorRT plugin is officially supported in onnxruntime since version 1.12. For earlier releases of onnxruntime, skip this step.

The results can be cross-validated between TensorRT and onnxruntime. For example, the [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy) tool can be used for easy comparison:

```bash
polygraphy run ./test/deberta_original.onnx --trt --onnxrt --workspace=4000000000
```
