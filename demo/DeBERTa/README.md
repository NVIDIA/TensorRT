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
A performance gap has been observed between Google's [BERT](https://arxiv.org/abs/1810.04805) design and Microsoft's [DeBERTa](https://arxiv.org/abs/2006.03654) design. The main reason of the gap is the disentangled attention design in DeBERTa triples the attention computation over BERT's regular attention. In addition to the extra matrix multiplications, the disentangled attention design also involves indirect memory accesses during the gather operations. In this regard, a [TensorRT plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin/disentangledAttentionPlugin) has been implemented to optimize DeBERTa's disentangled attention module, which is built-in since TensorRT 8.4 GA release.

This DeBERTa demo includes code and scripts for (i) exporting ONNX model from PyTorch, (ii) modifying ONNX model by inserting the plugin nodes, (iii) model inference options with TensorRT `trtexec` executable, TensorRT Python API, or ONNX Runtime with TensorRT execution provider, and (iv) measuring the correctness and performance of the optimized model.

The demo works with the [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta_v2) of DeBERTa.

## Performance Benchmark
TBD - Add some performance data here to show the plugin optimization

## Environment Setup
It is recommended to use docker for reproducing the following steps. Follow the setup steps in TensorRT OSS [README](https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment) to build and launch the container and build OSS:

**Example: Ubuntu 20.04 on x86-64 with cuda-11.6.2 (default)**
```bash
# Download this TensorRT OSS repo
git clone -b master https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive

## at root of TensorRT OSS
# build container
./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda11.6

# launch container
./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda11.6 --gpus all

## now inside container
# build OSS
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

## Step 1: PyTorch model to ONNX model
```bash
python deberta_pytorch2onnx.py --filename ./test/deberta.onnx
```

This will export the DeBERTa model from HuggingFace's DeBERTa-V2 implementation into ONNX format, with user given file name. Optionally, specify DeBERTa variant to export, such as `--variant microsoft/deberta-v3-xsmall`.

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

For `trtexec` test, it is recommended to used the [TensorRT NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt) where the executable is built-in. The DeBERTa plugin support is available since docker image `22.08-py3`, which contains TensorRT 8.4 GA pre-installed. For earlier images before `22.08`, skip this `trtexec` test.

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

Correctness check requires intermediate outputs from the model, thus it is necessary to modify the ONNX graph and add intermediate output nodes. The correctness check was added at the location of plugin outputs in each layer. The metric is average and maximum of the element-wise absolute error. Note that for FP16 precision with 10 significance bits, absolute error in the order of 1e-2 and 1e-3 is expected, and for FP32 precision with 23 significance bits, 1e-6 to 1e-7 is expected. Refer to [Machine Epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) for details. 

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
