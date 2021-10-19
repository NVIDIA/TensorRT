# Tacotron 2 and WaveGlow Inference with TensorRT

The Tacotron2 and WaveGlow models form a text-to-speech (TTS) system that enables users to synthesize natural sounding speech from raw transcripts without any additional information such as patterns and/or rhythms of speech. This is an implementation of Tacotron2 for PyTorch, tested and maintained by NVIDIA, and provides scripts to perform high-performance inference using NVIDIA TensorRT. More information about the TTS system and its training can be found in the
[NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2).

NVIDIA TensorRT is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. After optimizing the compute-intensive acoustic model with NVIDIA TensorRT, inference throughput increased by up to 1.4x over native PyTorch in mixed  precision.

### Software Versions

|Software|Version|
|--------|-------|
|Python|3.6.9|
|CUDA|11.4.2|
|Apex|0.1|
|TensorRT|8.2.0.6|
|PyTorch|1.9.1|


## Quick Start Guide

1. Build and launch the container as described in [TensorRT OSS README](https://github.com/NVIDIA/TensorRT/blob/master/README.md).

    **Note:** After this point, all commands should be run from within the container.

2. Verify TensorRT installation by printing the version:
    ```bash
    python3 -c "import tensorrt as trt; print(trt.__version__)"
    ```

3. Install prerequisite software for TTS sample:
    ```bash
    cd $TRT_OSSPATH/demo/Tacotron2
    bash ./scripts/install_prerequisites.sh
    ```
4. Download pretrained checkpoints from [NGC](https://ngc.nvidia.com/catalog/models) into the `./checkpoints` directory:

- [Tacotron2 checkpoint](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_fp16)
- [WaveGlow checkpoint](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16)

    ```bash
    bash ./scripts/download_checkpoints.sh
    ```

5. Export the models to ONNX intermediate representation (ONNX IR).
   Export Tacotron 2 to three ONNX parts: Encoder, Decoder, and Postnet:

	```bash
	mkdir -p output
	python3 tensorrt/convert_tacotron22onnx.py --tacotron2 checkpoints/tacotron2_pyt_ckpt_amp_v19.09.0/nvidia_tacotron2pyt_fp16_20190427 -o output/ --fp16
	```

    Convert WaveGlow to ONNX IR:

	```bash
	python3 tensorrt/convert_waveglow2onnx.py --waveglow ./checkpoints/waveglow_ckpt_amp_256_v19.10.0/nvidia_waveglow256pyt_fp16 --config-file config.json --wn-channels 256 -o output/ --fp16
    ```

	The above commands store the generated ONNX files under the `./output/` directory:
    `encoder.onnx`, `decoder_iter.onnx`, `postnet.onnx`, `waveglow.onnx`, and `decoder.onnx` (on TensorRT 8.0+ if `--no-loop` option is not specified).

6. Export the ONNX IRs to TensorRT engines with fp16 mode enabled:

	```bash
	python3 tensorrt/convert_onnx2trt.py --encoder output/encoder.onnx --decoder output/decoder.onnx --postnet output/postnet.onnx --waveglow output/waveglow.onnx -o output/ --fp16
	```

	After running the command, there should be four new engine files in `./output/` directory:
    `encoder_fp16.engine`, `decoder_with_outer_loop_fp16.engine`, `postnet_fp16.engine`, and `waveglow_fp16.engine`. On TensorRT <8.0 or if `--no-loop` option is specified, `decoder_iter_fp16.engine` is generated instead.

7. Run TTS inference pipeline with fp16:

	
	```bash
	python3 tensorrt/inference_trt.py -i phrases/phrase.txt --encoder output/encoder_fp16.engine --decoder output/decoder_with_outer_loop_fp16.engine --postnet output/postnet_fp16.engine --waveglow output/waveglow_fp16.engine -o output/ --fp16
	```

    On TensorRT <8.0 use `decoder_iter_fp16.engine` for the decoder instead.

## Performance

### Benchmarking

The following section shows how to benchmark the TensorRT inference performance for our Tacotron2 + Waveglow TTS.

#### TensorRT inference benchmark

Before running the benchmark script, please download the checkpoints and build the TensorRT engines for the Tacotron2 and Waveglow models as prescribed in the [Quick Start Guide](#quick-start-guide) above.

The inference benchmark is performed on a single GPU by the `inference_benchmark.sh` script, which runs 3 warm-up iterations then runs timed inference for 1000 iterations.

```bash
bash scripts/inference_benchmark.sh
```

*Note*: For benchmarking we use WaveGlow with 256 residual channels, and Tacotron2 decoder with outer loop for TensorRT inference.

### Results

> Note: Results last updated for TensorRT 8.0.1.6 release.

#### Inference performance: NVIDIA T4 (16GB)

|Framework|Batch size|Input length|Precision|Avg latency (s)|Latency std (s)|Latency confidence interval 90% (s)|Latency confidence interval 95% (s)|Latency confidence interval 99% (s)|Throughput (samples/sec)|Speed-up PyT+TRT/TRT|Avg mels generated (81 mels=1 sec of speech)| Avg audio length (s)| Avg RTF|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PyT+TRT|1| 128| FP16| 0.1662 | 0.0036 | 0.1705 | 0.1717 | 0.1736 | 871,568 | 7.64 | 566 | 6.99 | 42.03 |
|PyT    |1| 128| FP16| 1.27 | 0.07 | 1.36 | 1.38 | 1.44 |  121,184 | 1.00 | 601 | 7.42 | 5.84 |

#### Inference performance: NVIDIA V100 (16GB)

|Framework|Batch size|Input length|Precision|Avg latency (s)|Latency std (s)|Latency confidence interval 90% (s)|Latency confidence interval 95% (s)|Latency confidence interval 99% (s)|Throughput (samples/sec)|Speed-up PyT+TRT/TRT|Avg mels generated (81 mels=1 sec of speech)| Avg audio length (s)| Avg RTF|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PyT+TRT|1| 128| FP16| 0.1641 | 0.0046 | 0.1694 | 0.1707 | 0.1731 | 900,884 | 6.52 | 577 | 7.13 | 43.44 |
|PyT    |1| 128| FP16| 1.07 | 0.06 | 1.14 | 1.17 | 1.23 | 144,668 | 1.00 | 602 | 7.42 |  6.95 |
