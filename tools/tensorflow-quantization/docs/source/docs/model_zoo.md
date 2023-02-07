# **Model Zoo Results**
Results obtained on NVIDIA's A100 GPU and TensorRT 8.4.

## [ResNet](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization/examples/resnet)
### ResNet50-v1

| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 75.05        | 7.95               |
| PTQ (TensorRT)        | 74.96        | 0.46               |
| **QAT** (TensorRT)    | 75.12        | 0.45               |

### ResNet50-v2

| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 75.36        | 6.16               |
| PTQ (TensorRT)        | 75.48        | 0.57               |
| **QAT** (TensorRT)    | 75.65        | 0.57               |

### ResNet101-v1

| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 76.47        | 15.92              |
| PTQ (TensorRT)        | 76.32        | 0.84               |
| **QAT** (TensorRT)    | 76.26        | 0.84               |

### ResNet101-v2

| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 76.89        | 14.13              |
| PTQ (TensorRT)        | 76.94        | 1.05               |
| **QAT** (TensorRT)    | 77.15        | 1.05               |

*QAT fine-tuning hyper-parameters: `bs=32` (`bs=64` was OOM).*

## [MobileNet](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization/examples/mobilenet)
### MobileNet-v1

| Model                 | Accuracy (%) | Latency (ms, bs=1)  |
|-----------------------|--------------|---------------------|
| Baseline (TensorFlow) | 70.60        | 1.99                |
| PTQ (TensorRT)        | 69.31        | 0.16                |
| **QAT** (TensorRT)    | 70.43        | 0.16                |

### MobileNet-v2

| Model                 | Accuracy (%) | Latency (ms, bs=1)    |
|-----------------------|--------------|-----------------------|
| Baseline (TensorFlow) | 71.77        | 3.71                  |
| PTQ (TensorRT)        | 70.87        | 0.30                  |
| **QAT** (TensorRT)    | 71.62        | 0.30                  |

## [EfficientNet](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization/examples/efficientnet)
### EfficientNet-B0
| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 76.97        | 6.77               |
| PTQ (TensorRT)        | 71.71        | 0.67               |
| **QAT** (TensorRT)    | 75.82        | 0.68               |

*QAT fine-tuning hyper-parameters: `bs=64, ep=10, lr=0.001, steps_per_epoch=None`*.

### EfficientNet-B3
| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 81.36        | 10.33              |
| PTQ (TensorRT)        | 78.88        | 1.24               |
| **QAT** (TensorRT)    | 79.48        | 1.23               |

*QAT fine-tuning hyper-parameters: `bs=32, ep20, lr=0.0001, steps_per_epoch=None`*.

## [Inception](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization/examples/inception)
### Inception-v3

| Model                 | Accuracy (%) | Latency (ms, bs=1) |
|-----------------------|--------------|--------------------|
| Baseline (TensorFlow) | 77.86        | 9.01               |
| PTQ (TensorRT)        | 77.73        | 0.82               |
| **QAT** (TensorRT)    | 78.08        | 0.82               |

```{eval-rst}

.. NOTE::

    The results here were obtained with NVIDIA's A100 GPU and TensorRT 8.4.
    
    Accuracy metric: Top-1 validation accuracy with the full ImageNet dataset.

    Hyper-parameters

    #. QAT fine-tuning: `bs=64`, `ep=10`, `lr=0.001` (unless otherwise stated).
    #. PTQ calibration: `bs=64`.

```