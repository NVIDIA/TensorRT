# Detectron 2 Mask R-CNN R50-FPN 3x in TensorRT

Support for Detectron 2 Mask R-CNN R50-FPN 3x model in TensorRT. This script helps with converting, running and validating this model with TensorRT.

## Changelog

- July 2023:
  - Update benchmarks and include hardware used.
- October 2022:
  - Updated converter to support `tracing` export instead of deprecated `caffe2_tracing`

## Setup

In order for scripts to work we suggest an environment with TensorRT >= 8.4.1.

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

Install all dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```
Note: this sample cannot be run on Jetson platforms as `torch.distributed` is not available. To check whether your platform supports `torch.distributed`, open a Python shell and confirm that `torch.distributed.is_available()` returns `True`.

## Model Conversion

The workflow to convert Detectron 2 Mask R-CNN R50-FPN 3x model is basically Detectron 2 → ONNX → TensorRT, and so parts of this process require Detectron 2 to be installed. Official export to ONNX is documented [here](https://detectron2.readthedocs.io/en/latest/tutorials/deployment.html).

### Detectron 2 Deployment
Deployment is done through export model script located in `detectron2/tools/deploy/export_model.py` of Detectron 2 [github](https://github.com/facebookresearch/detectron2). Detectron 2 Mask R-CNN R50-FPN 3x model is dynamic with minimum testing dimension size of 800 and maximum of 1333. TensorRT plug-ins used for conversion of this model do not support dynamic shapes, as a result we have to set both height and width of the input tensor to 1344. 1344 instead of 1333 because model requires both height and width of the input tensor to be divisible by 32. In order to export this model with correct 1344x1344 resolution, we have to make a change to `export_model.py`. Currently lines 160-162:

```
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
```
have to be changed to:

```
aug = T.ResizeShortestEdge(
    [1344, 1344], 1344
)
```

Export script takes `--sample-image` as one of the arguments. Such image is used to adjust input dimensions and dimensions of tensors for the rest of the network. This sample image has to be an image of 1344x1344 dimensions, which contains at least one detectable by model object. My recommendation is to upsample one of COCO dataset images to 1344x1344. Sample command:

```
python detectron2/tools/deploy/export_model.py \
    --sample-image 1344x1344.jpg \
    --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --export-method tracing \
    --format onnx \
    --output ./ \
    MODEL.WEIGHTS path/to/model_final_f10217.pkl \
    MODEL.DEVICE cuda

```

Where `--sample-image` is 1344x1344 image; `--config-file` path to Mask R-CNN R50-FPN 3x config, included with detectron2; `MODEL.WEIGHTS` are weights of Mask R-CNN R50-FPN 3x that can be downloaded [here](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md). Resulted `model.onnx` will be an input to conversion script.

### Create ONNX Graph
This is supported Detectron 2 model:

| **Model**                                     | **Resolution** |
| ----------------------------------------------|----------------|
| Mask R-CNN R50-FPN 3x                         | 1344x1344      |

If Detectron 2 Mask R-CNN is ready to be converted (i.e. you ran `detectron2/tools/deploy/export_model.py`), run:

```
python create_onnx.py \
    --exported_onnx /path/to/model.onnx \
    --onnx /path/to/converted.onnx \
    --det2_config /detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --det2_weights /model_final_f10217.pkl \
    --sample_image any_image.jpg
```

This will create the file `converted.onnx` which is ready to convert to TensorRT.

It is important to mention that `--sample_image` in this case is used for anchor generation. Detectron 2 ONNX models do not have anchor data inside the graph, so anchors have to be generated "offline". If custom model is used, make sure preprocessing of your model matches what is coded in `get_anchors(self, sample_image)` function.

The script has a few optional arguments, including:

* `--first_nms_threshold [...]` allows overriding the default 1st NMS score threshold parameter, as the runtime latency of the NMS plugin is sensitive to this value. It's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce inference latency. In Mask R-CNN this will be a score threshold for Region Proposal Network.
* `--second_nms_threshold [...]` allows overriding the default 2nd NMS score threshold parameter, further improves the runtime latency of the NMS plugin. It's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce inference latency. It will be the second and last NMS.
* `--batch_size` allows selection of various batch sizes, default is 1.


Optionally, you may wish to visualize the resulting ONNX graph with a tool such as [Netron](https://netron.app/).

The input to the graph is a `float32` tensor with the selected input shape, containing RGB pixel data in the range of 0 to 255. All preprocessing will be performed inside the Model graph, so it is not required to further pre-process the input data.


The outputs of the graph are the same as the outputs of the [EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/master/plugin/efficientNMSPlugin) plugin and segmentation head output, name of the last node is `detection_masks`, shape is `[batch_size, max_proposals, mask_height, mask_width]`, dtype is float32.

### Build TensorRT Engine

TensorRT engine can be built directly with `trtexec` using the ONNX graph generated in the previous step. If it's not already in your `$PATH`, the `trtexec` binary is usually found in `/usr/src/tensorrt/bin/trtexec`, depending on your TensorRT installation method. Run:

```
trtexec --onnx=/path/to/converted.onnx --saveEngine=/path/to/engine.trt --useCudaGraph
```

However, the script `build_engine.py` is also provided in this repository for convenience, as it has been tailored to Detectron 2 2 Mask R-CNN R50-FPN 3x engine building and INT8 calibration. Run `python3 build_engine.py --help` for details on available settings.

#### FP16 Precision

To build the TensorRT engine file with FP16 precision, run:

```
python3 build_engine.py \
    --onnx /path/to/converted.onnx \
    --engine /path/to/engine.trt \
    --precision fp16
```

The file `engine.trt` will be created, which can now be used to infer with TensorRT.

For best results, make sure no other processes are using the GPU during engine build, as it may affect the optimal tactic selection process.

#### INT8 Precision

To build and calibrate an engine for INT8 precision, run:

```
python3 build_engine.py \
    --onnx /path/to/converted.onnx \
    --engine /path/to/engine.trt \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
```

Where `--calib_input` points to a directory with several thousands of images. For example, this could be a subset of the training or validation datasets that were used for the model. It's important that this data represents the runtime data distribution relatively well, therefore, the more images that are used for calibration, the better accuracy that will be achieved in INT8 precision. For models trained for the [COCO dataset](https://cocodataset.org/#home), we have found that 5,000 images gives a good result.

The `--calib_cache` is optional, and it controls where the calibration cache file will be written to. This is useful to keep a cached copy of the calibration results. Next time you need to build an int8 engine for the same network, if this file exists, the builder will skip the calibration step and use the cached values instead.

#### Benchmark Engine

Optionally, you can obtain execution timing information for the built engine by using the `trtexec` utility, as:

```
trtexec \
    --loadEngine=/path/to/engine.trt \
    --useCudaGraph --noDataTransfers \
    --iterations=100 --avgRuns=100
```

An inference benchmark will run, with GPU Compute latency times printed out to the console. Depending on your environment, you should see something similar to:

```
GPU Compute Time: min = 30.1864 ms, max = 37.0945 ms, mean = 34.481 ms, median = 34.4187 ms, percentile(99%) = 37.0945 ms
```

Some sample results comparing different data precisions are shown below. The following results were obtained using an RTX A5000 and TensorRT 8.6.1. mAP was evaluated for the COCO val2017 dataset using the instructions in [Evaluate mAP Metric](#evaluate-map-metric).

| **Precision**   | **Latency** | **bbox COCO mAP** | **segm COCO mAP** |
| ----------------|-------------|-------------------|-------------------|
| fp32            | 25.89 ms    | 0.402             | 0.368             |
| fp16            | 13.00 ms    | 0.402             | 0.368             |
| int8            | 7.29 ms     | 0.399             | 0.366             |

## Inference

For optimal performance, inference should be done in a C++ application that takes advantage of CUDA Graphs to launch the inference request. Alternatively, the TensorRT engine built with this process can also be executed through either [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) or [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

However, for convenience, a python inference script is provided here for quick testing of the built TensorRT engine.

### Inference in Python

To perform object detection on a set of images with TensorRT, run:

```
python infer.py \
    --engine /path/to/engine.trt \
    --input /path/to/images \
    --det2_config /detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output /path/to/output \
```

Where the input path can be either a single image file, or a directory of jpg/png/bmp images.

The script has a few optional arguments, including:
* `--nms_threshold` allows overriding the default second NMS score threshold parameter.
* `--iou_threshold` allows to set IoU threshold for the mask segmentation, default is 0.5.

The detection results will be written out to the specified output directory, consisting of a visualization image, and a tab-separated results file for each input image processed.

#### Sample Images

![infer_1](https://drive.google.com/uc?export=view&id=1AOW9IXqjrU7eVYmaue-pqijNucXmx_s0)

![infer_2](https://drive.google.com/uc?export=view&id=1m1fp2v41DOqKfj423G0-eyKVurrPNYGx)

### Evaluate mAP Metric

Given a validation dataset (such as [COCO val2017 data](http://images.cocodataset.org/zips/val2017.zip)), you can get the mAP metrics for the built TensorRT engine. This will use the mAP metrics tools functions from the [Detectron 2 evaluation](https://github.com/facebookresearch/detectron2/tree/main/detectron2/evaluation) repository. Make sure you follow [Use Builtin Datasets guide](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html) to correctly setup COCO or custom dataset. Additionally, run `eval_coco.py` in the same folder where `/datasets` is present, otherwise this error will appear:

```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/coco/annotations/instances_val2017.json'
```

To run evalutions, run:

```
python eval_coco.py \
    --engine /path/to/engine.trt \
    --input /path/to/coco/val2017 \
    --det2_config /detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --det2_weights /model_final_f10217.pkl
```

The script has a few optional arguments, including:
* `--nms_threshold` allows overriding the default second NMS score threshold parameter.
* `--iou_threshold` allows to set IoU threshold for the mask segmentation, default is 0.5.

The mAP metric is sensitive to the NMS score threshold used, as using a high threshold will reduce the model recall, resulting in a lower mAP value. It may be a good idea to build separate TensorRT engines for different purposes. That is, one engine with a default threshold (like 0.5) dedicated for mAP validation, and another engine with your application specific threshold (like 0.8) for deployment. This is why we keep the NMS threshold as a configurable parameter in the `create_onnx.py` script.
