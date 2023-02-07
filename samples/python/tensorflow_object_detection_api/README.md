# TensorFlow Object Detection API Models in TensorRT


Support for [TensorFlow Object Detection (TFOD) API](https://github.com/tensorflow/models/tree/master/research/object_detection) models in TensorRT, including Single Shot Detector, Faster R-CNN and Mask R-CNN models. This script helps with converting, running and validating these models with TensorRT.

## Setup

### TensorFlow and TensorRT Environment

In order for scripts to work we suggest an environment with TensorRT >= 8.0.1 and TensorFlow 2.5.

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

If you would like to use Docker, you can use an NGC image to fulfill these requirements, such as:

```
docker pull nvcr.io/nvidia/tensorflow:21.10-tf2-py3
```
### TFOD API Environment

To run these scripts, you will also need a recent TFOD installation. You can do so by cloning the https://github.com/tensorflow/models/tree/master/research/object_detection repository and following the [installation procedure for TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

Alternatively, if you're running on an NGC TF2 docker, the TFOD installation procedure can be done as follows:

```
cd /workspace
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.15.4/protoc-3.15.4-linux-x86_64.zip
unzip protoc*.zip bin/protoc -d /usr/local
git clone https://github.com/tensorflow/models.git
cd /workspace/models/research
git checkout 08b6803
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py ./
pip --use-deprecated=legacy-resolver install .
```

You can verify the installation by running:

```
pip show object-detection
```

### Other Dependencies

Install all dependencies listed in this sample's `requirements.txt` file:

```
pip install -r requirements.txt
```

You will also need the `onnx-graphsurgeon` python module. If not already installed by TensorRT, you can install it manually by running:

```
pip install onnx-graphsurgeon==0.3.10 --index-url https://pypi.ngc.nvidia.com
```

**NOTE:** Please make sure that the `onnx-graphsurgeon` module installed by pip is version == 0.3.10.

## Model Conversion

The workflow to convert a TFOD model is basically TensorFlow → ONNX → TensorRT, and so parts of this process require TensorFlow to be installed. If you are performing this conversion to run inference on the edge, such as for NVIDIA Jetson devices, it might be easier to do the ONNX conversion on a PC first, and then build the TensorRT engine on the device later.

### TensorFlow Saved Model

The starting point of conversion is a TensorFlow saved model. This can be exported from your own models trained with the TFOD API, or you can download a pre-trained model from the [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), such as:

```
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
tar -xvf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
```

When extracted, this package holds a directory named `saved_model` which holds the saved model ready for conversion.

Structure of the checkpoint is similar to this:

```
ssd_mobilenet_v2_320x320_coco17_tpu-8
├── checkpoint
│   ├── ckpt-0.data-00000-of-00001
│   └── ckpt-0.index
├── pipeline.config
└── saved_model
    └── saved_model.pb
```

**NOTE:** In order to proceed, you need to re-export the saved model. If you don't re-export the model with with **float_image_tensor** as input type, the conversion process will fail.

To re-export, run exporter_main_v2.py located in TFOD API:

```
cd /path/to/models/research/object_detection
python exporter_main_v2.py \
    --input_type float_image_tensor \
    --trained_checkpoint_dir /path/to/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint \
    --pipeline_config_path /path/to/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config \
    --output_directory /path/to/export
```

Where `--trained_checkpoint_dir` and `--pipeline_config_path` point to the corresponding paths in the training checkpoint, `--input_type` must be set to `float_image_tensor` for the ONNX conversion process to run correctly. On the path pointed by `--output_directory` you will then find the newly created export, under which there will be a re-exported saved model  in a directory aptly named `saved_model`.

>**NOTE:** When you try to re-export the **Mask R-CNN** pre-trained model, you may see an error message such as:
>
>```
>google.protobuf.text_format.ParseError: 153:40 : Message type "object_detection.protos.TFRecordInputReader" has no field named "s".
>```
>
>This error comes from a bug inside of the `/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config` file, in order to fix it, all you have to do is find a line with `input_path: "PATH_TO_BE_CONFIGURED"s`, currently it is line **153** and delete the final `s` on the end of this line, save the file and try to re-export again.

### Create ONNX Graph

This is a list of supported models, additionally keep in mind that pre-trained TF2 model zoo names do not always reflect the actual model input resolution, see correct supported model resolution breakdown here:

| **Model**                                     | **Resolution** |
| ----------------------------------------------|----------------|
| SSD MobileNet v2 320x320                      | **300x300**       |
| SSD MobileNet V1 FPN 640x640                  | 640x640        |
| SSD MobileNet V2 FPNLite 320x320              | 320x320        |
| SSD MobileNet V2 FPNLite 640x640              | 640x640        |
| SSD ResNet50 V1 FPN 640x640 (RetinaNet50)     | 640x640        |
| SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)   | 1024x1024      |
| SSD ResNet101 V1 FPN 640x640 (RetinaNet101)   | 640x640        |
| SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101) | 1024x1024      |
| SSD ResNet152 V1 FPN 640x640 (RetinaNet152)   | 640x640        |
| SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152) | 1024x1024      |
| Faster R-CNN ResNet50 V1 640x640              | 640x640        |
| Faster R-CNN ResNet50 V1 1024x1024            | 1024x1024      |
| Faster R-CNN ResNet50 V1 800x1333             | **1333x1333**      |
| Faster R-CNN ResNet101 V1 640x640             | 640x640        |
| Faster R-CNN ResNet101 V1 1024x1024           | 1024x1024      |
| Faster R-CNN ResNet101 V1 800x1333            | **1333x1333**      |
| Faster R-CNN ResNet152 V1 640x640             | 640x640        |
| Faster R-CNN ResNet152 V1 1024x1024           | 1024x1024      |
| Faster R-CNN ResNet152 V1 800x1333            | **1333x1333**      |
| Faster R-CNN Inception ResNet V2 640x640      | 640x640        |
| Faster R-CNN Inception ResNet V2 1024x1024    | **1333x1333**      |
| Mask R-CNN Inception ResNet V2 1024x1024      | 1024x1024      |


If the TF saved model is ready to be converted (i.e. you ran `exporter_main_v2.py` to re-export as float_image_tensor), run:

```
python create_onnx.py \
    --pipeline_config /path/to/exported/pipeline.config \
    --saved_model /path/to/exported/saved_model \
    --onnx /path/to/save/model.onnx
```

This will create the file `model.onnx` which is ready to convert to TensorRT.

The script has a few optional arguments, including:

* `--first_nms_threshold [...]` allows overriding the default 1st NMS score threshold parameter, as the runtime latency of the NMS plugin is sensitive to this value. It's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce inference latency. In case of SSD models it will be a score threshold for first and final NMS, in case of Faster R-CNN and Mask R-CNN this will be a score threshold for Region Proposal Network.
* `--second_nms_threshold [...]` allows overriding the default 2nd NMS score threshold parameter, further improves the runtime latency of the NMS plugin. It's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce inference latency. Only applicable in case of Faster R-CNN and Mask R-CNN since it will be the second and last NMS for both networks. 
* `--batch_size` allows selection of various batch sizes, default is 1.
* `--debug` allows to add an extra output to debug a particular node. 
* `--input_format` allows to set an input format of the network, either `NHWC` that is set by default or `NCHW`. 
* `--tf2onnx` allows to save an intermediate ONNX graph generated by t2onnx. 

Optionally, you may wish to visualize the resulting ONNX graph with a tool such as [Netron](https://netron.app/).

The input to the graph is a `float32` tensor with the selected input shape, containing RGB pixel data in the range of 0 to 255. All color value preprocessing will be performed inside the graph.

The outputs of the graph are the same as the outputs of the [EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/master/plugin/efficientNMSPlugin) plugin, named `num_detections`, `detection_boxes`, `detection_classes` and `detection_scores`. In the case of **Mask R-CNN**, the graph will have one more additional output from a segmentation head named `detection_masks`,  with shape `[batch_size, max_proposals, mask_height, mask_width]`, and float32 data type.

### Build TensorRT Engine

It is possible to build the TensorRT engine directly with `trtexec` using the ONNX graph generated in the previous step. However, the script `build_engine.py` is provided for convenience, as it has been tailored to engine building for TFOD models and calibration. Run `python build_engine.py --help` for details on available settings.

#### FP16 Precision

To build the TensorRT engine file with FP16 precision, run:

```
python build_engine.py \
    --onnx /path/to/saved/model.onnx \
    --engine /path/to/save/engine.trt \
    --precision fp16
```

The file `engine.trt` will be created, which can now be used to infer with TensorRT.

For best results, make sure no other processes are using the GPU during engine build, as it may affect the optimal tactic selection process.

**Note:** If you receive any error messages about non sufficient workspace size, especially when converting models with a larger batch size, increase the max workspace by adding an argument such as `--workspace 8` to assign up to an 8 Gb workspace size.

#### INT8 Precision

To build and calibrate an engine for INT8 precision, run:

```
python build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
```

Where `--calib_input` points to a directory with several thousands of images. For example, this could be a subset of the training or validation datasets that were used for the model. It's important that this data represents the runtime data distribution relatively well, therefore, the more images that are used for calibration, the better accuracy that will be achieved in INT8 precision. For models trained for the [COCO dataset](https://cocodataset.org/#home), we have found that 5,000 images gives a good result.

The `--calib_cache` controls where the calibration cache file will be written to. This is useful to keep a cached copy of the calibration results. Next time you need to build the engine for the same network, if this file exists, it will skip the calibration step and use the cached values instead.

#### Benchmark Engine

Optionally, you can obtain execution timing information for the built engine by using the `trtexec` utility, as:

```
trtexec \
    --loadEngine=/path/to/engine.trt \
    --useCudaGraph --noDataTransfers \
    --iterations=100 --avgRuns=100
```

If it's not already in your `$PATH`, the `trtexec` binary is usually found in `/usr/src/tensorrt/bin/trtexec`, depending on your TensorRT installation method.

An inference benchmark will run, with GPU Compute latency times printed out to the console. Depending on your environment, you should see something similar to:

```
GPU Compute Time: min = 1.55835 ms, max = 1.91591 ms, mean = 1.58719 ms, median = 1.578 ms, percentile(99%) = 1.90668 ms
```

## Inference

For optimal performance, inference should be done in a C++ application that takes advantage of CUDA Graphs to launch the inference request. Alternatively, the TensorRT engine built with this process can also be executed through either [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) or [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

However, for convenience, a python inference script is provided here for quick testing of the built TensorRT engine.

### Inference in Python

To perform object detection on a set of images with TensorRT, run:

```
python infer.py \
    --engine /path/to/saved/engine.trt \
    --input /path/to/images \
    --output /path/to/output \
    --preprocessor fixed_shape_resizer \
    --labels /path/to/labels_coco.txt
```

Where the input path can be either a single image file, or a directory of jpg/png/bmp images.

The argument `--preprocessor` corresponds to the image preprocessor set in your `pipeline.config` file, usually it is under `image_resizer` section, only two are now supported, namely `fixed_shape_resizer` and `keep_aspect_ratio_resizer`.

The argument `--labels` is a path to a file that contains label data and is included with this repository.

The script has a few optional arguments, including:
* `--nms_threshold` allows overriding the default NMS score threshold parameter.
* `--detection_type` allows to select a detection type, either `bbox` that is set by default or `segmentation` that works only with **Mask R-CNN**. 
* `--iou_threshold` allows to set IoU threshold for the mask segmentation, works only with **Mask R-CNN**, default is 0.5.

The detection results will be written out to the specified output directory, consisting of a visualization image, and a tab-separated results file for each input image processed.

#### SSD ResNet152 V1 FPN 1024x1024

![ssd_infer](https://drive.google.com/uc?export=view&id=1xSA7IkJuAScCf_NaVyiAX6dA9BPQVyJf)

#### Mask R-CNN  

![mrcnn_infer](https://drive.google.com/uc?export=view&id=1c_8hXdFjjpEWKJSJfqoeKKgVXE3BTuyQ)



### Evaluate mAP Metric

Given a validation dataset (such as [COCO val2017 data](http://images.cocodataset.org/zips/val2017.zip)) and ground truth annotations (such as [COCO instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)), you can get the mAP metrics for the built TensorRT engine. This will use the mAP metrics tools functions from the [TFOD's research object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) repository.

```
python eval_coco.py \
    --engine /path/to/engine.trt \
    --input /path/to/coco/val2017 \
    --annotations /path/to/coco/annotations/instances_val2017.json \
    --preprocessor fixed_shape_resizer
```

The script has a few optional arguments, including:
* `--nms_threshold` allows overriding the default NMS score threshold parameter.
* `--detection_type` allows to select a detection type, either `bbox` that is set by default. Alternatively you can set it to `segmentation` for **Mask R-CNN** models, to evaluate the masks instead.
* `--iou_threshold` allows to set IoU threshold for the mask segmentation, works only with **Mask R-CNN**, default is 0.5.

The argument`--preprocessor` corresponds to the image preprocessor set in your `pipeline.config` file as described in the inference section above.

The mAP metric is sensitive to the NMS score threshold used, as using a high threshold will reduce the model recall, resulting in a lower mAP value. Ideally, mAP should be measured with a threshold of 0, but such a low value will impact the runtime latency of the EfficientNMS_TRT plugin. It may be a good idea to build separate TensorRT engines for different purposes. That is, one engine with a low threshold (like 0) dedicated for mAP validation, and another engine with your application specific threshold (like 0.4) for deployment. This is why we keep the NMS threshold as a configurable parameter in the `create_onnx.py` script.

### TF vs TRT Comparison

To compare how the TensorRT detections match the original TensorFlow model results, you can run:

```
python compare_tf.py \
    --engine /path/to/saved/engine.trt \
    --saved_model /path/to/exported/saved_model \
    --input /path/to/images \
    --annotations /path/to/coco/annotations/instances_val2017.json \
    --output /path/to/output \
    --preprocessor fixed_shape_resizer \
    --labels /path/to/labels_coco.txt
```

The script has a few optional arguments, including:
* `--nms_threshold` allows overriding the default NMS score threshold parameter.
* `--detection_type` allows to select a detection type, either `bbox` that is set by default or `segmentation` that works only with **Mask R-CNN**.
* `--iou_threshold` allows to set IoU threshold for the mask segmentation, works only with **Mask R-CNN**, default is 0.5.
* `--num_images` the maximum number of images to visualize if you are passing a directory to `--input`.

This script will process the images found in the given input path through both TensorFlow and TensorRT using the corresponding saved model and engine. It will then write to the output path a set of visualization images showing the inference results of both frameworks for visual qualitative comparison. Argument `--preprocessor` corresponds to the image preprocessor set in your `pipeline.config` file as described in the inference section above.

If you run this on COCO val2017 images, you may also add the parameter `--annotations /path/to/coco/annotations/instances_val2017.json` to further compare against COCO ground truth annotations.

#### SSD ResNet152 V1 FPN 1024x1024

![ssd_compare_tf](https://drive.google.com/uc?export=view&id=1eX6LucHjDpmwG5tK-MGebRaW0zmdeee-)

#### Mask R-CNN

![mrcnn_compare_tf](https://drive.google.com/uc?export=view&id=1kNnfJ2H5OY85Z2e6KNxZgiYk3Lo-sB9r)

