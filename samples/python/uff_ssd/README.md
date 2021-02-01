# Object Detection with SSD in Python

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Processing the input graph](#processing-the-input-graph)
	* [uff_ssd plugins](#uff_ssd-plugins)
	* [Verifying the output](#verifying-the-output)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, uff_ssd, implements a full UFF-based pipeline for performing inference with an SSD (InceptionV2 feature extractor) network.

This sample is based on the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper. The SSD network, built on the VGG-16 network, performs the task of object detection and localization in a single forward pass of the network. This approach discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple features with different resolutions to naturally handle objects of various sizes.

This sample is based on the TensorFlow implementation of SSD. For more information, download [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz). Unlike the paper, the TensorFlow SSD network was trained on the InceptionV2 architecture using the MSCOCO dataset which has 91 classes (including the background class). The config details of the network can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config).

## How does this sample work?

The sample downloads a pretrained [ssd_inception_v2_coco_2017_11_17](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) model and uses it to perform inference. Additionally, it superimposes bounding boxes on the input image as a post-processing step.

The SSD network performs the task of object detection and localization in a single forward pass of the network. The TensorFlow SSD network was trained on the InceptionV2 architecture using the [MSCOCO dataset](http://cocodataset.org/#home).

The sample makes use of TensorRT plugins to run the SSD network. To use these plugins the TensorFlow graph needs to be preprocessed.

When picking an object detection model for our application the usual trade-off is between model accuracy and inference time. In this sample we show how inference time of pretrained network can be greatly improved, without any decrease in accuracy, using TensorRT. In order to do that, we take a pretrained Tensorflow model, and use TensorRT’s UffParser to build a TensorRT inference engine.

The main components of this network are the Preprocessor, FeatureExtractor, BoxPredictor, GridAnchorGenerator and Postprocessor.

**Preprocessor**
The preprocessor step of the graph is responsible for resizing the image. The image is resized to a 300x300x3 size tensor. The preprocessor step also performs normalization of the image so all pixel values lie between the range [-1, 1].

**FeatureExtractor**
The FeatureExtractor portion of the graph runs the InceptionV2 network on the preprocessed image. The feature maps generated are used by the anchor generation step to generate default bounding boxes for each feature map.

In this network, the size of feature maps that are used for anchor generation are [(19x19), (10x10), (5x5), (3x3), (2x2), (1x1)].

**BoxPredictor**
The BoxPredictor step takes in a high level feature map as input and produces a list of box encodings (x-y coordinates) and a list of class scores for each of these encodings per feature map. This information is passed to the postprocessor.

**GridAnchorGenerator**
The goal of this step is to generate a set of default bounding boxes (given the scale and aspect ratios mentioned in the config) for each feature map cell. This is implemented as a plugin layer in TensorRT called the `gridAnchorGenerator` plugin. The registered plugin name is `GridAnchor_TRT`.

**Postprocessor**
The postprocessor step performs the final steps to generate the network output. The bounding box data and confidence scores for all feature maps are fed to the step along with the pre-computed default bounding boxes (generated in the `GridAnchorGenerator` namespace). It then performs NMS (non-maximum suppression) which prunes away most of the bounding boxes based on a confidence threshold and IoU (Intersection over Union) overlap, thus storing only the top N boxes per class. This is implemented as a plugin layer in TensorRT called the `NMS` plugin. The registered plugin name is `NMS_TRT`.

**FlattenConcat**
The `FlattenConcat` plugin is used to flatten each input and then concatenate the results. This is applied to the location and confidence data before it is fed to the post processor step since the NMS plugin requires the data to be in this format.

Specifically, this sample:
- [Processing the input graph](#processing-the-input-graph)
- [uff_ssd plugins](#uff_ssd-plugins)
- [Verifying the output](#verifying-the-output)

### Processing the input graph

The TensorFlow SSD graph has some operations that are currently not supported in TensorRT. Using [GraphSurgeon](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/graphsurgeon/graphsurgeon.html), we can combine multiple operations in the graph into a single custom operation which can be implemented using a plugin layer in TensorRT. Currently, GraphSurgeon provides the ability to stitch all nodes within a namespace into one custom node.

To use GraphSurgeon, the `convert-to-uff` utility should be called with a `-p` flag and a config file. The config script should also include attributes for all custom plugins which will be embedded in the generated `.uff` file. Current sample scripts for SSD is located in `/usr/src/tensorrt/samples/sampleUffSSD/config.py`.

Using GraphSurgeon, we were able to remove the preprocessor namespace from the graph, stitch the `GridAnchorGenerator` namespace to create the `GridAnchorGenerator` plugin, stitch the postprocessor namespace to the `NMS` plugin and mark the concat operations in the BoxPredictor as `FlattenConcat` plugins.

The TensorFlow graph has some operations like `Assert` and `Identity` which can be removed for inferencing. Operations like `Assert` are removed and leftover nodes (with no outputs once assert is deleted) are then recursively removed.

`Identity` operations are deleted and the input is forwarded to all the connected outputs. Additional documentation on the graph preprocessor can be found in the [TensorRT API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/graphsurgeon/graphsurgeon.html).

### uff_ssd plugins

Details about how to create TensorRT plugins can be found in [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending).

**`GridAnchorGeneration` plugin**
This plugin layer implements the grid anchor generation step in the TensorFlow SSD network. For each feature map we calculate the bounding boxes for each grid cell. In this network, there are 6 feature maps and the number of boxes per grid cell are as follows:

-   [19x19] feature map: 3 boxes (19x19x3x4(co-ordinates/box))
-   [10x10] feature map: 6 boxes (10x10x6x4)
-   [5x5] feature map: 6 boxes (5x5x6x4)
-   [3x3] feature map: 6 boxes (3x3x6x4)
-   [2x2] feature map: 6 boxes (2x2x6x4)
-   [1x1] feature map: 6 boxes (1x1x6x4)

**`NMS` plugin**
The `NMS` plugin generates the detection output based on location and confidence predictions generated by the BoxPredictor. This layer has three input tensors corresponding to location data (`locData`), confidence data (`confData`) and priorbox data (`priorData`).

The inputs to detection output plugin have to be flattened and concatenated across all the feature maps. We use the `FlattenConcat` plugin implemented in the sample to achieve this. The location data generated from the box predictor has the following dimensions:

```
19x19x12 -> Reshape -> 1083x4 -> Flatten -> 4332x1
10x10x24 -> Reshape -> 600x4 -> Flatten -> 2400x1
```

and so on for the remaining feature maps.

After concatenating, the input dimensions for `locData` input are of the order of 7668x1.

The confidence data generated from the box predictor has the following dimensions:

```
19x19x273 -> Reshape -> 1083x91 -> Flatten -> 98553x1
10x10x546 -> Reshape -> 600x91 -> Flatten -> 54600x1
```

and so on for the remaining feature maps.

After concatenating, the input dimensions for `confData` input are of the order of 174447x1.

The prior data generated from the grid anchor generator plugin has the following dimensions, for example 19x19 feature map has 2x4332x1 (there are two channels here because one channel is used to store variance of each coordinate that is used in the NMS step). After concatenating, the input dimensions for priorData input are of the order of 2x7668x1.

```
struct DetectionOutputParameters
{
	bool shareLocation, varianceEncodedInTarget;
	int backgroundLabelId, numClasses, topK, keepTopK;
	float confidenceThreshold, nmsThreshold;
	CodeTypeSSD codeType;
	int inputOrder[3];
	bool confSigmoid;
	bool isNormalized;
};
```

`shareLocation` and `varianceEncodedInTarget` are used for the Caffe implementation, so for the TensorFlow network they should be set to `true` and `false` respectively. The `confSigmoid` and `isNormalized` parameters are necessary for the TensorFlow implementation. If `confSigmoid` is set to `true`, it calculates the sigmoid values of all the confidence scores. The `isNormalized` flag specifies if the data is normalized and is set to `true` for the TensorFlow graph.

### Verifying the output

After the builder is created (see [Building an Engine in Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_python)) and the engine is serialized (see [Serializing a Model in Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_python)), we can perform inference. Steps for deserialization and running inference are outlined in [Performing Inference In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_python).

The outputs of the SSD network are human interpretable. The post-processing work, such as the final NMS, is done in the NMS plugin. The results are organized as tuples of 7. In each tuple, the 7 elements are respectively image ID, object label, confidence score, (`x,y`) coordinates of the lower left corner of the bounding box, and (`x,y`) coordinates of the upper right corner of the bounding box. This information can be drawn in the output PPM image using the `writePPMFileWithBBox` function. The `visualizeThreshold` parameter can be used to control the visualization of objects in the image. It is currently set to 0.5 so the output will display all objects with confidence score of 50% and above.

## Prerequisites

1. Launch the [NVIDIA tf1 (Tensorflow 1.x)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running) container.
    ```bash
    docker run --rm -it --gpus all -v `pwd`:/workspace nvcr.io/nvidia/tensorflow:20.12-tf1-py3 /bin/bash
    ```

    Alternatively, install Tensorflow 1.15
    `pip3 install tensorflow>=1.15.3,<2.0`

2. Install the dependencies for Python.
   ```bash
   python3 -m pip install -r requirements.txt
   ```

  NOTE:
  - On PowerPC systems, you will need to manually install TensorFlow using IBM's [PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm).
  - On Jetson boards, you will need to manually install TensorFlow by following the documentation for [Xavier](https://docs.nvidia.com/deeplearning/dgx/install-tf-xavier/index.html) or [TX2](https://docs.nvidia.com/deeplearning/dgx/install-tf-jetsontx2/index.html).

2.  Optional: To evaluate the accuracy of the trained model using the VOC dataset, perform the following steps.

  Download the VOC 2007 dataset. Run the following command from the sample root directory.
   ```bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   ```

  The first command downloads the VOC dataset from the Oxford servers, and the second command unpacks the dataset.

  **NOTE:** If the download link is broken, try alternate source http://vision.cs.utexas.edu/voc/VOC2007_test/. If you don’t want to save VOC in the sample root directory, you'll need to adjust the `--voc_dir` argument to `voc_evaluation.py` script before running it. The default value of this argument is `<SAMPLE_ROOT>/VOCdevkit/VOC2007`.

## Running the sample

Both the `detect_objects.py` and `voc_evaluation.py` scripts support separate advanced features, for example, lower precision inference, changing workspace directory and changing batch size.

1.  Run the inference script:
   ```bash
   python3 detect_objects.py <IMAGE_PATH>
   ```
  Where `<IMAGE_PATH>` contains the image you want to run inference on using the SSD network. The script should work for all popular image formats, like PNG, JPEG, and BMP. Since the model is trained for images of size 300 x 300, the input image will be resized to this size (using bilinear interpolation), if needed.

  For example:
   ```bash
   wget -nc http://images.cocodataset.org/val2017/000000252219.jpg -O test.jpg
   python3 detect_objects.py test.jpg
   ```

  When the inference script is run for the first time, it will run the following things to prepare its workspace:
  - The script downloads the pretrained `ssd_inception_v2_coco_2017_11_17` model from the TensorFlow object detection API. The script converts this model to TensorRT format, and the conversion is tailored to this specific version of the model.
  - The script builds a TensorRT inference engine and saves it to a file. During this step, all TensorRT optimizations will be applied to frozen graph. This is a time consuming operation and it can take a few minutes.

  After the workspace is ready, the script launches inference on the input image and saves the results to a location that will be printed on standard output. You can then open the saved image file and visually confirm that the bounding boxes are correct.

2.  Run the VOC evaluation script.

  1. Run the script using TensorRT:
   ```bash
   python3 voc_evaluation.py
   ```

  2. Run the script using TensorFlow:
   ```bash
   python3 voc_evaluation.py tensorflow
   ```
  **NOTE:** Running the script using TensorFlow will much slower than the TensorRT evaluation.

  3. AP and mAP metrics are displayed at the end of the script execution. The metrics for the TensorRT engine should match those of the original TensorFlow model.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

# Additional resources

The following resources provide a deeper understanding about the SSD model and object detection:

**Model**
- [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

**Dataset**
- [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz)
- [MSCOCO dataset](http://cocodataset.org/#home)

**Documentation**
- [Introduction to NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working with TensorRT Using the Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)
- [SSD: Single Shot MultiBox Detector Paper](https://arxiv.org/abs/1512.02325)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
