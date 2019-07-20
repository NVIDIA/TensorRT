# Object Detection With A TensorFlow SSD Network


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Processing the input graph](#processing-the-input-graph)
	* [Preparing the data](#preparing-the-data)
	* [sampleUffSSD plugins](#sampleuffssd-plugins)
	* [Verifying the output](#verifying-the-output)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleUffSSD, preprocesses a TensorFlow SSD network, performs inference on the SSD network in TensorRT, using TensorRT plugins to speed up inference.

This sample is based on the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper. The SSD network performs the task of object detection and localization in a single forward pass of the network.

The SSD network used in this sample is based on the TensorFlow implementation of SSD, which actually differs from the original paper, in that it has an inception_v2 backbone. For more information about the actual model, download [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz). The TensorFlow SSD network was trained on the InceptionV2 architecture using the [MSCOCO dataset](http://cocodataset.org/#home) which has 91 classes (including the background class). The config details of the network can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config).

## How does this sample work?

The SSD network performs the task of object detection and localization in a single forward pass of the network. The TensorFlow SSD network was trained on the InceptionV2 architecture using the MSCOCO dataset.

The sample makes use of TensorRT plugins to run the SSD network. To use these plugins, the TensorFlow graph needs to be preprocessed, and we use the GraphSurgeon utility to do this.

The main components of this network are the Image Preprocessor, FeatureExtractor, BoxPredictor, GridAnchorGenerator and Postprocessor.

**Image Preprocessor**
The image preprocessor step of the graph is responsible for resizing the image. The image is resized to a 300x300x3 size tensor. This step also performs normalization of the image so all pixel values lie between the range [-1, 1].

**FeatureExtractor**
The FeatureExtractor portion of the graph runs the InceptionV2 network on the preprocessed image. The feature maps generated are used by the anchor generation step to generate default bounding boxes for each feature map.

In this network, the size of feature maps that are used for anchor generation are [(19x19), (10x10), (5x5), (3x3), (2x2), (1x1)].

**BoxPredictor**
The BoxPredictor step takes in a high level feature map as input and produces a list of box encodings (x-y coordinates) and a list of class scores for each of these encodings per feature map. This information is passed to the postprocessor.

**GridAnchorGenerator**
The goal of this step is to generate a set of default bounding boxes (given the scale and aspect ratios mentioned in the config) for each feature map cell. This is implemented as a plugin layer in TensorRT called the `gridAnchorGenerator` plugin. The registered plugin name is `GridAnchor_TRT`.

**Postprocessor**
The postprocessor step performs the final steps to generate the network output. The bounding box data and confidence scores for all feature maps are fed to the step along with the pre-computed default bounding boxes (generated in the `GridAnchorGenerator` namespace). It then performs NMS (non-maximum suppression) which prunes away most of the bounding boxes based on a confidence threshold and IoU (Intersection over Union) overlap, thus storing only the top `N` boxes per class. This is implemented as a plugin layer in TensorRT called the NMS plugin. The registered plugin name is `NMS_TRT`.

**Note:** This sample also implements another plugin called `FlattenConcat` which is used to flatten each input and then concatenate the results. This is applied to the location and confidence data before it is fed to the post processor step since the NMS plugin requires the data to be in this format.

For details on how a plugin is implemented, see the implementation of `FlattenConcat` plugin and `FlattenConcatPluginCreator` in the `sampleUffSSD.cpp` file in the `tensorrt/samples/sampleUffSSD` directory.

Specifically, this sample performs the following steps:
	- [Processing the input graph](#processing-the-input-graph)
	- [Preparing the data](#preparing-the-data)
	- [sampleUffSSD plugins](#sampleuffssd-plugins)
	- [Verifying the output](#verifying-the-output)

### Processing the input graph

The TensorFlow SSD graph has some operations that are currently not supported in TensorRT. Using a preprocessor on the graph, we can combine multiple operations in the graph into a single custom operation which can be implemented as a plugin layer in TensorRT. Currently, the preprocessor provides the ability to stitch all nodes within a namespace into one custom node.

To use the preprocessor, the `convert-to-uff` utility should be called with a `-p` flag and a config file. The config script should also include attributes for all custom plugins which will be embedded in the generated `.uff` file. Current sample script for SSD is located in `/usr/src/tensorrt/samples/sampleUffSSD/config.py`.

Using the preprocessor on the graph, we were able to remove the `Preprocessor` namespace from the graph, stitch the `GridAnchorGenerator` namespace together to create the `GridAnchorGenerator` plugin, stitch the `postprocessor` namespace together to get the NMS plugin and mark the concat operations in the BoxPredictor as `FlattenConcat` plugins.

The TensorFlow graph has some operations like `Assert` and `Identity` which can be removed for the inferencing. Operations like `Assert` are removed and leftover nodes (with no outputs once assert is deleted) are then recursively removed.

`Identity` operations are deleted and the input is forwarded to all the connected outputs. Additional documentation on the graph preprocessor can be found in the [TensorRT API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/graphsurgeon/graphsurgeon.html).

### Preparing the data

The generated network has an input node called `Input`, and the output node is given the name `MarkOutput_0` by the UFF converter. These nodes are registered by the UFF Parser in the sample.

```
parser->registerInput("Input", DimsCHW(3, 300, 300),
UffInputOrder::kNCHW);  
parser->registerOutput("MarkOutput_0");  
```  

The input to the SSD network in this sample is 3 channel 300x300 images. In the sample, we normalize the image so the pixel values lie in the range [-1,1]. This is equivalent to the image preprocessing stage of the network.

Since TensorRT does not depend on any computer vision libraries, the images are represented in binary `R`, `G`, and `B` values for each pixel. The format is Portable PixMap (PPM), which is a netpbm color image format. In this format, the `R`, `G`, and `B` values for each pixel are represented by a byte of integer (0-255) and they are stored together, pixel by pixel.

There is a simple PPM reading function called `readPPMFile`.

### sampleUffSSD plugins

Details about how to create TensorRT plugins can be found in [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending).

The `config.py` defined for the `convert-to-uff` command should have the custom layers mapped to the plugin names in TensorRT by modifying the `op` field. The names of the plugin parameters should also exactly match those expected by the TensorRT plugins. For example, for the `GridAnchor` plugin, the `config.py` should have the following:

```
PriorBox = gs.create_plugin_node(name="GridAnchor",
op="GridAnchor_TRT",  
	numLayers=6,  
	minSize=0.2,  
	maxSize=0.95,  
	aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],  
	variance=[0.1,0.1,0.2,0.2],  
	featureMapShapes=[19, 10, 5, 3, 2, 1])  
```  

Here, `GridAnchor_TRT` matches the registered plugin name and the parameters have the same name and type as expected by the plugin.

If the `config.py` is defined as above, the NvUffParser will be able to parse the network and call the appropriate plugins with the correct parameters.

Details about some of the plugin layers implemented for SSD in TensorRT are given below.

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

The inputs to detection output plugin have to be flattened and concatenated across all the feature maps. We use the `FlattenConcat` plugin implemented in the sample to achieve this.  The location data generated from the box predictor has the following dimensions:

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

After concatenating, the input dimensions for `confData` input are 174447x1.

The prior data generated from the grid anchor generator plugin has 6 outputs and their dimensions are as follows:

```
Output 1 corresponds to the 19x19 feature map and has dimensions 2x4332x1    
Output 2 corresponds to the 10x10 feature map and has dimensions 2x2400x1    
```
and so on for the other feature maps.

**Note:** There are two channels in the outputs because one channel is used to store variance of each coordinate that is used in the NMS step. After concatenating, the input dimensions for `priorData` input are of the order of 2x7668x1.

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

`shareLocation` and `varianceEncodedInTarget` are used for the Caffe SSD network implementation, so for the TensorFlow network they should be set to `true` and `false` respectively. The `confSigmoid` and `isNormalized` parameters are necessary for the TensorFlow implementation. If `confSigmoid` is set to `true`, it calculates the sigmoid values of all the confidence scores. The `isNormalized` flag specifies if the data is normalized and is set to `true` for the TensorFlow graph.

### Verifying the output

After the builder is created (see [Building An Engine In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c)) and the engine is serialized (see [Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)), we can perform inference. Steps for deserialization and running inference are outlined in [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c).

The outputs of the SSD network are human interpretable. The post-processing work, such as the final NMS, is done in the `NMS` plugin. The results are organized as tuples of 7. In each tuple, the 7 elements are respectively image ID, object label, confidence score, (`x,y`) coordinates of the lower left corner of the bounding box, and (`x,y`) coordinates of the upper right corner of the bounding box. This information can be drawn in the output PPM image using the `writePPMFileWithBBox` function. The `visualizeThreshold` parameter can be used to control the visualization of objects in the image. It is currently set to 0.5 so the output will display all objects with confidence score of 50% and above.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Concatenation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#concatenation-layer)
The Concatenation layer links together multiple tensors of the same non-channel sizes along the channel dimension.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[Padding layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#padding-layer)
The Padding layer implements spatial zero-padding of tensors along the two innermost dimensions.

[Plugin layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-layer)
Plugin layers are user-defined and provide the ability to extend the functionalities of TensorRT. See [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending) for more details.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.

## Prerequisites

1.  Install the UFF toolkit and graph surgeon; depending on your TensorRT installation method, to install the toolkit and graph surgeon, choose the method you used to install TensorRT for instructions (see [TensorRT Installation Guide: Installing TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing)).

2.  Download the [ssd_inception_v2_coco TensorFlow trained model](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz).

3.  Perform preprocessing on the tensorflow model using the UFF converter.
	1.  Copy the TensorFlow protobuf file (`frozen_inference_graph.pb`) from the downloaded directory in the previous step to the working directory (for example `/usr/src/tensorrt/samples/sampleUffSSD/`).

	2.  Run the following command for the conversion.
	`convert-to-uff frozen_inference_graph.pb -O NMS -p config.py`

		This saves the converted `.uff` file in the same directory as the input with the name `frozen_inference_graph.pb.uff`.

		The `config.py` script specifies the preprocessing operations necessary for the SSD TensorFlow graph. The plugin nodes and plugin parameters used in the `config.py` script should match the registered plugins in TensorRT.

	3.  Copy the converted `.uff` file to the data directory and rename it to `sample_ssd_relu6.uff <TensorRT Install>/data/ssd/sample_ssd_relu6.uff`.

4.  The sample also requires a `labels.txt` file with a list of all labels used to train the model. The labels file for this network is `<TensorRT Install>/data/ssd/ssd_coco_labels.txt`.


## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleUffSSD` directory. The binary named `sample_uff_ssd` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleUffSSD
	make
	```
	Where `<TensorRT root directory>` is where you installed TensorRT.

2. Run the sample to perform object detection and localization.

	To run the sample in FP32 mode:
	`./sample_uff_ssd`

	To run the sample in INT8 mode:
	`./sample_uff_ssd --int8`

	**Note:** To run the network in INT8 mode, refer to `BatchStreamPPM.h` for details on how
calibration can be performed. Currently, we require a file called `list.txt`, with a list of all PPM images for calibration in the `<TensorRT Install>/data/ssd/` folder. The PPM images to be used for calibration can also reside in the same folder.

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_uff_ssd # ./build/x86_64-linux/sample_uff_ssd
	[I] ../data/samples/ssd/sample_ssd_relu6.uff
	[I] Begin parsing model...
	[I] End parsing model...
	[I] Begin building engine...
	I] Num batches 1
	[I] Data Size 270000
	[I] *** deserializing
	[I] Time taken for inference is 4.24733 ms.
	[I] KeepCount 100
	[I] Detected dog in the image 0 (../../data/samples/ssd/dog.ppm) with confidence 89.001 and coordinates (81.7568,23.1155),(295.041,298.62).
	[I] Result stored in dog-0.890010.ppm.
	[I] Detected dog in the image 0 (../../data/samples/ssd/dog.ppm) with confidence 88.0681 and coordinates (1.39267,0),(118.431,237.262).
	[I] Result stored in dog-0.880681.ppm.
	&&&& PASSED TensorRT.sample_uff_ssd # ./build/x86_64-linux/sample_uff_ssd
	```

	This output shows that the sample ran successfully; `PASSED`.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_uff_ssd
Optional Parameters:
  -h, --help Display help information.
  --useDLACore=N    Specify the DLA engine to run on.
  --fp16            Specify to run in fp16 mode.
  --int8            Specify to run in int8 mode.
```  

# Additional resources

The following resources provide a deeper understanding about the TensorFlow SSD network structure:

**Models**
- [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

**Network**
- [ssd_inception_v2_coco_2017_11_17](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz)

**Dataset**
- [MSCOCO dataset](http://cocodataset.org/#home)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

- There might be some precision loss when running the network in INT8 mode causing some objects to go undetected. Our general observation is that >500 images is a good number for calibration purposes.
- On Windows, the Python script convert-to-uff is not available. You can generate the required .uff file on a Linux machine and copy it over in order to run this sample.
