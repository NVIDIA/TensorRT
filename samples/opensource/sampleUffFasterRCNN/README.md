# Object Detection With A TensorFlow FasterRCNN Network

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Processing the input graph](#processing-the-input-graph)
    * [Preparing the data](#preparing-the-data)
    * [sampleUffFasterRCNN plugins](#sampleufffasterrcnn-plugins)
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
This sample, sampleUffFasterRCNN, serves as a demo on how to use a TensorFlow based Faster-RCNN model. It uses the `Proposal` and `CropAndResize` TensorRT plugins to implement the proposal layer and ROIPooling layer as custom layers since TensorRT has no native support for them.

## How does this sample work?

The UFF Faster R-CNN network performs the task of object detection and localization in a single forward pass of the network. The Faster R-CNN network was trained on the ResNet-10 backbone (feature extractor) to detect 4 classes of objects: `Automobile`, `Roadsign`, `Bicycle` and `Person` along with the `background` class(nothing).

This sample makes use of TensorRT plugins to run the UFF Faster R-CNN network. To use these plugins, the TensorFlow graph needs to be preprocessed, and we use the GraphSurgeon utility to do this.

The main components of this network are the Image Preprocessor, FeatureExtractor, Region Proposal Network (RPN), Proposal, ROIPooling (CropAndResize), Classifier and Postprocessor.

**Image Preprocessor**
The image preprocessor step of the graph is responsible for resizing the image. The image is resized to a 3x272x480(CHW) size tensor. This step also performs per-channel mean value subtraction of the images. After preprocessing, the input images's channel order is `BGR` instead of `RGB`.

**FeatureExtractor**
The FeatureExtractor portion of the graph runs the ResNet10 network on the preprocessed image. The feature maps generated are used by the RPN layer and the Proposal layer to generate the Regions of Interest(ROIs) that may contain objects. As a second branch, the feature maps are also used in the ROIPooling (or more precisely, CropAndResize layer) to crop out the patches from the feature maps with the specified ROIs output from Proposal layer.

In this network, the feature maps come from an intermediate layer's output in the ResNet-10 backbone. The intermediate layer has a cumulative stride of 16.

**Region Proposal Network (RPN)**
The RPN takes the feature maps from the stride-16 backbone and append a small Convolutional Neural Network (CNN) head after it to detect whether a specific region of the image has object or not. It also outputs a rough coordinates of the candidate object.

**Proposal**
The Proposal layer takes the input of the RPN and do some refinement of the candidate boxes from the RPN. The refinement includes taking the top boxes that has the highest confidence and do NMS (non-maximum suppression) against them. Finally, taking the top boxes again according to their confidence after NMS operation.

This operation is implemented in the `Proposal` plugin as a TensorRT plugin.

**CropAndResize**
The CropAndResize layer performs a TensorFlow implementation of the original ROIPooling layer in the Caffe implementation. The CropAndResize layer resizes the ROIs from the Proposal layer to a common target size and the output results are followed by a classifier to distinguish which class the ROI belongs to. The difference between the CropAndResize operation and the ROIPooling operation is the former use bilinear interpolation while the latter uses pooling.

This operation is implemented in the `CropAndResize` plugin as a TensorRT plugin.

**Classifier**
The classifier is a small network that takes the output of the CropAndResize layer as input and distinguish which class the ROI belongs to. Apart from that, it also gives a delta coordinates to refine the coordinates output from the RPN layer.

**Postprocessor**
The Postprocessor applies the delta values from the classifier output to the coordinates from the RPN output and do NMS after that to get the final detection results.

Specifically, this sample performs the following steps:
- [Processing the input graph](#processing-the-input-graph)
- [Preparing the data](#preparing-the-data)
- [sampleUffFasterRCNN plugins](#sampleufffasterrcnn-plugins)
- [Verifying the output](#verifying-the-output)


### Processing the input graph

The TensorFlow FasterRCNN graph has some operations that are currently not supported in TensorRT. Using a preprocessor on the graph, we can combine multiple operations in the graph into a single custom operation which can be implemented as a plugin layer in TensorRT. Currently, the preprocessor provides the ability to stitch all nodes within a namespace into one custom node.
  
To use the preprocessor, the `convert-to-uff` utility should be called with a `-p` flag and a config file. The config script should also include attributes for all custom plugins which will be embedded in the generated `.uff` file. Current sample script for UFF Faster R-CNN is located in `config.py` in this sample.

### Preparing the data

The generated network has an input node called `input_1`, and the output nodes's names are `dense_class/Softmax`, `dense_regress/BiasAdd` and `proposal`. These nodes are registered by the UFF Parser in the sample.

The input to the UFF Faster R-CNN network in this sample is 3 channel 480x272 images. In the sample, we subtract the per-channel mean values for the input images.

Since TensorRT does not depend on any computer vision libraries, the images are represented in binary R, G, and B values for each pixel. The format is Portable PixMap (PPM), which is a netpbm color image format. In this format, the R, G, and B values for each pixel are represented by a byte of integer (0-255) and they are stored together, pixel by pixel. The channel order of the input image is actually BGR instead of RGB due to implementation.
  
There is a simple PPM reading function called `readPPMFile`.

### sampleUffFasterRCNN plugins

Details about how to create TensorRT plugins can be found in [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending).

The `config.py` defined for the`convert-to-uff` command should have the custom layers mapped to the plugin names in TensorRT by modifying the op field. The names of the plugin parameters should also exactly match those expected by the TensorRT plugins.

If the `config.py` is defined as above, the NvUffParser will be able to parse the network and call the appropriate plugins with the correct parameters.

Details about some of the plugin layers implemented for UFF Faster R-CNN in TensorRT are given below.

**CropAndResize plugin**
The `CropAndResize` plugin crops out patches from the feature maps according to the ROI coordinates from the Proposal layer and resizes them to a common target size, for example, 7x7. The output tensor is used as input of the classifier that follows `CropAndResize` plugin.

**Proposal plugin**
The `Proposal` plugin does the refinement of the candidate boxes from the RPN. The refinement includes selecting the top boxes according to their confidence, doing NMS and finally selecting the top boxes that has the highest confidence after NMS.

### Verifying the output

After the builder is created (see [Building An Engine In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c)) and the engine is serialized (see [Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)), we can perform inference. Steps for deserialization and running inference are outlined in [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c). The outputs of the UFF FasterRCNN network are human interpretable. The results are visualized by drawing the bounding boxes on the images.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Padding layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#padding-layer)
The IPaddingLayer implements spatial zero-padding of tensors along the two innermost dimensions.

[Plugin layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-layer)
Plugin layers are user-defined and provide the ability to extend the functionalities of TensorRT. See [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending) for more details.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Prerequisites
1. Install the UFF toolkit and graph surgeon; depending on your TensorRT installation method, to install the toolkit and graph surgeon, choose the method you used to install TensorRT for instructions (see [TensorRT Installation Guide: Installing TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing)).  
    
2. We provide a bash script to download the model as well as other data required for this sample: `./download_model.sh`.

   The model is downloaded and unzipped in the directory `uff_faster_rcnn` and the `pb` model is `uff_faster_rcnn/faster_rcnn.pb`.

   Along with the `pb` mode there are some PPM images and a `list.txt` in the directory. These PPM images are the test images used in this sample. The `list.txt` is used in the INT8 mode for listing the image names used in INT8 calibration step in TensorRT.

3. Perform preprocessing on the TensorFlow model using the UFF converter.  
	1.  Copy the TensorFlow protobuf file (`faster_rcnn.pb`) from the downloaded directory in the previous step to the working directory (for example `/usr/src/tensorrt/data/faster-rcnn-uff`).

	2.  Patch the UFF converter.

		Apply a patch to the UFF converter to fix an issue with the Softmax layer in the UFF package. Let `UFF_ROOT` denotes the root directory of the Python UFF package, for example, `/usr/lib/python2.7/dist-packages/uff`

		Then, apply the patch with the following command:
		`patch UFF_ROOT/converters/tensorflow/converter_functions.py < fix_sofmax.patch` 

		The patch file `fix_softmax.patch` is generated using the UFF package version 0.6.3 in TensorRT 5.1 GA. Ensure your UFF package version is also 0.6.3 before applying the patch. For TensorRT 6.0, feel free to ignore this since it should already be fixed.

	3.  Run the following command for the conversion.
		```
		convert-to-uff -p config.py -O dense_class/Softmax -O dense_regress/BiasAdd -O proposal faster_rcnn.pb  
		```  
		This saves the converted `.uff` file in the same directory as the input with the name `faster_rcnn.uff`.  
  
		The `config.py` script specifies the preprocessing operations necessary for the UFF Faster R-CNN TensorFlow graph. The plugin nodes and plugin parameters used in the `config.py` script should match the registered plugins in TensorRT.  

4. The sample also requires a `list.txt` file with a list of all the calibration images (basename, without suffix) when running in INT8 mode. Copy the `list.txt` to the same directory that contains the `pb` model.  

5. Copy the PPM images in the data directory the same directory that contains the `pb` model.


## Running the sample

1. Following the [top level guide](../../../README.md) to build the OSS samples(including this sample, of course). The binary named `sample_uff_fasterRCNN` will be created in the `build/cmake/out` directory.

2. Run the sample to perform object detection and localization.

   To run the sample in FP32 mode:
	```
	./sample_uff_fasterRCNN --datadir /data/uff_faster_rcnn -W 480 -H 272 -I 2016_1111_185016_003_00001_night_000441.ppm
	```

   To run the sample in INT8 mode:
	```
	./sample_uff_fasterRCNN --datadir /data/uff_faster_rcnn -i -W 480 -H 272 -I 2016_1111_185016_003_00001_night_000441.ppm
	```
	
3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
    Detected Automobile in 2016_1111_185016_003_00001_night_000441.ppm with confidence 99.9734%
    Detected Automobile in 2016_1111_185016_003_00001_night_000441.ppm with confidence 99.9259%
    Detected Automobile in 2016_1111_185016_003_00001_night_000441.ppm with confidence 98.7359%
    Detected Automobile in 2016_1111_185016_003_00001_night_000441.ppm with confidence 92.4371%
    Detected Automobile in 2016_1111_185016_003_00001_night_000441.ppm with confidence 89.7888%
	```
   This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.  
```
Usage: ./sample_uff_fasterRCNN --datadir /data/uff_faster_rcnn -h
--help[-h] Display help information
--datadir[-d] Specify path to a data directory, overriding the default. This option can be repeated to add multiple directories. If the option is unspecified, the default is to search data/faster-rcnn/ and data/samples/faster-rcnn/.
--useDLACore[-u] Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--fp16[-f] Specify to run in fp16 mode.
--int8[-i] Specify to run in int8 mode.
--inputWidth[-W] Specify the input width of the model.
--inputHeight[-H] Specify the input height of the model.
--batchSize[-B] Specify the batch size for inference.
--profile[-p] Whether to do per-layer profiling.
--repeat[-r] Specify the repeat number to execute the TRT context, used to smooth the profiling time.
--inputImages[-I] Specify the input images for inference.
--saveEngine[-s] Path to save engine.
--loadEngine[-l] Path to load engine.
```

# Additional resources

The following resources provide a deeper understanding about sampleUffFasterRCNN.

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

July 2019
This is the first release of the `README.md` file and sample.


# Known issues

There are no known issues in this sample.
