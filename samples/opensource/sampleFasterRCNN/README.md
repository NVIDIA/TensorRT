# Object Detection With Faster R-CNN

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Preprocessing the input](#preprocessing-the-input)
    * [Defining the network](#defining-the-network)
    * [Building the engine](#building-the-engine)
    * [Running the engine](#running-the-engine)
    * [Verifying the output](#verifying-the-output)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleFasterRCNN, uses TensorRT plugins, performs inference, and implements a fused custom layer for end-to-end inferencing of a Faster R-CNN model. Specifically, this sample demonstrates the implementation of a Faster R-CNN network in TensorRT, performs a quick performance test in TensorRT, implements a fused custom layer, and constructs the basis for further optimization, for example using INT8 calibration, user trained network, etc.  The Faster R-CNN network is based on the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).

## How does this sample work?

Faster R-CNN is a fusion of Fast R-CNN and RPN (Region Proposal Network). The latter is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. It can be merged with Fast R-CNN into a single network because it is trained end-to-end along with the Fast R-CNN detection network and thus shares with it the full-image convolutional features, enabling nearly cost-free region proposals. These region proposals will then be used by Fast R-CNN for detection.

Faster R-CNN is faster and more accurate than its predecessors (RCNN, Fast R-CNN) because it allows for an end-to-end inferencing and does not need standalone region proposal algorithms (like selective search in Fast R-CNN) or classification method (like SVM in RCNN).

Specifically, this sample performs the following steps:
- [Preprocessing the input](#preprocessing-the-input)
- [Defining the network](#defining-the-network)
- [Building the engine](#building-the-engine)
- [Running the engine](#running-the-engine)
- [Verifying the output](#verifying-the-output)

The sampleFasterRCNN sample uses a plugin from the TensorRT plugin library to include a fused implementation of Faster R-CNN’s Region Proposal Network (RPN) and ROIPooling layers. These particular layers are from the Faster R-CNN paper and are implemented together as a single plugin called `RPNROIPlugin`. This plugin is registered in the TensorRT Plugin Registry with the name `RPROI_TRT`.

### Preprocessing the input

Faster R-CNN takes 3 channel 375x500 images as input. Since TensorRT does not depend on any computer vision libraries, the images are represented in binary `R`, `G`, and `B` values for each pixels. The format is Portable PixMap (PPM), which is a netpbm color image format. In this format, the `R`, `G`, and `B` values for each pixel are usually represented by a byte of integer (0-255) and they are stored together, pixel by pixel.

However, the authors of Faster R-CNN have trained the network such that the first Convolution layer sees the image data in `B`, `G`, and `R` order. Therefore, you need to reverse the order when the PPM images are being put into the network input buffer.
```
float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
// pixel mean used by the Faster R-CNN's author
float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f }; // also in BGR order
for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
{
	for (int c = 0; c < INPUT_C; ++c)
	{
		// the color image to input should be in BGR order
		for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
        {
            data[i*volImg + c*volChl + j] =  float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
        }
	}
}
```
There is a simple PPM reading function called `readPPMFile`.

**Note:** The `readPPMFile` function will not work correctly if the header of the PPM image contains any annotations starting with `#`.

Furthermore, within the sample there is another function called `writePPMFileWithBBox`, that plots a given bounding box in the image with one-pixel width red lines.

In order to obtain PPM images, you can easily use the command-line tools such as ImageMagick to perform the resizing and conversion from JPEG images.

If you choose to use off-the-shelf image processing libraries to preprocess the inputs, ensure that the TensorRT inference engine sees the input data in the form that it is supposed to.

### Defining the network

The network is defined in a prototxt file which is shipped with the sample and located in the `data/faster-rcnn` directory. The prototxt file is very similar to the one used by the inventors of Faster R-CNN except that the RPN and the ROI pooling layer is fused and replaced by a custom layer named `RPROIFused`.

This sample uses the plugin registry to add the plugin to the network. The Caffe parser adds the plugin object to the network based on the layer name as specified in the Caffe prototxt file, for example, `RPROI`.

### Building the engine

To build the TensorRT engine, see [Building An Engine In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c).

**Note:** In the case of the Faster R-CNN sample, `maxWorkspaceSize` is set to `10 * (2^20)`, namely 10MiB, because there is a need of roughly 6MiB of scratch space for the plugin layer for batch size 5.

After the engine is built, the next steps are to serialize the engine, then run the inference with the deserialized engine. For more information, see [Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c).

### Running the engine

To deserialize the engine, see [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c).

In `sampleFasterRCNN.cpp`, there are two inputs to the inference function:
-   `data` is the image input    
-   `imInfo` is the image information array which stores the number of rows, columns, and the scale for each image in a batch.

and four outputs:
-   `bbox_pred` is the predicted offsets to the heights, widths and center coordinates.    
-   `cls_prob` is the probability associated with each object class of every bounding box.    
-   `rois` is the height, width, and the center coordinates for each bounding box.    
-   `count` is deprecated and can be ignored.

	**Note:** The `count` output was used to specify the number of resulting NMS bounding boxes if the output is not aligned to `nmsMaxOut`. Although it is deprecated, always allocate the engine buffer of size `batchSize * sizeof(int)` for it until it is completely removed from the future version of TensorRT.

### Verifying the output

The outputs of the Faster R-CNN network need to be post-processed in order to obtain human interpretable results.

First, because the bounding boxes are now represented by the offsets to the center, height, and width, they need to be unscaled back to the raw image space by dividing the scale defined in the `imInfo` (image info).

Ensure you apply the inverse transformation on the bounding boxes and clip the resulting coordinates so that they do not go beyond the image boundaries.

Lastly, overlapped predictions have to be removed by the non-maximum suppression algorithm. The post-processing codes are defined within the CPU because they are neither compute intensive nor memory intensive.

After all of the above work, the bounding boxes are available in terms of the class number, the confidence score (probability), and four coordinates. They are drawn in the output PPM images using the `writePPMFileWithBBox` function.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Plugin (RPROI) layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-layer)
Plugin layers are user-defined and provide the ability to extend the functionalities of TensorRT. See  [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending)  for more details.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Running the sample

1.  Download the [faster_rcnn_models.tgz](https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz) dataset.
    ```
    cd <TensorRT directory>
    wget --no-check-certificate https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0 -O data/faster-rcnn/faster-rcnn.tgz
    ```

2.  Extract the dataset into the `data/faster-rcnn` directory.
	```
    tar zxvf data/faster-rcnn/faster-rcnn.tgz -C data/faster-rcnn --strip-components=1 --exclude=ZF_*
	```

3.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleFasterRCNN` directory. The binary named `sample_fasterRCNN` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples
	make
	```
	Where `<TensorRT root directory>` is where you installed TensorRT.

4.  Run the sample to perform inference.
	`./sample_fasterRCNN`

5.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	Sample output
	[I] Detected car in 000456.ppm with confidence 99.0063%  (Result stored in car-0.990063.ppm).
	[I] Detected person in 000456.ppm with confidence 97.4725%  (Result stored in person-0.974725.ppm).
	[I] Detected cat in 000542.ppm with confidence 99.1191%  (Result stored in cat-0.991191.ppm).
	[I] Detected dog in 001150.ppm with confidence 99.9603%  (Result stored in dog-0.999603.ppm).
	[I] Detected dog in 001763.ppm with confidence 99.7705%  (Result stored in dog-0.997705.ppm).
	[I] Detected horse in 004545.ppm with confidence 99.467%  (Result stored in horse-0.994670.ppm).
	&&&& PASSED TensorRT.sample_fasterRCNN # ./build/x86_64-linux/sample_fasterRCNN
	```
    This output shows that the sample ran successfully; `PASSED`.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. The following example output is printed when running the sample:
```
./sample_fasterRCNN --help
Usage: ./build/x86_64-linux/sample_fasterRCNN
Optional Parameters:
  -h, --help        Display help information.
  --useDLACore=N    Specify the DLA engine to run on.
```


# Additional resources

The following resources provide a deeper understanding about object detection with Faster R-CNN:

**Faster R-CNN**
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

**Documentation**
- [TensorRT Sample Support Guide: sampleFasterRCNN](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#fasterrcnn_sample)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
