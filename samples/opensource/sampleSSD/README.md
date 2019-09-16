# Object Detection With SSD

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Preprocessing the input](#preprocessing-the-input)
    * [Defining the network](#defining-the-network)
    * [Building the engine](#building-the-engine)
    * [Verifying the output](#verifying-the-output)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
    * [TensorRT plugin layers in SSD](#tensorrt-plugin-layers-in-ssd)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sample SSD, is based on the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)  paper. The SSD network performs the task of object detection and localization in a single forward pass of the network. This network is built using the VGG network as a backbone and trained using [PASCAL VOC 2007+ 2012](https://github.com/weiliu89/caffe/tree/ssd) datasets.

Unlike Faster R-CNN, SSD completely eliminates proposal generation and subsequent pixel or feature resampling stages and encapsulates all computation in a single network. This makes SSD straightforward to integrate into systems that require a detection component.

## How does this sample work?

This sample pre-processes the input to the SSD network and performs inference on the SSD network in TensorRT, using plugins to run layers that are not natively supported in TensorRT. Additionally, the sample can also be run in INT8 mode for which it first performs INT8 calibration and then does inference int INT8.

Specifically, this sample:
-  [Preprocesses the input](#preprocessing-the-input)
-  [Defines the network](#defining-the-network)
-  [Builds the engine](#building-the-engine)
-  [Verifies the output](#verifying-the-output)

### Preprocessing the input

The input to the SSD network in this sample is a RGB 300x300 image. The image format is Portable PixMap (PPM), which is a netpbm color image format. In this format, the `R`, `G`, and `B` values for each pixel are represented by a byte of integer (0-255) and they are stored together, pixel by pixel.

The authors of SSD have trained the network such that the first Convolution layer sees the image data in `B`, `G`, and `R` order. Therefore, the channel order needs to be changed when the PPM image is being put into the network’s input buffer.

```
float pixelMean[3]{ 104.0f, 117.0f, 123.0f }; // also in BGR order
float* data = new float[N * kINPUT_C * kINPUT_H * kINPUT_W];
     for (int i = 0, volImg = kINPUT_C * kINPUT_H * kINPUT_W; i < N; ++i)
    {
           for (int c = 0; c < kINPUT_C; ++c)
           {
                  // the color image to input should be in BGR order
                  for (unsigned j = 0, volChl = kINPUT_H * kINPUT_W; j < volChl; ++j)
                  {
                        data[i * volImg + c * volChl + j] = float(ppms[i].buffer[j * kINPUT_C + 2 - c]) - pixelMean[c];
                  }
           }
    }
```

The `readPPMFile` and `writePPMFileWithBBox` functions read a PPM image and produce output images with red colored bounding boxes respectively.

**Note:** The `readPPMFile` function will not work correctly if the header of the PPM image contains any annotations starting with `#`.

### Defining the network

The network is defined in a prototxt file which is shipped with the sample and located in the `data/ssd` directory. The original prototxt file provided by the authors is modified and included in the TensorRT in-built plugin layers in the prototxt file.

The built-in plugin layers used in sampleSSD are Normalize, PriorBox, and DetectionOutput. The corresponding registered plugins for these layers are `Normalize_TRT`, `PriorBox_TRT` and `NMS_TRT`.

To initialize and register these TensorRT plugins to the plugin registry, the `initLibNvInferPlugins` method is used. After registering the plugins and while parsing the prototxt file, the NvCaffeParser creates plugins for the layers based on the parameters that were provided in the prototxt file automatically. The details about each parameter is provided in the `README.md` and can be modified similar to the Caffe Layer parameter.

### Building the engine

The sampleSSD sample builds a network based on a Caffe model and network description. For details on importing a Caffe model, see [Importing A Caffe Model Using The C++ Parser API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_caffe_c). The SSD network has few non-natively supported layers which are implemented as plugins in TensorRT. The Caffe parser can create plugins for these layers internally using the plugin registry.

This sample can run in FP16 and INT8 modes based on the user input. For more details, see [INT8 Calibration Using C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#optimizing_int8_c) and [Enabling FP16 Inference Using C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#enable_fp16_c). The sample selects the entropy calibrator as a default choice. The `CalibrationMode` parameter in the sample code needs to be set to `0` to switch to the Legacy calibrator.

For details on how to build the TensorRT engine, see [Building An Engine In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c). After the engine is built, the next steps are to serialize the engine and run the inference with the deserialized engine. For more information about these steps, see [Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c).

### Verifying the output

After deserializing the engine, you can perform inference. To perform inference, see [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c).

In sampleSSD, there is a single input:
-  `data`, namely the image input  

And 2 outputs:
-  `detectionOut` is the detection array, containing the image ID, label, confidence, 4 coordinates
-  `keepCount` is the number of valid detections
 
The outputs of the SSD network are directly human interpretable. The results are organized as tuples of 7. In each tuple, the 7 elements are:
-   image ID
-   object label
-   confidence score
-   (x,y) coordinates of the lower left corner of the bounding box
-   (x,y) coordinates of the upper right corner of the bounding box
  
This information can be drawn in the output PPM image using the `writePPMFileWithBBox` function. The `kVISUAL_THRESHOLD` parameter can be used to control the visualization of objects in the image. It is currently set to 0.6, therefore, the output will display all objects with confidence score of 60% and above.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`. 

[Concatenation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#concatenation-layer)
The Concatenation layer links together multiple tensors of the same non-channel sizes along the channel dimension.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[Plugin layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-layer)
Plugin layers are user-defined and provide the ability to extend the functionalities of TensorRT. See [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending) for more details.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

### TensorRT plugin layers in SSD

sampleSSD has three plugin layers; Normalize, PriorBox and DetectionOutput. The details about each layer and its parameters is shown below in `caffe.proto` format.

```
message LayerParameter {
  optional DetectionOutputParameter detection_output_param = 881;
  optional NormalizeParameter norm_param = 882;
  optional PriorBoxParameter prior_box_param ==883;
}

// Message that stores parameters used by Normalize layer
NormalizeParameter {
  optional bool across_spatial = 1 [default = true];
  // Initial value of scale. Default is 1.0
  optional FillerParameter scale_filler = 2;
  // Whether or not scale parameters are shared across channels.
  optional bool channel_shared = 3 [default = true];
  // Epsilon for not dividing by zero while normalizing variance
  optional float eps = 4 [default = 1e-10];
}

// Message that stores parameters used by PriorBoxLayer
message PriorBoxParameter {
  // Encode/decode type.
  enum CodeType {
    CORNER = 1;
    CENTER_SIZE = 2;
    CORNER_SIZE = 3;
  }
  // Minimum box size (in pixels). Required!
  repeated float min_size = 1;
  // Maximum box size (in pixels). Required!
  repeated float max_size = 2;
  // Various aspect ratios. Duplicate ratios will be ignored.
  // If none is provided, we use default ratio 1.
  repeated float aspect_ratio = 3;
  // If true, will flip each aspect ratio.
  // For example, if there is aspect ratio "r",
  // we will generate aspect ratio "1.0/r" as well.
  optional bool flip = 4 [default = true];
  // If true, will clip the prior so that it is within [0, 1]
  optional bool clip = 5 [default = false];
  // Variance for adjusting the prior bboxes.
  repeated float variance = 6;
  // By default, we calculate img_height, img_width, step_x, step_y based on
  // bottom[0] (feat) and bottom[1] (img). Unless these values are explicitly
  // provided.
  // Explicitly provide the img_size.
  optional uint32 img_size = 7;
  // Either img_size or img_h/img_w should be specified; not both.
  optional uint32 img_h = 8;
  optional uint32 img_w = 9;

  // Explicitly provide the step size.
  optional float step = 10;
  // Either step or step_h/step_w should be specified; not both.
  optional float step_h = 11;
  optional float step_w = 12;

  // Offset to the top left corner of each cell.
  optional float offset = 13 [default = 0.5];
}

message NonMaximumSuppressionParameter {
  // Threshold to be used in NMS.
  optional float nms_threshold = 1 [default = 0.3];
  // Maximum number of results to be kept.
  optional int32 top_k = 2;
  // Parameter for adaptive NMS.
  optional float eta = 3 [default = 1.0];
}

// Message that stores parameters used by DetectionOutputLayer
message DetectionOutputParameter {
  // Number of classes to be predicted. Required!
  optional uint32 num_classes = 1;
  // If true, bounding box are shared among different classes.
  optional bool share_location = 2 [default = true];
  // Background label id. If there is no background class,
  // set it as -1.
  optional int32 background_label_id = 3 [default = 0];
  // Parameters used for NMS.
  optional NonMaximumSuppressionParameter nms_param = 4;

  // Type of coding method for bbox.
  optional PriorBoxParameter.CodeType code_type = 5 [default = CORNER];
  // If true, variance is encoded in target; otherwise we need to adjust the
  // predicted offset accordingly.
  optional bool variance_encoded_in_target = 6 [default = false];
  // Number of total bboxes to be kept per image after nms step.
  // -1 means keeping all bboxes after nms step.
  optional int32 keep_top_k = 7 [default = -1];
  // Only consider detections whose confidences are larger than a threshold.
  // If not provided, consider all boxes.
  optional float confidence_threshold = 8;
  // If true, visualize the detection results.
  optional bool visualize = 9 [default = false];
  // The threshold used to visualize the detection results.
}
```

## Prerequisites

Due to the size of the SSD Caffe model, it is not included in the product bundle. Before you can run the sample, you’ll need to download the model, perform some configuration, and generate INT8 calibration batches.

1.  Download [models_VGGNet_VOC0712_SSD_300x300.tar.gz](https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view).

2.  Extract the contents.
    `tar xvf models_VGGNet_VOC0712_SSD_300x300.tar.gz`
    
    1. Generate MD5 hash and compare against the reference below:
        `md5sum models_VGGNet_VOC0712_SSD_300x300.tar.gz`

        If the model is correct, you should see the following MD5 hash output:
        `9a795fc161fff2e8f3aed07f4d488faf models_VGGNet_VOC0712_SSD_300x300.tar.gz`

    2. Edit the `deploy.prototxt` file and change all the Flatten layers to Reshape operations with the following parameters:
        ```
        reshape_param {
            shape {
                dim: 0
                dim: -1
                dim: 1
                dim: 1
            }
        }
        ```

    3. Update the `detection_out` layer to add the `keep_count` output as expected by the TensorRT DetectionOutput Plugin.
    `top: "keep_count"`

    4. Rename the updated `deploy.prototxt` file to `ssd.prototxt` and move the file to the `data` directory.
    `mv ssd.prototxt <TensorRT_Install_Directory>/data/ssd`

    5. Move the `caffemodel` file to the `data` directory.
        ```
        mv VGG_VOC0712_SSD_300x300_iter_120000.caffemodel <TensorRT_Install_Directory>/data/ssd
        ```

3.  Generate the INT8 calibration batches.
    1.  Install Pillow.
        -   For Python 2 users, run:
             `python2 -m pip install Pillow`

        -   For Python 3 users, run:
            `python3 -m pip install Pillow`

    2.  Generate the INT8 batches.
        `prepareINT8CalibrationBatches.sh`

        The script selects 500 random JPEG images from the PASCAL VOC dataset and converts them to PPM images. These 500 PPM images are used to generate INT8 calibration batches.

        **Note:** Do not move the batch files from the `<TensorRT_Install_Directory>/data/ssd/batches` directory.

        If you want to use a different dataset to generate INT8 batches, use the `batchPrepare.py` script and place the batch files in the `<TensorRT_Install_Directory>/data/ssd/batches` directory.

## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleSSD` directory. The binary named `sample_ssd` will be created in the `<TensorRT root directory>/bin` directory.
    ```
    cd <TensorRT root directory>/samples/sampleSSD
    make
    ```
    Where `<TensorRT root directory>` is where you installed TensorRT.
    
2. Run the sample to perform inference on the digit:
    ```
    ./sample_ssd [-h] [--fp16] [--int8]
    ```
3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    &&&& RUNNING TensorRT.sample_ssd # ./sample_ssd
    [I] Begin parsing model...
    [I] FP32 mode running...
    [I] End parsing model...
    [I] Begin building engine...
    [I] [TRT] Detected 1 input and 2 output network tensors.
    [I] End building engine...
    [I] *** deserializing
    [I] Image name:../data/samples/ssd/bus.ppm, Label: car, confidence: 96.0587 xmin: 4.14486 ymin: 117.443 xmax: 244.102 ymax: 241.829
    &&&& PASSED TensorRT.sample_ssd # ./build/x86_64-linux/sample_ssd
    ```

    This output shows that the sample ran successfully; `PASSED`.
 

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./build/x86_64-linux/sample_ssd
Optional Parameters:
    -h, --help      Display help information.
    --useDLACore=N  Specify the DLA engine to run on.
    --fp16          Specify to run in fp16 mode.
    --int8          Specify to run in int8 mode.
```

# Additional resources

The following resources provide a deeper understanding about how the SSD model works:

**Models**
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

**Dataset**
- [PASCAL VOC 2007+ 2012](https://github.com/weiliu89/caffe/tree/ssd)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

- On Windows, the INT8 calibration script is not supported natively. You can generate the INT8 batches on a Linux machine and copy them over in order to run sample_ssd in INT8 mode.
