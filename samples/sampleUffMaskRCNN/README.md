# Object Detection And Instance Segmentations With A TensorFlow Mask R-CNN Network

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Generating UFF Model](#generating-uff-model)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleUffMaskRCNN, performs inference on the Mask R-CNN network in TensorRT. Mask R-CNN is based on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) paper which performs the task of object detection and object mask predictions on a target image. This sample’s model is based on the Keras implementation of Mask R-CNN and its training framework can be found in the [Mask R-CNN Github repository](https://github.com/matterport/Mask_RCNN). We have verified that the pre-trained Keras model (with backbone ResNet101 + FPN and dataset coco) provided in the [v2.0](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0) release can be converted to UFF and consumed by this sample. And, it is also feasible to deploy your customized Mask R-CNN model trained with specific backbone and datasets.

## How does this sample work?

This sample makes use of TensorRT plugins to run the Mask R-CNN model. To use these plugins, the Keras model should be converted to Tensorflow `.pb` model. Then this `.pb` model needs to be preprocessed and converted to the UFF model with the help of GraphSurgeon and the UFF utility.

The main components of this network are the `ResizeNearest`, `ProposalLayer`, `PyramidROIAlign`, `DetectionLayer` and `SpecialSlice`.

- `ResizeNearest` - Nearest neighbor interpolation for resizing features. This works for the FPN (Feature Pyramid Network) module.

- `ProposalLayer` - Generate the first stage's proposals based on anchors and RPN's (Region Proposal Network) outputs (scores, bbox_deltas).

- `PyramidROIAlign` - Crop and resize the feature of ROIs (first stage's proposals) from the corresponding feature layer.

- `DetectionLayer` - Refine the first stage's proposals to produce final detections.

- `SpecialSlice` - A workaround plugin to slice detection output [y1, x1, y2, x2, class_id, score] to [y1, x1, y2 , x2] for data with more than one index dimensions (batch_idx, proposal_idx, detections(y1, x1, y2, x2)).


### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[Deconvolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#deconvolution-layer)
The IDeconvolutionLayer computes a 2D (channel, height, and width) deconvolution, with or without bias.

[Padding layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#padding-layer)
The IPaddingLayer implements spatial zero-padding of tensors along the two innermost dimensions.

[Plugin layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-layer)
Plugin layers are user-defined and provide the ability to extend the functionalities of TensorRT. See [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending) for more details.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.


## Generating UFF model

1. Install [TensorFlow 1.15](https://www.tensorflow.org/install/pip) or launch the NVIDIA Tensorflow 1.x container in a separate terminal for this step.

2. Install the required packages alongwith UFF toolkit and graph surgeon
    ```bash
    cd $TRT_OSSPATH/samples/opensource/sampleUffMaskRCNN/converted/
    pip3 install -r requirements.txt
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com uff
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com graphsurgeon
    ```

3.  Modify the `conv2d_transpose` conversion function in UFF, for example `/usr/lib/python3.6/dist-packages/uff/converters/tensorflow/converter_functions.py`.
    ```bash
    uff_graph.conv_transpose(
        inputs[0], inputs[2], inputs[1],
        strides, padding,
        dilation=None, number_groups=number_groups,
        left_format=lhs_fmt, right_format=rhs_fmt,
        name=name, fields=fields
        )
    ```

4.  Download the Mask R-CNN repo and export to `PYTHONPATH`.
    ```bash
    git clone https://github.com/matterport/Mask_RCNN.git
    export PYTHONPATH=$PYTHONPATH:$PWD/Mask_RCNN
    ```

5.  Apply the patch into Mask R-CNN repo to update the model from NHWC to NCHW.
    ```bash
    cd Mask_RCNN
    git checkout 3deaec5
    patch -p1 < ../0001-Update-the-Mask_RCNN-model-from-NHWC-to-NCHW.patch
    cd -
    ```

6.  Download the pre-trained Keras model
    ```bash
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    ```

    **Note:** The md5sum of model file is e98aaff6f99e307b5e2a8a3ff741a518.

7.  Convert the h5 model to the UFF model and copy it to your data folder
    ```bash
    python3 mrcnn_to_trt_single.py -w mask_rcnn_coco.h5 -o mrcnn_nchw.uff -p ./config.py
    ```

8.  Populate your `$TRT_DATADIR/maskrcnn` folder with the test images.
    ```bash
    cp $TRT_DATADIR/faster-rcnn/001763.ppm $TRT_DATADIR/maskrcnn
    cp $TRT_DATADIR/faster-rcnn/004545.ppm $TRT_DATADIR/maskrcnn
    ```

## Running the sample

Switch back to the TensorRT container/environment for running the sample.

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to perform object detection and object mask prediction.

    ```bash
    ./sample_uff_maskRCNN --datadir=<path/to/data> --fp16 --batch N
    ```

    For example:
    ```bash
    sample_uff_maskRCNN --datadir $TRT_DATADIR/maskrcnn --fp16
    ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    [I] Detected dog in../../data/001763.ppm with confidence 99.9064 and coordinates (257.351, 14.2238, 489.272, 364.817)
    [I] Detected dog in../../data/001763.ppm with confidence 99.8484 and coordinates (14.3269, 52.0974, 320.913, 363.364)
    [I] The results are stored in current directory: 0.ppm
    [I] Detected horse in../../data/004545.ppm with confidence 99.9796 and coordinates (164.81, 22.6816, 386.512, 308.955)
    [I] Detected bottle in../../data/004545.ppm with confidence 98.5529 and coordinates (218.719, 237.04, 229.382, 261.205)
    [I] The results are stored in current directory: 1.ppm
    &&&& PASSED TensorRT.sample_maskrcnn # ../build/cmake/out/sample_uff_mask_rcnn -d ../../data/
    ```
    This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


## Additional resources

The following resources provide a deeper understanding about sampleUffMaskRCNN.

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

July 2019
This is the first release of the `README.md` file and sample.


## Known issues
There are no known issues in this sample.
