# cropAndResize Plugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `cropAndResizePlugin` performs object detection for the Faster R-CNN model. This plugin is included in TensorRT and used in [sampleUffFasterRCNN] to perform inference.

`cropAndResizePlugin` implements the TensorFlow style of ROIPooling(a.k.a. CropAndResize). It crops multiple region of interests(ROIs) from the input image with given ROI coordinates and then (bilinearly) resizes the cropped patches to a target spatial(width and height) size. 

Note this implementation is different from the original Caffe implement of ROIPooling. Also, this implementation doesn't fuse the ROIPooling layer and the Proposal layer into a single layer as the `nvFasterRCNN` plugin does.

This plugin is optimized for the above steps and it allows you to do Faster R-CNN inference in TensorRT.


### Structure

The `cropAndResizePlugin` takes two inputs; `feature_maps` and `rois`.

`feature_maps`
Input featuremaps where the patches are going to be cropped from. feature_maps has shape `[N, C, H, W]` where `N` is the batch size, `C` is the channel number, `H` is the height, `W` is the width. The feature map contains the features that we want to crop patches from and do classification after that to determine the class of each patch(potential object) and the bounding boxes deltas for that object.


`rois`
Coordinates of region of interest bounding boxes, normalized to the range of (0, 1). `rois` has shape `[N, B, 4, 1]`, where `N` is the batch size, `B` is the maximum number of region of interest bounding boxes per image(featuremap), and `4` represents and region of interest bounding box coordinates `[y_1, x_1, y_2, x_2]`. Here, `x_1` and `y_1` are the coordinates of bounding box at the top-left corner, and `x_2` and `y_2` are the coordinates of bounding box at the bottom-right corner.


The `cropAndResizePlugin` generates the following one outputs:

`pfmap`
ROI pooled feature map corresponding to the region of interest. `pfmap` has shape `[N, B, C, crop_height, crop_width]` where `N` is the batch size, `B` is the maximum number of region of interest bounding boxes, `C` is the number of channels in the feature map, `crop_height` is the height of ROI pooled feature map, and `crop_width` is the width of ROI pooled feature map.


The ROI pooling step uses the inferred region of interest bounding boxes information to extract its corresponding regions on feature map, and does POI pooling to get uniformly shaped features from different shaped region of interest bounding boxes.

## Parameters

`cropAndResizePlugin` has plugin creator class `cropAndResizePluginCreator` and plugin class `CropAndResizePlugin`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int`     |`crop_width`                |The height of the output in pixels after ROI pooling on the feature map.
|`int`     |`crop_height`               |The width of the output in pixels after ROI pooling on the feature map.

## Additional resources

The following resources provide a deeper understanding of the `cropAndResizePlugin` plugin:

**Networks:**
-   [Faster R-CNN](https://arxiv.org/abs/1506.01497)

**Documentation:**
-   [Original ROI Pooling Definition from Fast R-CNN](https://arxiv.org/abs/1504.08083)
-   [CropAndResize Op in TensorFlow](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/crop-and-resize)
-   [tf.image.crop_and_resize API](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize) 

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
