# proposal Plugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `proposalPlugin` performs object detection for the Faster R-CNN model. This plugin is included in TensorRT and used in [sampleUffFasterRCNN] to perform inference.

`proposalPlugin` decodes predicted bounding boxes, extracts their corresponding objectness score, extracts region of interest from predicted bounding boxes using non maximum suppression, for downstreaming ROIPooling tasks.

This plugin is optimized for the above steps and it allows you to do Faster R-CNN inference in TensorRT.


### Structure

The `proposalPlugin` takes two inputs; `scores` and `deltas`.

`scores`
Bounding box (region proposal) objectness scores. scores has shape `[N, A, H, W]` where `N` is the batch size, `A` is the number anchor boxes per pixel on the feature map, `H` is the height of feature map, and `W` is the width of feature map. The second dimension is `A` because UFF Faster R-CNN uses Sigmoid activation to classify the objectness for each bounding box.

`deltas`
Predicted bounding box offsets. `deltas` has shape `[N, A x 4, H, W]` where `N` is the batch size, `A` is the number anchor boxes per pixel on the feature map, `H` is the height of feature map, and `W` is the width of feature map. The second dimension is `A x 4` because each anchor box or bounding box consists of four parameters.


The `proposalPlugin` generates the following output:

`rois`
Coordinates of region of interest bounding boxes on the original input image. `rois` has shape `[N, B, 4, 1]`, where `N` is the batch size, `B` is the maximum number of region of interest bounding boxes, and `4` represents and region of interest bounding box coordinates `[y_1, x_1, y_2, x_2]`. Here, `x_1` and `y_1` are the coordinates of bounding box at the top-left corner, and `x_2` and `y_2` are the coordinates of bounding box at the bottom-right corner. The four coordinates are normalized to the range of (0, 1).


`proposalPlugin` essentially does region proposal inference.

The proposal inference step includes three steps: extract objectness scores from `scores` input, decode predicted bounding box from `deltas` input, non-maximum suppression and get the region of interest bounding boxes using the extracted objectness scores and the decoded bounding boxes.


## Parameters

`proposalPlugin` has plugin creator class `ProposalPluginCreator` and plugin class `proposalPlugin`.

The plugin parameters are defined below and consists of the following attributes:


| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int`     |`input_height`                |The height of the input featuremap, used to de-normlize the coordinates.
|`int`     |`input_width`                 |The width of the input featuremap, used to de-normalize the coordinates.
|`int`     |`rpn_stride`                  |The cummulative stride from model input to the region proposal network(RPN), usually is 16.
|`float`   |`roi_min_size`                |The minimum size of the ROIs(at model input scaling), this is used to filter out small ROIs.
|`float`   |`nms_iou_threshold`           |The IoU threshold for non-maximum-suppression(NMS).
|`int`     |`pre_nms_top_n`               |The number of ROIs to be retained before doing NMS.
|`int`     |`post_nms_top_n`              |The number of ROIs to be retained after doing NMS.
|`list`    |`anchor_sizes`                |The list of anchor sizes, used to construct the anchor boxes.
|`list`    |`anchor_ratios`               |The list of anchor aspect ratios, used to construct the anchor boxes.



## Additional resources

The following resources provide a deeper understanding of the `proposalPlugin` plugin:

**Networks:**
-   [Faster R-CNN](https://arxiv.org/abs/1506.01497)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
