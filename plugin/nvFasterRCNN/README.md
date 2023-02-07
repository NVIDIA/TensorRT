# NvPluginFasterRCNN Plugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `NvPluginFasterRCNN` performs object detection for the Faster R-CNN model. This plugin is included in TensorRT and used in [sampleFasterRCNN](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#fasterrcnn_sample) to perform inference.

`NvPluginFasterRCNN` decodes predicted bounding boxes, extracts their corresponding objectness score, extracts region of interest from predicted bounding boxes using non maximum suppression, and extracts the feature map of region of interest (ROI) using ROI pooling for downstreaming object classification tasks.

This plugin is optimized for the above steps and it allows you to do Faster R-CNN inference in TensorRT.


### Structure

The `NvPluginFasterRCNN` takes four inputs; `scores`, `deltas`, `fmap` and `iinfo`.

`scores`
Bounding box (region proposal) objectness scores. scores has shape `[N, A x 2, H, W]` where `N` is the batch size, `A` is the number anchor boxes per pixel on the feature map, `H` is the height of feature map, and `W` is the width of feature map. The second dimension is `A x 2` because Faster R-CNN uses binary Softmax (probability of having object and probability of not having object) to classify the objectness for each bounding box.

`deltas`
Predicted bounding box offsets. `deltas` has shape `[N, A x 4, H, W]` where `N` is the batch size, `A` is the number anchor boxes per pixel on the feature map, `H` is the height of feature map, and `W` is the width of feature map. The second dimension is `A x 4` because each anchor box or bounding box consists of four parameters.

`fmap`
Feature map using for bounding box regression and classification. `fmap` has shape `[N, C, H, W]` where `N` is the batch size, `C` is the number of channels in feature map, `H` is the height of feature map, and `W` is the width of feature map.

`iinfo`
Original image input information. `iinfo` has shape `[N, 3]` where `N` is the batch size, `3` represents the height, width, and resize scale (the same as `featureStride`) of original input image.

The `NvPluginFasterRCNN` generates the following two outputs:

`rois`
Coordinates of region of interest bounding boxes on the original input image. `rois` has shape `[N, 1, nmsMaxOut, 4]`, where `N` is the batch size, `nmsMaxOut` is the maximum number of region of interest bounding boxes, and `4` represents and region of interest bounding box coordinates `[x_1, y_1, x_2, y_2]`. Here, `x_1` and `y_1` are the coordinates of bounding box at the top-left corner, and `x_2` and `y_2` are the coordinates of bounding box at the bottom-right corner.

`pfmap`
ROI pooled feature map corresponding to the region of interest. `pfmap` has shape `[N, nmsMaxOut, C, poolingH, poolingW]` where `N` is the batch size, `nmsMaxOut` is the maximum number of region of interest bounding boxes, `C` is the number of channels in the feature map, `poolingH` is the height of ROI pooled feature map, and `poolingW` is the width of ROI pooled feature map.

`NvPluginFasterRCNN` essentially does region proposal inference followed by region of interest (ROI) pooling.

The proposal inference step includes three steps: extract objectness scores from `scores` input, decode predicted bounding box from `deltas` input, non-maximum suppression and get the region of interest bounding boxes using the extracted objectness scores and the decoded bounding boxes.

The ROI pooling step uses the inferred region of interest bounding boxes information to extract its corresponding regions on feature map, and does POI pooling to get uniformly shaped features from different shaped region of interest bounding boxes.

## Parameters

`NvPluginFasterRCNN` has plugin creator class `RPROIPluginCreator` and plugin class `RPROIPlugin`.

The `RPROIParams` data structure was used to create `RPROIPlugin` instance. The data structure is defined below and consists of the following attributes:
```
struct RPROIParams
{
	int poolingH, poolingW, featureStride, preNmsTop,
		nmsMaxOut, anchorsRatioCount, anchorsScaleCount;
	float iouThreshold, minBoxSize, spatialScale;
};
```

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int`     |`poolingH`                |The height of the output in pixels after ROI pooling on the feature map.
|`int`     |`poolingW`                |The width of the output in pixels after ROI pooling on the feature map.
|`int`     |`featureStride`           |The ratio of the input image size to the feature map size; assuming the max pooling layers in the neural network uses square filters. For example, the input image size is `[1600, 800]`, after max pooling of size `[4, 4]` twice, the feature map now becomes `[100, 50]`, and `featureStride = 4^2 = 16`. In the Faster R-CNN settings from the paper, the value is `16`.
|`int`     |`preNmsTop`               |The number of region proposals before applying NMS using objectness which is the probability of containing an object in the region proposal. The region proposals will be sorted using its objectness. If the number of regions you proposed from the previous region proposal network (RPN) is greater than `preNmsTop`, the exceeded region proposals with low objectness will be ignored. This value is particularly useful during training to control the number of bounding boxes for regression, but is theoretically useless during inference. In the Faster R-CNN settings from the paper, the value is `6000`.
|`int`     |`nmsMaxOut`               |The number of region proposals after applying NMS. The region proposals will be sorted using its objectness and then applied NMS. At most the `nmsMaxOut` region proposals exist after NMS is considered as regions of interest.
|`int`     |`anchorsRatioCount`       |The number of anchor box ratios. For example, if the anchor box ratios (aspect ratios) are 1:1, 1:2, and 2:1, then `anchorsRatioCount = 3`.
|`int`     |`anchorsScaleCount`       |The number of anchor box scales. If the anchor box scales (scale factors) are 8, 16, and 32, then `anchorsScaleCount = 3`.
|`float`   |`iouThreshold`            |The IOU threshold used for the NMS step.
|`float`   |`minBoxSize`              |The minimum box size used for the anchor box calculation.
|`float`   |`spatialScale`            |The inverse of `featureStride`, in other words, `spatialScale = 1.0 / featureStride`.


## Additional resources

The following resources provide a deeper understanding of the `NvPluginFasterRCNN` plugin:

**Networks:**
-   [Faster R-CNN](https://arxiv.org/abs/1506.01497)

**Documentation:**
-   [ROI Pooling Definition from Fast R-CNN](https://arxiv.org/abs/1504.08083)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.