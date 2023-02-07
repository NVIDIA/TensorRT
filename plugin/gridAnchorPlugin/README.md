# gridAnchorPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
	* [Example: Creating `GridAnchorGenerator` For An SSD Network](#example-creating-gridanchorgenerator-for-an-ssd-network)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

Some object detection neural networks such as Faster R-CNN and SSD use region proposal networks that require anchor boxes to generate predicted bounding boxes. This plugin is included in TensorRT and used in [sampleUffSSD](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#uffssd_sample) to run SSD.
  
The `gridAnchorPlugin` generates anchor boxes (prior boxes) from the feature map in object detection models such as SSD. It generates anchor box coordinates `[x_min, y_min, x_max, y_max]` with variances (scaling factors) `[var_0, var_1, var_2, var_3]` for the downstream bounding box decoding steps. It uses a series of CUDA kernels in the `gridAnchorLayer.cu` file to accelerate the process in TensorRT.

If the feature maps are square, then the `GridAnchor_TRT` plugin should be used. If the feature maps
are rectangular but non-square, then the `GridAnchorRect_TRT` plugin should be used.

### Structure

The `GridAnchorGenerator` plugin takes no inputs. However, it uses the attributes from `GridAnchorParameters` array typed `mParam`, the number of the feature maps we are generating anchor boxes for, and generates `mNumLayers` outputs (one per each feature map).

Each output has shape of `[2, H x W x mNumPriors x 4, 1]`. The first dimension has two channels.

-  The first channel is for the coordinates of the proposed anchor box. The position consists of four coordinates `[x_min, y_min, x_max, y_max]`.
-  The second channel is for the variance pre-calculated for bounding box decoding. The variance was copied from the `GridAnchorParameters.variance` that you provided to create the plugin.

## Parameters

The `GridAnchor_TRT` plugin consists of the plugin creator class `GridAnchorPluginCreator` and the plugin class `GridAnchorGenerator`.
The `GridAnchorRect_TRT` plugin consists of the plugin creator class `GridAnchorRectPluginCreator` and the plugin class `GridAnchorGenerator`.

`GridAnchorPluginCreator` and `GridAnchorPluginCreator` both take the following parameters as user input:
| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`float`   |`minSize`                 |Scale of anchors corresponding to finest resolution with respect to the height of input image. It corresponds to the `s_min` of the SSD paper. Default value is `0.2F`.
|`float`   |`maxSize`                 |Scale of anchors corresponding to coarsest resolution with respect to the height of input image. It corresponds to the `s_max` of the SSD paper. Default value is `0.95F`.
|`float*`  |`aspectRatios`            |List of aspect ratios to place on each grid point.
|`int*`    |`featureMapShapes`        |Shapes of the feature maps. If creating a `GridAnchorRect_TRT` plugin, this must be a list of size `numLayers * 2` where the height and width of each feature map is listed in order. If creating a `GridAnchor_TRT` plugin, this must be list of size `numLayers` where the height (or width) of each feature map is listed in order.
|`float*`  |`variance`                |Variance for adjusting the prior boxes.
|`int`     |`numLayers`               |Number of feature maps. Default value is `6`.
  
`GridAnchorGenerator`'s constructor takes `numLayers` and an array of `GridAnchorParameters` typed parameters. The corresponding `GridAnchorParameters` are created internally by `GridAnchorPluginCreator` (not the plugin user's responsibility). `GridAnchorParameters` consists of the following parameters:

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`float`   |`minSize`                 |Scale of anchors corresponding to finest resolution with respect to the height of input image. It corresponds to the `s_min` of the SSD paper.
|`float`   |`maxSize`                 |Scale of anchors corresponding to coarsest resolution with respect to the height of input image. It corresponds to the `s_max` of the SSD paper.
|`float*`  |`aspectRatios`            |List of aspect ratios to place on each grid point.
|`int`     |`numAspectRatios`         |Number of elements in aspectRatios.
|`int`     |`H`                       |Height of feature map to generate anchors for.
|`int`     |`W`                       |Width of feature map to generate anchors for.
|`float[4]`|`variance`                |Variance for adjusting the prior boxes.

### Example: Creating `GridAnchorGenerator` For An SSD Network

If we were to create a `GridAnchorGenerator` for a SSD network consisting of 6 layers, then the following parameters would be passed to the plugin creator class `GridAnchorPluginCreator`:
```
numLayers=6,
minSize=0.2,
maxSize=0.95,
aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
variance=[0.1, 0.1, 0.2, 0.2],
featureMapShapes=[19, 10, 5, 3, 2, 1]
```
 
The `GridAnchorGenerator` uses distinct `GridAnchorParameters` for each feature map to generate anchor boxes, therefore, it takes an array of `GridAnchorParameters` with a length of the number of feature maps (`mNumLayers`) to create the plugin. In the above example, we have 6 layers, the plugin needs an array of 6 `GridAnchorParameters` to create the plugin. In this particular example, all the `GridAnchorParameters` except for the first one in the array are the same according to the SSD model settings. After the plugin is created, each feature map, except for the first feature map, will have 5 + 1 anchor boxes, where 5 is the number of elements in `aspectRatios` and 1 is an additional default anchor box with an aspect ratio of 1.0. The first layer, as described in the [SSD: Single Shot MultiBox Detector paper](https://arxiv.org/pdf/1512.02325.pdf), has fewer number of anchor boxes. In our case, it was set to 3, in our code, `int numFirstLayerARs = 3`, and there will be no additional default anchor box of aspect ratio 1.0. The feature map shapes also supports rectangular inputs. The height and width of the feature maps are put in order in the list above.

**Note:** The above settings are slightly different to the original published SSD paper.

## Additional resources

The following resources provide a deeper understanding of the `gridAnchorPlugin` plugin:

**Networks:**
-   [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
-   [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

**Documentation:**
-   [GridAnchorParameters detailed descriptions](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/structnvinfer1_1_1plugin_1_1_grid_anchor_parameters.html)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.