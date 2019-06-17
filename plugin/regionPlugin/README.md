# regionPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `regionPlugin`  is specifically used to generate encoded bounding box predictions, encoded bounding box objectness, and probabilities of bounding box being candidate objects for the YOLOv2 object detection model in TensorRT.


### Structure

The `regionPlugin` takes one input and generates one output.

The input has a shape of `[N, C, H, W]`, where:
-   `N` is the batch size
-   `C` is the number of channels in the input tensor. For example, `C = num * (coords + 1 + classes)`.
-   `H` is the height of feature map
-   `W` is the width of feature map

The information order of the channels are:
-   `[t_x, t_y, t_w, t_h, t_o, d_1, d_2, ..., d_classes] for bbox_1`
-   `[t_x, t_y, t_w, t_h, t_o, d_1, d_2, ..., d_classes] for bbox_2`, and so on
-   `[t_x, t_y, t_w, t_h, t_o, d_1, d_2, ..., d_classes]` for `bbox_num`, totalling `num * (coords + 1 + classes)` channels.
   
Specifically:
-  ` t_x, t_y, t_w, t_h` are the predicted offsets of bounding boxes before sigmoid activation
-   `t_o` is the predicted objectness of the bounding box before sigmoid activation (see [YOLOv2 paper](https://arxiv.org/abs/1612.08242))
-   `d_1, d_2, ..., d_classes` are the digits for each candidate class before the Softmax activation.

The output has the same shape as the input, in other words, `[N, C, H, W]`. The information order of the channels are:
-  `[sigmoid(t_x), sigmoid(t_y), t_w, t_h, sigmoid(t_o), p_1, p_2, ..., p_classes]` for `bbox_1`
-  `[sigmoid(t_x), sigmoid(t_y), t_w, t_h, sigmoid(t_o), p_1, p_2, ..., p_classes]` for `bbox_2`, and so on
-  `[sigmoid(t_x), sigmoid(t_y), t_w, t_h, sigmoid(t_o), p_1, p_2, ..., p_classes]` for `bbox_num`, totalling `num * (coords + 1 + classes)` channels

Specifically:
-   `sigmoid(t_x), sigmoid(t_y)`, and `sigmoid(t_o)` are sigmoid activated `t_x, t_y`, and `t_o` from the input
-   `p_1, p_2, ..., p_classes` are the probability for each candidate class after the Softmax activation.
 
**Note:** `t_w` and `t_h` from the input remain unchanged.


## Parameters

The `regionPlugin` has a plugin creator class `RegionPluginCreator` and plugin class `Region`.

The following parameters were used to create the `Region` instance.

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int`     |`num`                     |The number of predicted bounding box for each grid cell.
|`int`     |`coords`                  |The number of coordinates for the bounding box. This value has to be `4`. Other values for `coords` are not supported currently.
|`int`     |`classes`                 |The number of candidate classes to be predicted.
|`smTree`  |`softmaxTree`             |When performing yolo9000, `softmaxTree` is helping to perform Softmax on confidence scores, for example, to get the precise candidate classes through the word-tree structured candidate classes definition. `softmaxTree` is not required for non-hierarchical classification. The definition of `softmaxTree` can be found in [NvInferPlugin.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_plugin_8h_source.html) and [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/structnvinfer1_1_1plugin_1_1softmax_tree.html).
|`bool`    |`hasSoftmaxTree`          |If `softmaxTree` is not `nullptr`, it is `true`; else it is `false`.
|`int`     |`C`                       |The number of channels in the input tensor. `C = num * (coords + 1 + classes)` has to be satisfied.
|`int`     |`H`                       |The height of the input tensor (feature map).
|`int`     |`W`                       |The width of the input tensor (feature map).



## Additional resources

The following resources provide a deeper understanding of the `regionPlugin` plugin:

**Networks**
- [YOLOv2 paper](https://arxiv.org/abs/1612.08242)   


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.