# decodeBbox3D Plugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `decodeBbox3DPlugin` performs 3D bounding boxes decoding for PointPillars model.

`decodeBbox3DPlugin` implements the 3D bounding boxes decoding. It applies deltas to anchor boxes and produces the 3D bounding boxes in the format: `(x, y, z, dx, dy, dz, rotation)`. The deltas are the input tensors of this plugin, and the anchors are computed on-the-fly based on the parameters passed in from the corresponding node in ONNX graph.

This plugin is optimized for the above steps and it allows you to do PointPillars inference in TensorRT.


### Structure

The `decodeBbox3DPlugin` takes 3 inputs; `cls_preds`, `box_preds`, and `dir_cls_preds`.

`cls_preds`
Predicted confidence(score) values. This tensor has a shape `[N, H, W, A*C]`, where `N` is the batch size, `H` is the height of the feature map, `W` is the width of the feature map, and `A` is the number of anchor boxes per grid point on the feature map, and `C` is the number of object classes.


`box_preds`
Predicted per-class bounding box delta values. This tensor has a shape `[N, H, W, A*7]`, where `N, H, W, A` are as above, and `7` is just the size of a bounding box delta. The decoded bounding box also have a length of 7 and are represented as `(x, y, z, dx, dy, dz, rotation)`.


`dir_cls_preds`
Predicted direction confidence(score) values. There are two directions for each bounding box(0 and 1). The one with the larger score will be the final detected direction and the direction value(0 or 1) will be used to compute the rotation values in 3D bounding boxes.


The `decodeBbox3DPlugin` generates the following 2 outputs:

`output_boxes`
The decoded 3D bounding boxes. This tensor has a shape of `[N, H*W*A, 9]`, where `N, H, W, A` are as above, and the number 9 represents the bounding boxes(7) plus the score(1) and class ID(1). The last dimension are encoded as `(x, y, z, dx, dy, dz, rotation, class_id, score)`.


`num_boxes`
The number of valid bounding boxes in `output_boxes`. Bounding boxes whose score are higher than a certain threshold is regarded as a valid bounding box. The score threshold is a parameter of this plugin(see below). With the number of valid boxes, we can easily retrieve the useful bounding boxes from thousands of `output_boxes`.

The decoding will produce `num_boxes` number of boxes and those boxes can be applied to a 3D Non-Maximum-Suppression(NMS) operation(not covered in this plugin) to eliminate highly overlapped boxes and produce the final detection result.

## Parameters

`decodeBbox3DPlugin` has plugin creator class `decodeBbox3DPluginCreator` and plugin class `decodeBbox3DPlugin`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`list of floats`     |`anchors`                |The anchor sizes and aspect ratios.
|`list of floats`     |`anchor_bottom_height`               |The heights of the anchor bottom, per class.
|`float`   | `dir_limit_offset`  | Direction limit offset.
|`float`   | `dir_offset`        | Direction offset.
|`int`     | `num_dir_bins`      | Number of direction bins.
|`list of floats` |`point_cloud_range` | The range of point cloud coordinates.
|`float`       | `score_thresh`   | The score threshold for bounding boxes.
    
## Additional resources

The following resources provide a deeper understanding of the `decodeBbox3DPlugin` plugin:

**Networks:**
-   [PointPillars](https://arxiv.org/pdf/1812.05784)

**Documentation:**
-   [PointPillars](https://arxiv.org/pdf/1812.05784)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

Dec 2021
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
