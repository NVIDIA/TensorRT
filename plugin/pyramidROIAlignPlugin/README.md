# PyramidROIAlign

**Table Of Contents**
- [Changelog](#changelog)
- [Description](#description)
- [Structure](#structure)
- [Parameters](#parameters)
- [Compatibility Modes](#compatibility-modes)
- [Additional Resources](#additional-resources)
- [License](#license)
- [Known issues](#known-issues)

## Changelog

February 2022
Major refactoring of the plugin to add new features and compatibility modes.

June 2019
This is the first release of this `README.md` file.

## Description

The `PyramidROIAlign` plugin performs the ROIAlign operations on the output feature maps of an FPN (Feature Pyramid Network). This is used in many implementations of FasterRCNN and MaskRCNN. This operation is also known as ROIPooling.

## Structure

#### Inputs

This plugin works in NCHW format. It takes five input tensors:

`rois` is the proposal ROI coordinates, these usually come from a Proposals layer or an NMS operation, such as the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin. Its shape is `[N, R, 4]` where `N` is the batch_size, `R` is the number of ROI candidates and `4` is the number of coordinates.

`feature_map_[0-3]` are the four feature map outputs of an FPN. It is expected these to form a multi-scale pyramid, with the 0th feature map being the highest resolution map. For example:
- `feature_map_0` with shape `[N, C, 256, 256]`, usually corresponds to `P2`.
- `feature_map_1` with shape `[N, C, 128, 128]`, usually corresponds to `P3`.
- `feature_map_2` with shape `[N, C, 64, 64]`, usually corresponds to `P4`.
- `feature_map_3` with shape `[N, C, 32, 32]`, usually corresponds to `P5`.

#### Outputs

This plugin generates one output tensor of shape `[N, R, C, pooled_size, pooled_size]` where `C` is the same number of channels as the feature maps, and `pooled_size` is the configured height (and width) of the feature area after ROIAlign.

## Parameters

This plugin has the plugin creator class `PyramidROIAlignPluginCreator` and the plugin class `PyramidROIAlign`.
  
The following parameters are used to create a `PyramidROIAlign` instance:

| Type    | Parameter              | Default    | Description
|---------|------------------------|------------|--------------------------------------------------------
| `int`   | `pooled_size`          | 7          | The spatial size of a feature area after ROIAlgin will be `[pooled_size, pooled_size]`
| `int[]` | `image_size`           | 1024,1024  | An 2-element array with the input image size of the entire network, in `[image_height, image_width]` layout
| `int`   | `fpn_scale`            | 224        | The canonical ImageNet size with which the FPN scale map selection is calculated.
| `int`   | `sampling_ratio`       | 0          | If set to 1 or larger, the number of samples to take for each output element. If set to 0, this will be calculated adaptively by the size of the ROI.
| `int`   | `roi_coords_absolute`  | 1          | If set to 0, the ROIs are normalized in [0-1] range. If set to 1, the ROIs are in full image space.
| `int`   | `roi_coords_swap`      | 0          | If set to 0, the ROIs are in `[x1,y1,x2,y2]` format (PyTorch standard). If set to 1, they are in `[y1,x1,y2,x2]` format (TensorFlow standard).
| `int`   | `roi_coords_plusone`   | 0          | If set to 1, the ROI area will be calculated with a +1 offset on each dimension to enforce a minimum size for malformed ROIs.
| `int`   | `roi_coords_transform` | 2          | The coordinate transformation method to use for the ROI Align operation. If set to 2, `half_pixel` sampling will be performed. If set to 1, `output_half_pixel` will be performed. If set to 0, no pixel offset will be applied. More details on compatibility modes below.

## Compatibility Modes

There exist many implementations of FasterRCNN and MaskRCNN, and unfortunately, there is no consensus on a canonical way to execute the ROI Pooling of an FPN. This plugin attempts to support multiple common implementations, configurable via the various parameters that have been exposed.

#### Detectron 2

To replicate the standard ROI Pooling behavior of [Detectron 2](https://github.com/facebookresearch/detectron2), set the parameters as follows:

- `roi_coords_transform`: 2. This implementation uses half_pixel coordinate offsets.
- `roi_coords_plusone`: 0. This implementation does not offset the ROI area calculation.
- `roi_coords_swap`: 0. This implementation follows the PyTorch standard for coordinate layout.
- `roi_coords_absolute`: 1. This implementation works will full-size ROI coordinates.
- `sampling_ratio`: 0. This implementation uses an adaptive sampling ratio determined from each ROI area.
- `fpn_scale`: 224. This implementation uses the standard 224 value to determine the scale selection thresholds.

#### MaskRCNN Benchmark

To replicate the standard ROI Pooling behavior of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), set the parameters as follows:

- `roi_coords_transform`: 1. This implementation uses output_half_pixel coordinate offsets.
- `roi_coords_plusone`: 1. This implementation offsets the ROI area calculation by adding 1 to each ROI dimension.
- `roi_coords_swap`: 0. This implementation follows the PyTorch standard for coordinate layout.
- `roi_coords_absolute`: 1. This implementation works will full-size ROI coordinates.
- `sampling_ratio`: 2. This implementation performs two samples per output element.
- `fpn_scale`: 224. This implementation uses the standard 224 value to determine the scale selection thresholds.

#### Backward-Compatibility

This plugin underwent a major refactoring since its initial version. In order to replicate the original plugin behavior, use the following parameters or set the `legacy` parameter to 1.

- `roi_coords_transform`: -1. This is a special mode exclusively to allow back-compatibility with the original PyramidROIAlign plugin.
- `roi_coords_plusone`: 0. The original implementation had no such logic.
- `roi_coords_swap`: 1. The original implementation expected TensorFlow-like coordinate layout.
- `roi_coords_absolute`: 0. The original implementation expected normalized ROI coordinates.
- `sampling_ratio`: 1. The original implementation performed a single sample per output element.
- `fpn_scale`: 158. The original implementation used a non-standard FPN scale threshold which works out to canonical value of 158, use this value to replicate the old behavior.

#### Other Implementations

Other FPN ROI Pooling implementations may be adapted by having a better understanding of how the various parameters work internally.

**FPN Scale**: This values helps to determine the threshold used to select one feature map or another, according to the size of each ROI. The value defined here comes from the canonical ImageNet size as defined in Equation 1 of the original [FPN paper](https://arxiv.org/pdf/1612.03144.pdf). This size defines the minimum `sqrt(height * width)` that will be sampled from the `P4` feature map. Therefore, by setting the default value of 224, the following feature map selection schedule is used:

- `sqrt(height * width)` between [0 - 111]: Sample from `P2`
- `sqrt(height * width)` between [112 - 223]: Sample from `P3`
- `sqrt(height * width)` between [224 - 447]: Sample from `P4`
- `sqrt(height * width)` between [448 - max]: Sample from `P5`

If your FPN implementation uses a different mapping, set the `fpn_scale` parameter such that its value thresholds the `P4` feature map, `fpn_scale/2` thresholds `P3`, and `fpn_scale*2` thresholds `P5`.

**Coordinate Transformation**: This flag primarily defines various offsets applied to coordinates when performing the bilinear interpolation sampling for ROI Align. The three supported values work as follows:
- `roi_coords_transform` = -1: This is a back-compatibility that calculates the scale by subtracting one to both the input and output dimensions. This is similar to the `align_corners` resize method.
- `roi_coords_transform` = 0: This is a naive implementation where no pixel offset is applied anywhere. It is similar to the `asymmetric` resize method.
- `roi_coords_transform` = 1: This performs half pixel offset by applying a 0.5 offset only in the output element sampling. This is similar to the `output_half_pixel` ROI Align method.
- `roi_coords_transform` = 2: This performs half pixel offset by applying a 0.5 offset in the output element sampling, but also to the input map coordinate. This is similar to the `half_pixel` ROI Align method, and is the favored method of performing ROI Align.

**Force Malformed Boxes by +1**: Some legacy implementations of ROI Pooling, especially those that do not deal properly with pixel sampling, enforce a minimum ROI area size. This is performed by adding a value of 1, in absolute image space, to each dimension of the ROI box. This method is not very common, but some [older implementations](https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/csrc/cuda/ROIPool_cuda.cu#L35-L37) use it, so this attribute has been exposed to replicate this behavior.

## Additional Resources

The following resources provide a deeper understanding of the `PyramidROIAlign` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)
- [FPN](https://arxiv.org/abs/1612.03144)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Known issues

There are no known issues in this plugin.
