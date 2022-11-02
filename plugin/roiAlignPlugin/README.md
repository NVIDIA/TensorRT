# ROIAlign

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The ROIAlign plugin implements the Region of Interest (RoI) align operation described in the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870), in compliance with the [ONNX specification for the same](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign) (insofar as TRT limitations allow).  

### Structure

This plugin has the plugin creator class `ROIAlignPluginCreator` and the plugin class `ROIAlign` which extends `IPluginV2DynamicExt`.

The RoiAlign plugin consumes the following inputs:

1. `X` with shape (`batch_size`, `C`, `height`, `width`): An input tensor.
2. `rois` with shape (`num_rois`, `4`): Regions of interest (ROIs).
3. `batch_indices` with shape (`num_rois`,): The set of indices in `X` to apply pooling. i.e. The ROI defined by `rois[i]`, should be used to pool `X[batch_indices[i]]`.

The RoiAlign plugin produces the following output:

1. `Y` with shape (`num_rois`, `C`, `output_height`, `output_width`): `Y[i]` is the output produced when pooling is applied over the ROI defined by `rois[i]` on `X[batch_indices[i]]`.

## Parameters
  
The `ROIAlign` plugin has the following parameters:

| Type             | Parameter                       | Description
|------------------|---------------------------------|--------------------------------------------------------
|`int`             |`coordinate_transformation_mode`                    | Whether or not the input coordinates should be shifted by 0.5 pixels. Allowed values are 0 and 1. Default is 1 and indicates a shift.
|`int`             |`mode`                    | The pooling method. Average pooling (indicated by 1) and max pooling (indicated by 0) are supported. Allowed values are 0 and 1. Default is 1.
|`int`             |`output_height`                    | Height of the pooled output (`Y`). Allowed values are positive integers. Default is 1.
|`int`             |`output_width`                    | Width of the pooled output (`Y`). Allowed values are positive integers. Default is 1.
|`int`             |`sampling_ratio`                    | Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If positive, then exactly `sampling_ratio` x `sampling_ratio` grid points are used. If zero, then an adaptive number of grid points are used (computed as ceil(roi_width / output_width), and likewise for height). Allowed values are non-negative integers. Default is 0.
|`float`           |`spatial_scale`                    | Multiplicative factor used to translate the ROI coordinates from their input spatial scale to the scale used when pooling. Allowed values are positive. Default is 1.0.

## Additional resources

The following resources provide a deeper understanding of the `ROIAlign` plugin:

- [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
- [ONNX specification for ROIAlign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.

## Changelog

June 2022: This is the first release of this `README.md` file.

## Known issues

There are no known issues in this plugin.
