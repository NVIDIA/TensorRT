# roiAlign Plugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `RoiAlign` plugin extract a small feature map from each RoI in detection and segmentation based tasks. It removes the harsh quantization of RoI Pool, properly aligning the extracted features with the input.


### Structure

This plugin supports the NCHW format. It takes three inputs: `X`, `rois` and `batch_indices`.

`X`
Arbitrary feature map from convolution layer of shape `[N, C, H, W]` 

`rois`
Regions of Interest used to extract the small feature map of shape `[num_rois, 4]`, given as `[[x1, y1, x2, y2], ...]`. `num_rois` is the total number of rois of all
`N` batch.

`batch_indices`
1-D tensor of shape `[num_rois]`. Each roi in `rois` has a batch index denoting the corresponding image in `X`, `batch_indices` contains all batch index of `rois` with the same row order: `rois[batch_indices[i]]` is the roi of `X[batch_indices[i]]`

`RoiAlign` plugin generate one output:

`Y`
The extracted small feature map of shape `[num_rois, C, pooled_height, pooled_width]`. The `i-th` batch element `Y[i-1]` is a pooled feature map corresponding to the `i-th` ROI `rois[i-1]`.

## Parameters

This plugin has the plugin creator class `RoiAlignPluginDynamicCreator` and the plugin class `RoiAlignPluginDynamic`.
  
The following parameters were used to create `RoiAlignPluginDynamic` instance:

| Type               | Parameter                      | Description
|--------------------|--------------------------------|--------------------------------------------------------
|`string`            |`coordinate_transformation_mode`| Allowed values are 'half_pixel' and 'output_half_pixel'. Use the value 'half_pixel' to pixel shift the input coordinates by -0.5 (the recommended behavior). Use the value 'output_half_pixel' to omit the pixel shift for the input (use this for a backward-compatible behavior).
|`int`             |`mode`                         | The pooling method. Two modes are supported: 'avg' and 'max'. Default is 'avg'.
|`int`             |`output_height`                | The output height of `Y`. Default 1.
|`int`             |`output_width`                 | The output width of `Y`. Default 1.
|`int`             |`sampling_ratio`               | Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If == 0, then an adaptive number of grid points are used (computed as ceil(roi_width / output_width), and likewise for height). Default is 0.
|`float`           |`spatial_scale`                | Multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling, i.e., spatial scale of the input feature map `X` relative to the input image. E.g.; default is 1.0f.


## Additional resources

The following resources provide a deeper understanding of the `RoiAlignPluginDynamic` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

May 2022
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
