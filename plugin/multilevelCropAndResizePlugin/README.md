
# MultilevelCropAndResize

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `MultilevelCropAndResize` plugin performs the ROIAlign operation on the output feature maps from FPN (Feature Pyramid Network). It is used for MaskRCNN inference in Transfer Learning Toolkit. 


### Structure

This plugin supports the NCHW format. It takes 6 inputs in the following order: `roi`, and 5 `feature_maps` from FPN (Note: 5 `feature_maps` are required for this plugin and will not function properly with a lesser number of `feature_maps`).

`roi` is the ROI candidates from the `MultilevelProposeROI` plugin. Its shape is `[N, rois, 4]` where `N` is the batch_size, `rois` is the number of ROI candidates and `4` is the number of coordinates.

`feature_maps` are the output of FPN. In TLT MaskRCNN, the model we provide contains 5 feature maps from FPN's different stages.

This plugin generates one output tensor of shape `[N, rois, C, pooled_size, pooled_size]` where `C` is the channel of mutiple feature maps from FPN and `pooled_size` is the height(and width) of the feature area after ROIAlign.

## Parameters

This plugin has the plugin creator class `MultilevelCropAndResizePluginCreator` and the plugin class `MultilevelCropAndResize`.
  
The following parameters were used to create `MultilevelCropAndResize` instance:

| Type             | Parameter                       | Description
|------------------|---------------------------------|--------------------------------------------------------
|`int`             |`pooled_size`                    | The spatial size of a feature area after ROIAlgin will be `[pooled_size, pooled_size]`  
|`int[3]`          |`image_size`                     | The size of the input image in CHW. Defaults to [3, 832, 1344]

## Additional resources


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

March 2022: This is the second release of this `README.md` file.

June 2020: First release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
