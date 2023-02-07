# SpecialSlice 

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `SpecialSlice` plugin slice the detections of MaskRCNN from `[y1, x1, y2, x2, class_label, score]` to `[y1, x1, y2, x2]`. It is used in sampleMaskRCNN.


### Structure

This plugin supports the NCHW format. It takes one input tensor: `detections` 

`detections` is the output of `DetectionLayer` in MaskRCNN model. Its shape is `[N, num_det, 6]` where `N` is the batch size, `num_det` is the number of detections generated from `DetectionLayer` and `6` means 6 elements of a detection `[y1, x1, y2, x2, class_label, score]`.

This plugin generates one output tensor of shape `[N, num_det, 4]`. 

## Parameters

This plugin has the plugin creator class `FlattenConcatPluginCreator` and the plugin class `FlattenConcat`.

This plugin has no parameter.
  

## Additional resources

The following resources provide a deeper understanding of the `flattenConcat` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

June 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
