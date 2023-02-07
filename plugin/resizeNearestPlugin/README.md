# ResizeNearest

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `ResizeNearest` plugin performs nearest neighbor interpolation among feature map. It is used in sample MaskRCNN. 


### Structure

This plugin supports the NCHW format. It takes one input tensor `feature_map`

`feature_map` can be arbitrary feature map from convolution layer of shape `[N, C, H, W]` 

`Resizenearest` generates the resized feature map according to scale factor. For example, if input feature is of `[N, C, H, W]`and`scale=2.0`, then the output feature will be of `[N, C, 2.0 * H, 2.0 * W]`

## Parameters

This plugin has the plugin creator class `ResizeNearestPluginCreator` and the plugin class `ResizeNearest`.
  
The following parameters were used to create `ResizeNearest` instance:

| Type               | Parameter                      | Description
|--------------------|--------------------------------|--------------------------------------------------------
|`float`             |`scale`                         | Scale factor for resize


## Additional resources

The following resources provide a deeper understanding of the `ResizeNearest` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

June 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
