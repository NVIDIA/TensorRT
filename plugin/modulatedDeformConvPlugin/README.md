# modulatedDeformConv

**Table Of Contents**
- [Description](#description)
- [Structure](#structure)
- [Parameters](#parameters)
- [Additional Resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `modulatedDeformConvPlugin` performs the modulated deformable convolution operations. The modulated Deformable Convolution is similar to regular Convolution but its receptive field is deformed because of additional spatial offsets and modulating scalars. This plugin is used in OCDNet in TAO Toolkit.

## Structure

#### Inputs

This plugin works in NCHW format. It takes five input tensors:

`x` is the input data tensor. Its shape is `[batch_size, input_channels, in_height, in_width]`.

`offset` is the offset tensor which denotes the offset for the sampling location in a convolutional kernel.
Its shape is `[batch_size, 2 * deformable_group * kernel_height * kernel_width , out_height, out_width]`.

`mask` is the modulated scalar tensor. Its shape is `[batch_size, deformable_group * kernel_height * kernel_width, out_height, out_width]`.

`weight` is the kernel tensor. Its shape is `[output_channels, input_channels // group, kernel_height, kernel_width]`.

`bias` is an optional tensor. Its shape is `[output_channels]`.


#### Outputs

This plugin generates one output tensor of shape `[batch_size, output_channels, out_height, out_width]`.


## Parameters

This plugin has the plugin creator class `ModulatedDeformableConvPluginDynamicCreator` and the plugin class `ModulatedDeformableConvPluginDynamic`.

The following parameters are used to create a `ModulatedDeformableConvPluginDynamic` instance:

| Type    | Parameter         | Description
|---------|-------------------|--------------------------------------------------------
| `int[2]`| `stride`          | It is a distance (in pixels) to slide the filter on the feature map. Defaults to [1,1].
| `int[2]`| `padding`         | It is a number of pixels to add to each axis. Defaults to [0,0].
| `int[2]`| `dilation`        | Denotes the distance in width and height between elements (weights) in the filter. Defaults to [1,1].
| `int`   | `group`           | It is the number of groups which input_channels and output_channels should be divided into. For example, `group` equal to 1 means that all filters are applied to the whole input (usual convolution), `group` equal to 2 means that both input and output channels are separated into two groups and the i-th output group is connected to the i-th input group channel. `group` equal to a number of output feature maps implies depth-wise separable convolution. Defaults to 1.
| `int`   | `deformable_group`| It is the number of groups in which offset input and output should be split into along the channel axis. Defaults to 1.


## Additional Resources

The following resources provide a deeper understanding of the `modulatedDeformConvPlugin` plugin:

- [Deformable ConvNets v2](https://arxiv.org/pdf/1811.11168.pdf)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

## Changelog
- April 2025: Added version 2 of the plugin that uses the IPluginV3 interface. The version 1 (using IPluginV2DynamicExt interface) is now deprecated. The version 2 mirrors version 1 in IO and attributes.
- Jan 2023: Initial release of IPluginV2DynamicExt implementation.


## Known issues

There are no known issues in this plugin.
