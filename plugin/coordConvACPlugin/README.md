# coordConvACPlugin

**Table Of Contents**
- [coordConvACPlugin](#coordconvacplugin)
  - [Description](#description)
    - [Structure](#structure)
  - [Additional resources](#additional-resources)
  - [License](#license)
  - [Changelog](#changelog)
  - [Known issues](#known-issues)

## Description

The `coordConvACPlugin` implements the `CoordConv` layer. This layer was first introduced by Uber AI Labs in 2018, and improves on regular convolution by adding additional channels containing relative coordinates to the input tensor. These additional channels allows the subsequent convolution to retain information about where it was applied.

Each node with the op name `CoordConvAC` in `ONNX` graph will be mapped to that plugin. `Conv` node should follow after each `CoordConvAC` node into `ONNX` graph. 

For example, say we have an input tensor for a Conv layer named `X` with shape `[N, C, H, W]`, where `N` is the batch size, `C` is the number of channels, `H` is the height, and `W` is the width. In the CoordConv layer for each input the plugin will concatenate two additional channels with shape `[1, H, W]` at the end. The first extra channel contains relative coordinates along the Y axis and the second extra channel contains coordinates along the X axis. As a result we will get a new input tensor with shape `[N, C+2, H, W]` before applying regular convolution.

Relative coordinates have values in the range of `[-1,1]` where -1 is the value of the top row of the first channel (Y axis) and the left column of second channel (X axis), and 1 is the value for the bottom row of the first channel (Y axis) and the right column of second channel (X axis). All other values of the matrix will be filled in with a constant step value. between -1 and 1.

Formula for calculating constant step value is:

`STEP_VALUE_H = 2 / (H - 1)` - step value for the 1st channel

`STEP_VALUE_W = 2 / (W - 1)` - step value for the 2nd channel

Below are examples of 1st and 2nd channels for input data with `H=5, W=5, STEP_VALUE_H=0.5, and STEP_VALUE_W=0.5`

First channel with Y relative coordinates

| | | | | | 
| ------------- |:-------------:| -----|-------------| -----:|
| -1	| -1	| -1	| -1	| -1 |
| -0.5 | 	-0.5 | 	-0.5 | 	-0.5 | 	-0.5 | 
| 0 | 	0 | 	0 | 	0 | 	0 | 
| 0.5 | 	0.5 | 	0.5 | 	0.5 | 	0.5 | 
| 1 | 	1 | 	1 | 	1 | 	1 | 

Second channel with X relative coordinates

|     |     |     |     |     | 
| ------------- |:-------------:| -----|-------------| -----:|
| -1	| -0.5	| 0	| 0.5	| 1 |
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 
|  -1 | 	-0.5 | 	0 | 	0.5 | 	1 | 

These two matrices will be concatenated with the input data with the formula `CONCAT([INPUT_DATA, 1ST_CHANNEL, 2ND_CHANNEL])` on the channel dimension.

  
### Structure

This plugin takes one input and generates one output. Input shape is `[N, C, H, W]` and the output shape is `[N, C + 2, H, W]`.

## Additional resources

The following resources provide a deeper understanding of the `coordConvACPlugin` plugin:

**Networks**  
- Paper about Coord Conv layer ["An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"](https://arxiv.org/abs/1807.03247)    
- Blog post by Uber AI Labs about [CoordConv layer](https://eng.uber.com/coordconv/)
- Open-source implementations of the layer in Pytorch [source1](https://github.com/walsvid/CoordConv), [source2](https://github.com/mkocabas/CoordConv-pytorch)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

April 2020
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
