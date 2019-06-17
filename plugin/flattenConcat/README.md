# flattenConcat

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `flattenConcat` plugin performs input tensor flattening followed by concatenation in a single step. The SSD object detection model has bounding box predictions, including bounding box location predictions and object classifications, from different feature maps in the neural network. The bounding box location prediction tensors from different feature maps are flattened and concatenated into a single tensor, as well as the object classification tensors. Merging tensor flattening and concatenation into a single step accelerates the inference speed in TensorRT.


### Structure

This plugin supports the NCHW format. It takes an arbitrary number of input tensors of shape `[N, C_1, H, W], [N, C_2, H, W], ..., [N, C_k, H, W]`, flattens and concatenates these input tensors, and generates an output tensor of shape `[N, C, 1, 1]` where `C = (C_1 + C_2 + ... + C_k) * H * W`.

For example, you have input tensor `A` of shape `[2, 2, 2, 2]`:
```
[[[[ 0 1]
   [ 2 3]]

  [[ 4 5]
   [ 6 7]]]
  

 [[[ 8 9]
   [10 11]]

  [[12 13]
   [14 15]]]]
```

and input tensor `B` of shape `[2, 3, 2, 2]`:
  
```
[[[[16 17]
   [18 19]]

  [[20 21]
   [22 23]]
 
  [[24 25]
   [26 27]]]

 
 [[[28 29]
   [30 31]]

  [[32 33]
   [34 35]]

  [[36 37]
   [38 39]]]]
```
 
After `flattenConcat` for the two inputs, the output tensor of shape `[2, 20, 1, 1]` is:
```
[[[[ 0]]

  [[ 1]]

  [[ 2]]

  [[ 3]]

  [[ 4]]

  [[ 5]]

  [[ 6]]

  [[ 7]]

  [[16]]

  [[17]]

  [[18]]

  [[19]]

  [[20]]

  [[21]]

  [[22]]

  [[23]]

  [[24]]

  [[25]]

  [[26]]

  [[27]]]


 [[[ 8]]

  [[ 9]]

  [[10]]

  [[11]]

  [[12]]

  [[13]]

  [[14]]

  [[15]]

  [[28]]

  [[29]]

  [[30]]

  [[31]]

  [[32]]

  [[33]]

  [[34]]

  [[35]]

  [[36]]

  [[37]]

  [[38]]

  [[39]]]]
```

## Parameters

This plugin has the plugin creator class `FlattenConcatPluginCreator` and the plugin class `FlattenConcat`.
  
The following parameters were used to create `FlattenConcat` instance:

| Type             | Parameter                      | Description
|------------------|--------------------------------|--------------------------------------------------------
|`int`             |`concatAxis`                    |The dimension along which to concatenate. Currently only `concatAxis = 1` is supported.
|`bool`            |`ignoreBatch`                   |Whether to ignore batch or not. Currently only `ignoreBatch = false` is supported.


## Additional resources

The following resources provide a deeper understanding of the `flattenConcat` plugin:

- [SSD Caffe Implementation](https://github.com/weiliu89/caffe/tree/ssd)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.