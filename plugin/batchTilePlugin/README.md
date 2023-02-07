# batchTilePlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `batchTilePlugin` tiles a tensor `N` times along its first dimension (batch dimension) where `N` is the batch size. The result tensor will have shape `N` on its first dimension and each `tensor[i: i+1,...]` is a copy of input tensor (for integer `i < N`).

This plugin takes 2 input tensors, the first input tensor should have first dim `N` and the second tensor should have first dim 1. The output tensor will be the tile results of the second tensor.

### Structure

Both input tensors to this plugin must have 4 dimensions (NCHW)

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

and input tensor `B` of shape `[1, 3, 2, 2]`:
  
```
[[[[16 17]
   [18 19]]

  [[20 21]
   [22 23]]
 
  [[24 25]
   [26 27]]]]
```
 
After `batchTilePlugin` for the two inputs, the output tensor of shape `[2, 3, 2, 2]` is:

```
[[[[16 17]
   [18 19]]

  [[20 21]
   [22 23]]
 
  [[24 25]
   [26 27]]]
   
   
 [[[16 17]
   [18 19]]

  [[20 21]
   [22 23]]
 
  [[24 25]
   [26 27]]]]
```

## Parameters

This plugin has the plugin creator class `BatchTilePluginCreator` and the plugin class `BatchTilePlugin`.
  
No parameter is required for this plugin.


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

Jul. 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.