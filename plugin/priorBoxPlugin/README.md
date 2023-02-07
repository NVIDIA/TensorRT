# priorBoxPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `priorBoxPlugin` generates prior boxes (anchor boxes) from a feature map in object detection models such as SSD. This plugin is included in TensorRT and used in [sampleSSD](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#sample_ssd) to run SSD.

This sample generates anchor box coordinates `[x_min, y_min, x_max, y_max]` with variances (scaling factors) `[var_0, var_1, var_2, var_3]` for the downstream bounding box decoding steps. The `priorBoxPlugin` uses a series of CUDA kernels in the `priorBoxLayer.cu` file to accelerate the process. The differences between `priorBoxPlugin` and `gridAnchorPlugin` is that `priorBoxPlugin` generates prior boxes for one feature map in the model at one time, while `gridAnchorPlugin` generates all prior boxes for all feature maps in the model at one time.
  
### Structure

Plugin `PriorBox` is created for each feature map. Plugin `PriorBox` takes no input (or one input to infer its shape information), and uses `PriorBoxParameters` to generate one output. The input is the feature map that needs to generate prior boxes and the output is the prior box data generated. The input has shape of `[N, C, H, W]` where `N` is the batch size, `C` is the number of channels, `H` is the height of the feature map input, and `W` is the width of the feature map input.

The output has shape `[2, H * W * numPriors * 4, 1]`. The first channel is for prior box coordinates. The second channel is for prior box scaling factors, which is simply a copy of the variance provided.

`H` and `W` are the height and width of the feature map the plugin is working on. `numPriors` is the number of prior boxes generated for one grid cell on the feature map. The value of `numPriors` is determined by the number of minimum sized box values, the number of maximum sized box values, the number of aspect ratios, and if we flip the aspect ratios or not. All the coordinates of prior boxes generated are in the format of `[x_min, y_min, x_max, y_max]`, and are scaled against image width and height in a range of `[0, 1]`.

A typical `PriorBox` layer in SSD300 implemented in Caffe looks similar to:
```
layer {
	name: "conv6_2_mbox_priorbox"
	type: "PriorBox"
	bottom: "conv6_2"
	bottom: "data"
	top: "conv6_2_mbox_priorbox"
	prior_box_param {
		min_size: 111.0
		max_size: 162.0
		aspect_ratio: 2
		aspect_ratio: 3
		flip: true
		clip: false
		variance: 0.1
		variance: 0.1
		variance: 0.2
		variance: 0.2
		step: 32
		offset: 0.5
	}
}
```


## Parameters

This plugin has the plugin creator class `PriorBoxPluginCreator` and the plugin class `PriorBox`.

The `PriorBox` instance is created using `PriorBoxParameters`. The `PriorBoxParameters` is defined in [NvInferPlugin.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/structnvinfer1_1_1plugin_1_1_prior_box_parameters.html).  It consists of the following parameters:

| Type        | Parameter                | Description
|-------------|--------------------------|--------------------------------------------------------
|`float *`    |`minSize`                 |The minimum box size in pixels. Can not be `nullptr`. `minSize` points to a series of minimum box size values which are used to generate prior boxes with different aspect ratios. The width of prior box `w` is equal to one of the minimum box size values times the square root of one of the values from aspect ratios. The width of prior box `h` is equal to one of the minimum box size values divided by the square root of one of the values from aspect ratios. For example, to generate a prior box of min size = 30 with aspect ratio = 2, the width and height of the prior bounding box generated are 42 and 21 respectively. In the original SSD paper, only minimum box size value is provided for each feature map.
|`float *`    |`maxSize`                 |The maximum box size in pixels. Can be `nullptr`. `maxSize` points to a series of maximum box size values which are used to generate additional prior boxes with aspect ratio `1`. The width of prior box is equal to the square root of one of the minimum box sizes values times the square root of its corresponding maximum box size value. For example, if min size = 30 and max size = 60, an additional prior box of width 42 and height 42 will be generated. In the original SSD paper, only one maximum box size value is provided for each feature map.
|`float *`    |`aspectRatios`            |The aspect ratios of the boxes. Can be `nullptr`. There is a built-in default aspect ratio of `1`. Therefore, it is not required to provide aspect ratio of `1` here. For example, if `aspectRatios = [2, 3]`, if `flip = true`, aspect ratios actually used is `[1, 2, 1/2, 3, 1/3]`; and if `flip = false`, aspect ratios actually used is `[1, 2, 3]`.
|`int`        |`numMinSize`              |The number of elements in `minSize`. Must be larger than `0`.
|`int`        |`numMaxSize`              |The number of elements in `maxSize`. Can be `0` or same as `numMinSize`.
|`int`        |`numAspectRatios`         |The number of elements in `aspectRatios`. Can be `0`.
|`bool`       |`flip`                    |If `true`, will flip each aspect ratio. For example, if there is aspect ratio `r`, the aspect ratio `1.0/r` will be generated as well.
|`bool`       |`clip`                    |If `true`, will clip the prior so that it is within `[0,1]`. Some prior boxes generated close to the border of the image will have coordinates larger than `1.0` or smaller than `0`. Setting `clip = true` will clip the out-of range coordinates so that all the coordinates fall into `[0, 1]`.
|`float`      |`variance [4]`            |The variances (scale factors) for adjusting the prior box coordinates encoding and decoding.
|`int`        |`imgH`                    |The image height. If `0`, then the `H` dimension of the data tensor will be used. The height of the image input to the model. For example, for SSD300 model, `imgH = 300`.
|`int`        |`imgW`                    |The image width. If `0`, then the `W` dimension of the data tensor will be used. The width of the image input to the model. For example, for SSD300 model, `imgW = 300`.
|`float`      |`stepH`                   |The step in `H`. If `0`, then `(float)imgH/h` will be used where `h` is the `H` dimension of the first input tensor. For example, for SSD300 model, `imgH = 300` and the height of the first feature map is 38 x 38. Then, `stepH = 300 / 38 = 7.895`.
|`float`      |`stepW`                   |The step in `W`. If `0`, then `(float)imgW/w` will be used where `w` is the `W` dimension of the first input tensor. For example, for SSD300 model, `imgW = 300` and the width of the first feature map is 38 x 38. Then, `stepW = 300 / 38 = 7.895`.
|`float`      |`offset`                  |Offset to the top left corner of each cell. This value is usually set to `0.5` to make sure that the prior boxes generated have centroid located at the center of the grid in the feature map.


## Additional resources

The following resources provide a deeper understanding of the `priorBoxPlugin` plugin:

**Networks**
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)    


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.