# nmsPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
	* [`CodeType`](#codetype)
		* [`CodeTypeSSD::CORNER`](#codetypessd_corner)
		* [`CodeTypeSSD::CENTER_SIZE`](#codetypessdcenter_size)
		* [`CodeTypeSSD::CORNER_SIZE`](#codetypessdcorner_size)
		* [`CodeTypeSSD::TF_CENTER`](#codetypessdtf_center)
	* [`inputOrder`](#inputorder)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `nmsPlugin`, similar to the `batchedNMSPlugin`, implements a `non_max_suppression` (NMS) operation over bounding boxes for object detection networks. This plugin is included in TensorRT and used in [sampleSSD](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#sample_ssd) and [uff_ssd](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#uff_ssd) to run SSD.
  
Additionally, the `nmsPlugin` has a bounding box decoding step prior to the `non_max_suppression` step. The `nmsPlugin` takes the predicted encoded bounding box data as input, decodes them, followed by the non maximum suppression step in a GPU-accelerated fashion in TensorRT.


### Structure

This plugin takes three inputs, `loc_data`, `conf_data` and `prior_data` and generates one output containing the following information:
-   `image id`
-   `bounding box label`
-   `confidence score`
-   `bounding box labels`
    
and another output containing the number of valid detections for each batch item after non maximum suppression.

Where:
-   `loc_data` is the predicted bounding box data subject to decoding. It has a shape of `[batchSize, numPriors * numLocClasses * 4, 1, 1]`. Where:
	-   `numPriors` is the total number of prior boxes for one sample
	-   `numLocClasses` is 1 if each bounding box predicts the probability for all candidate classes, else the number of candidate classes if the bounding box only does binary prediction for one candidate class. For example, say you have 81 candidate classes, if your bounding box could predict the probability for all the 81 candidate classes, `numLocClasses` will be 1 in this case. This is also the original implementation of the SSD model. However, if your bounding box was designed to do binary classification, you would need 81 bounding boxes for one anchor box, `numLocClasses` will be 81 in this case.
-   `conf_data` is the object classification confidence probability distribution of the bounding box. It has a shape of `[batchSize, numPriors * numClasses, 1, 1]`.
-   `prior_data` is the anchor box data with additional scaling factor or variance used for bounding box encoding and decoding generated from the custom plugin `gridAnchorPlugin`. It has a shape of `[batchSize, 2, numPriors * 4, 1]`. The `prior_data` input has two channels.
	-   The first channel is the anchor box data.
	-   The second channel is the scaling factor or variance used for bounding box encoding and decoding.

After decoding, the decoded boxes will proceed to the non maximum suppression step, which performs the same action as `batchedNMSPlugin`. The only difference is that instead of generating four outputs:
-   `nmsed box count` (1 value)
-   `nmsed box locations` (4 values)
-   `nmsed box scores` (1 value)
-   `nmsed box class IDs` (1 value)

The `nmsPlugin` generates an output of shape `[batchSize, 1, keepTopK, 7]` which contains the same information as the outputs `nmsed box locations`, `nmsed box scores`, and `nmsed box class IDs` from `batchedNMSPlugin`, and an another output of shape `[batchSize, 1, 1, 1]` which contains the same information as the output `nmsed box count` from `batchedNMSPlugin`.

## Parameters

The plugin has the plugin creator class `NMSPluginCreator` and the plugin class `DetectionOutput`.
  
The `DetectionOutput` plugin instance is created using an array of `DetectionOutputParameters` type parameters. `DetectionOutputParameters` consists of the following parameters:

| Type             | Parameter                      | Description
|------------------|--------------------------------|--------------------------------------------------------
|`bool`            |`shareLocation`                 |If `true`, the bounding boxes are shared among different classes.
|`bool`            |`varianceEncodedInTarget`       |If `true`, variance is encoded in target, you will not use the variance values to adjust the predicted bounding box. Otherwise, you will use the variance values to adjust the predicted bounding box accordingly.
|`int`             |`backgroundLabelId`             |The background label ID. If there is no background class, set it to `-1`.
|`int`             |`numClasses`                    |Number of classes to be predicted.
|`int`             |`topK`                          |Number of boxes per image with top confidence scores that are fed into the NMS algorithm.
|`int`             |`keepTopK`                      |Number of total bounding boxes to be kept per image after the NMS step.
|`float`           |`confidenceThreshold`           |Considers detections whose confidences are larger than a threshold.
|`float`           |`nmsThreshold`                  |Intersection over union (IoU) threshold to be used in NMS.
|`codeTypeSSD`     |`codeType`                      |Type of coding method for `bbox`.
|`int`             |`inputOrder`                    |Specifies the order of inputs `{loc_data, conf_data, priorbox_data}`, in other words, `inputOrder[0]` is for `loc_data`, `inputOrder[1]` is for `conf_data` and `inputOrder[2]` is for `priorbox_data`. For example, if your inputs in the memory are in the order of `loc_data`, `priorbox_data`, `conf_data`, then `inputOrder` should be `[0, 2, 1]`.
|`bool`            |`confSigmoid`                   |Set to `true` to calculate sigmoid of confidence scores.
|`bool`            |`isNormalized`                  |Set to `true` if bounding box data is normalized by the network, in other words, the bounding box coordinates used in the model are not pixel coordinates.
|`int`             |`scoreBits`                     |The number of bits to represent the score values during radix sort. The number of bits to represent score values(confidences) during radix sort. This valid range is 0 < scoreBits <= 10. The default value is 16(which means to use full bits in radix sort). Setting this parameter to any invalid value will result in the same effect as setting it to 16. This parameter can be tuned to strike for a best trade-off between performance and accuracy. Lowering scoreBits will improve performance but with some minor degradation to the accuracy. This parameter is only valid for FP16 data type for now.

### `CodeType`

The bounding boxes are used in an encoded format in the model. In order to get the exact bounding box coordinates on the original input image, we need to know the encoding method and decode them.

The bounding box is decoded using the `decodeBBoxes` CUDA kernel defined in the `decodeBBoxes.cu` file based on the encoding and decoding method used during model training. Currently, we support the following encoding methods:
- [`CodeTypeSSD::CORNER`](#codetypessd_corner)
- [`CodeTypeSSD::CENTER_SIZE`](#codetypessdcenter_size)
- [`CodeTypeSSD::CORNER_SIZE`](#codetypessdcorner_size)
- [`CodeTypeSSD::TF_CENTER`](#codetypessdtf_center)
  
`CodeTypeSSD` is defined in [NvInferPlugin.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_plugin_8h_source.html) and has a brief description on [NvInferPlugin.h File Reference](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_plugin_8h.html). The mathematical formulation of the coding methods are listed below.

#### `CodeTypeSSD::CORNER`

Without using or having variance encoded, the encoded bounding box representation is:
```
[x_{min, gt} - x_{min, anchor}, y_{min, gt} - y_{min, anchor}, x_{max, gt} - x_{max, anchor}, y_{max, gt} - y_{max, anchor}]
```

Using or having variance encoded, the encoded bounding box representation is:
```
[(x_{min, gt} - x_{min, anchor}) / variance_0, (y_{min, gt} - y_{min, anchor}) / variance_1, (x_{max, gt} - x_{max, anchor}) / variance_2, (y_{max, gt} - y_{max, anchor}) / variance_3]
```

#### `CodeTypeSSD::CENTER_SIZE`

Without using or having variance encoded, the encoded bounding box representation is:
```
[(x_{center, gt} - x_{center, anchor}) / w_{anchor}, (y_{center, gt} - y_{center, anchor}) / h_{anchor}, ln(w_{gt} / w_{anchor}), ln(h_{gt} / h_{anchor})]
```

Using or having variance encoded, the encoded bounding box representation is:
```
[(x_{center, gt} - x_{center, anchor}) / w_{anchor} / variance_0, (y_{center, gt} - y_{center, anchor}) / h_{anchor} / variance_1, ln(w_{gt} / w_{anchor}) / variance_2, ln(h_{gt} / h_{anchor}) / variance_3]
```

#### `CodeTypeSSD::CORNER_SIZE`

Without using or having variance encoded, the encoded bounding box representation is:
```
[(x_{min, gt} - x_{min, anchor}) / w_{anchor}, (y_{min, gt} - y_{min, anchor}) / h_{anchor}, (x_{max, gt} - x_{max, anchor}) / w_{anchor}, (y_{max, gt} - y_{max, anchor}) / h_{anchor}]
```

Using or having variance encoded, the encoded bounding box representation is:
```
[(x_{min, gt} - x_{min, anchor}) / w_{anchor} / variance_0, (y_{min, gt} - y_{min, anchor}) / h_{anchor} / variance_1, (x_{max, gt} - x_{max, anchor}) / w_{anchor} / variance_2, (y_{max, gt} - y_{max, anchor}) / h_{anchor} / variance_3]
```

#### `CodeTypeSSD::TF_CENTER`

Using or having variance encoded, the encoded bounding box representation is:
```
[(y_{center, gt} - y_{center, anchor}) / h_{anchor} / variance_0, (x_{center, gt} - x_{center, anchor}) / w_{anchor} / variance_1, ln(h_{gt} / h_{anchor}) / variance_2, ln(w_{gt} / w_{anchor}) / variance_3]
```

**Note:** This code is almost the same to `CodeTypeSSD::CENTER_SIZE` using variance encoded except that the order of coordinates were different.

### `inputOrder`

When converting the frozen graph `pb` file to the unified framework format `uff` file using `convert-to-uff`, make sure to generate a human readable graph file with argument `-t`. The order of the tensor inputs to the plugin will be exactly the same to the order of tensor inputs in the corresponding node shown in the human readable graph file.

## Additional resources

The following resources provide a deeper understanding of the `nmsPlugin` plugin:

**Networks:**
-   [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
-   [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
