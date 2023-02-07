# MultilevelProposeROI

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `MultilevelProposeROI` plugin generates the first-stage detection (ROI candidates) from the scores, refinement information from RPN(Region Proposal Network) and pre-defined anchors. It is
used in sampleMaskRCNN.   


### Structure

This plugin supports the NCHW format. It takes two input tensors: `object_score` and `object_delta` 

`object_score` is the objectness score from RPN. `object_score`'s shape is `[N, anchors, 2, 1]` where `N` is the batch_size, `anchors` is the total number of anchors and `2` means 2
classes of objectness --- foreground and background . 

`object_delta` is the refinement information from RPN of shape `[N, anchors, 4, 1]`. `4` means 4 elements of refinement information --- `[dy, dx, dh, dw]`

This plugin generates one output tensor of shape `[N, keep_topk, 4]` where `keep_topk` is the maximum number of detections left after NMS and `4` means coordinates of ROI
candidates `[y1, x1, y2, x2]`

Instead of fed as input in Keras, the default anchors used in this plugin are generated upon `initialization`.   
For resnet50 + 832*1344 input shape, the number of anchors can be computed as 
```
Anchors in feature map P2: 208*336*3
Anchors in feature map P3: 104*168*3
Anchors in feature map P4: 52*84*3
Anchors in feature map P5: 26*42*3
Anchors in feature map P6(maxpooling): 13*21*3

```

## Parameters

This plugin has the plugin creator class `MultilevelProposeROIPluginCreator` and the plugin class `MultilevelProposeROI`.
  
The following parameters were used to create `MultilevelProposeROI` instance:

| Type              | Parameter                        | Description
|-------------------|----------------------------------|--------------------------------------------------------
|`int`              |`prenms_topk`                     |The number of ROIs which will be kept before NMS. 
|`int`              |`keep_topk`                       |Number of detections will be kept after NMS.
|`float`            |`iou_threshold`                   |IOU threshold value used in NMS.
|`int[3]`           |`image_size`                      |Input image shape in CHW. Defaults to [3, 832, 1344]

## Limitations

The attribute `prenms_topk` is capped at 4096 to support embedded devices with smaller shared memory capacity.

To enable support for a device with higher memory, calls to `sortPerClass`, `PerClassNMS` and `KeepTopKGatherBoxScore` can be modified in `MultilevelPropose` ([maskRCNNKernels.cu](https://github.com/NVIDIA/TensorRT/blob/main/plugin/common/kernels/maskRCNNKernels.cu)).

## Additional resources


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

January 2022: The [Limitations](#limitations) section was added to this `README.md` file to document limitations of the plugin related to the maximum number of anchors it can support. 

June 2020: This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
