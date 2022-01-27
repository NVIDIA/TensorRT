# DetectionLayer

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `DetectionLayer` plugin performs bounding boxes refinement of MaskRCNN's detection head and generate the final detection output of MaskRCNN. It is used in sampleMaskRCNN.  


### Structure

This plugin supports the NCHW format. It takes three input tensors: `delta_bbox`, `score` and `roi`

`delta_bbox` is the refinement information of roi boxes generated from `ProposalLayer`. `delta_bbox` tensor's shape is `[N, rois, num_classes*4, 1, 1]` where `N` is batch size,
`rois` is the total number of ROI boxes candidates per image, and `num_classes*4` means 4 refinement elements (`[dy, dx, dh, dw]`) for each roi box as different classes.

`score` is the predicted class scores of ROI boxes generated from `ProposalLayer` of shape `[N, rois, num_classes, 1, 1]`. There is `argmax`operation in `Detectionlayer` to determine the final class of detection
candidates.   

`roi` is the coordinates of ROI boxes candidates from `ProposalLayer` of shape `[N, rois, 4]`. 

This plugin generates output of shape `[N, keep_topk, 6]` where `keep_topk` is the maximum number of detections left after NMS and '6' means 6 elements of an detection `[y1, x1, y2, x2,
class_label, score]`

## Parameters

This plugin has the plugin creator class `DetectionlayerPluginCreator` and the plugin class `Detectionlayer`.
  
The following parameters were used to create `Detectionlayer` instance:

| Type               | Parameter                          | Description
|--------------------|------------------------------------|--------------------------------------------------------
|`int`               |`num_classes`                       |Number of detection classes(including `background`). `num_classes=81` for COCO dataset
|`int`               |`keep_topk`                         |Number of detections will be kept after NMS.  
|`float`             |`score_threshold`                   |Confidence threshold value. This plugin will drop a detection if its class confidence(score) is under "score_threshold". 
|`float`             |`iou_threshold`                     |IOU threshold value used in NMS.

## Limitations

The number of anchors is capped at 1024 to support embedded devices with smaller shared memory capacity.

To enable support for a device with higher memory, calls to `sortPerClass`, `PerClassNMS` and `KeepTopKGather` can be modified in `RefineBatchClassNMS` ([maskRCNNKernels.cu](https://github.com/NVIDIA/TensorRT/blob/main/plugin/common/kernels/maskRCNNKernels.cu)).

## Additional resources

The following resources provide a deeper understanding of the `Detectionlayer` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

January 2022: The [Limitations](#limitations) section was added to this `README.md` file to document limitations of the plugin related to the maximum number of anchors it can support.  

June 2019: First release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
