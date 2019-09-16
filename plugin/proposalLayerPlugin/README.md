# ProposalLayer

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `ProposalLayer` plugin generates the first-stage detection (ROI candidates) out of the scores, refinement info from RPN(Region Proposal Network) and pre-defined anchors. It is
used in sampleMaskRCNN.   


### Structure

This plugin supports the NCHW format. It takes two input tenosrs: `object_score` and `object_delta` 

`object_score` is the objectness score from RPN. `objetc_score`'s shape is `[N, anchors, 2, 1]` where `N` is the batch_size, `anchors` is the total number of anchors and `2` means 2
classes of objectness --- foreground and background . 

`object_delta` is the refinement info from RPN of shape `[N, anchors, 4, 1]`. `4` means 4 elements of refinement information --- `[dy, dx, dh, dw]`

This plugin generates one output tensor of shape `[N, keep_topk, 4]` where `keep_topk` is the maximum number of detections left after NMS and `4` means coordinates of ROI
candidates `[y1, x1, y2, x2]`

Instead of fed as input in Keras, the default anchors are generated in this plugin during `initialization`.   
For resnet101 + 1024*1024 input shape, the number of anchors can be computed as 
```
Anchors in feature map P2: 256*256*3 
Anchors in feature map P3: 128*128*3
Anchors in feature map P4: 64*64*3
Anchors in feature map P5: 32*32*3
Anchors in feature map P6(maxpooling): 16*16*3  

total number of anchors: 87296*3 = 261888
```

## Parameters

This plugin has the plugin creator class `ProposalLayerPluginCreator` and the plugin class `ProposalLayer`.
  
The following parameters were used to create `ProposalLayer` instance:

| Type              | Parameter                        | Description
|-------------------|----------------------------------|--------------------------------------------------------
|`int`              |`prenms_topk`                     |The number of ROIs which will be kept before NMS. 
|`int`              |`keep_topk`                       |Number of detections will be kept after NMS.
|`float`            |`iou_threshold`                   |IOU threshold value used in NMS.


## Additional resources

The following resources provide a deeper understanding of the `ProposalLayer` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

June 2019
This is the first release of this `README.md` file.


## Known issues

There are no known issues in this plugin.
