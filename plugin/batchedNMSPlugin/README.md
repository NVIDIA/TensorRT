# batchedNMSPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Algorithms](#algorithms)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `batchedNMSPlugin` implements a non-maximum suppression (NMS) step over boxes for object detection networks.

Non-maximum suppression is typically the universal step in object detection inference. This plugin is used after you’ve processed the bounding box prediction and object classification to get the final bounding boxes for objects.
  
With this plugin, you can incorporate the non-maximum suppression step during TensorRT inference. During inference, the neural network generates a fixed number of bounding boxes with box coordinates, identified class and confidence levels. Not all bounding boxes, but the most representative ones, have to be drawn on the original image.

Non-maximum suppression is the way to eliminate the boxes which have low confidence or do not have object in and keep the most representative ones. For example, the objects within an image might be covered by many boxes with different levels of confidence. The goal of the non-maximum suppression step is to find the most confident box for the object and remove all the less confident ones.

This plugin accelerates this non maximum suppression step during TensorRT inference on GPU.

### Structure

The `batchedNMSPlugin` takes two inputs, boxes input and scores input.

**Boxes input**
The boxes input are of shape `[batch_size, number_boxes, number_classes, number_box_parameters]`. The box location usually consists of four parameters such as `[x1, y1, x2, y2]` where (x1, y1) and (x2, y2) are the coordinates of any diagonal pair of box corners. For example, if your model outputs `8732` bounding boxes given one image, there are `100` candidate classes, the shape of boxes input will be `[8732, 100, 4]`.

**Scores input**
The scores input are of shape `[batch_size, number_boxes, number_classes]`. Each box has an array of probability for each candidate class.

The boxes input and scores input generates the following four outputs:

- `num_detections`
The `num_detections` input are of shape `[batch_size, 1]`. The last dimension of size 1 is an INT32 scalar indicating the number of valid detections per batch item. It can be less than `keepTopK`. Only the top `num_detections[i]` entries in `nmsed_boxes[i]`, `nmsed_scores[i]` and `nmsed_classes[i]` are valid.

- `nmsed_boxes`
A `[batch_size, keepTopK, 4]` float32 tensor containing the coordinates of non-max suppressed boxes.

- `nmsed_scores`
A `[batch_size, keepTopK]` float32 tensor containing the scores for the boxes.

- `nmsed_classes`
A `[batch_size, keepTopK]` float32 tensor containing the classes for the boxes.


## Parameters

The `batchedNMSPlugin` has plugin creator class `BatchedNMSPluginCreator` and plugin class `BatchedNMSPlugin`.

The `batchedNMSPlugin` is created using `BatchedNMSPluginCreator` with `NMSParameters` typed parameters. The `NMSParameters` data structure is listed as follows and is defined in the [NvInferPlugin.h header file](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_plugin_8h_source.html).

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`bool`    |`shareLocation`           |If set to `true`, the boxes input are shared across all classes. If set to `false`, the boxes input should account for per-class box data.
|`int`     |`backgroundLabelId`       |The label ID for the background class. If there is no background class, set it to `-1`.
|`int`     |`numClasses`              |The number of classes in the network.
|`int`     |`topK`                    |The number of bounding boxes to be fed into the NMS step.
|`int`     |`keepTopK`                |The number of total bounding boxes to be kept per-image after the NMS step. Should be less than or equal to the `topK` value.
|`float`   |`scoreThreshold`          |The scalar threshold for score (low scoring boxes are removed).
|`float`   |`iouThreshold`            |The scalar threshold for IOU (new boxes that have high IOU overlap with previously selected boxes are removed).
|`bool`    |`isNormalized`            |Set to `false` if the box coordinates are not normalized, meaning they are not in the range `[0,1]`. Defaults to `true`.
|`bool`    |`clipBoxes`               |Forcibly restrict bounding boxes to the normalized range `[0,1]`. Only applicable if `isNormalized` is also `true`. Defaults to `true`.
|`int`     |`scoreBits`               |The number of bits to represent the score values during radix sort. The number of bits to represent score values(confidences) during radix sort. This valid range is 0 < scoreBits <= 10. The default value is 16(which means to use full bits in radix sort). Setting this parameter to any invalid value will result in the same effect as setting it to 16. This parameter can be tuned to strike for a best trade-off between performance and accuracy. Lowering scoreBits will improve performance but with some minor degradation to the accuracy. This parameter is only valid for FP16 data type for now.

## Algorithms

The NMS algorithm used in this particular plugin first sorts the bounding boxes indices by the score for each class, then sorts the bounding boxes by the updated scores, and finally collects the desired number of bounding boxes with the highest scores.

It is mainly accelerated using the `nmsInference` kernel defined in the `batchedNMSInference.cu` file.

Specifically, the NMS algorithm:
- Sorts the bounding box indices by the score for each class. Before sorting, the bounding boxes with a score less than `scoreThreshold` are discarded by setting their indices to `-1` and their scores to `0`. This is using the `sortScoresPerClass` kernel defined in the `sortScoresPerClass.cu` file.

- Finds the most confident box for the object and removes all the less confident ones using the iterative non-maximum suppression step step for each class. Starting from the bounding box with the highest score in each class, the bounding boxes that has overlap higher than `iouThreshold` is suppressed by setting their indices to `-1` and their scores to `0`. Then all the less confident bounding boxes were suppressed for each class. This is using the `allClassNMS` kernel defined in the `allClassNMS.cu` file.

- Sorts the bounding boxes per image using the updated scores. At this time, all the classes were mixed before sort. Discarded and suppressed bounding boxes will go to the end of the sorted array since their score is `0`. This is using the `sortScoresPerImage` kernel defined in the `sortScoresPerImage.cu` file.
  
- Collects the desired number, `keepTopK`, of bounding box indices with the highest scores from the top of the sorted array, their bounding box coordinates, and their object classification information. This is using the `gatherNMSOutputs` kernel defined in the `gatherNMSOutputs.cu` file.


## Additional resources

The following resources provide a deeper understanding of the `batchedNMSPlugin` plugin:

**Networks**
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)    
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)    
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)


**Documentation**
- [NMSParameter detailed descriptions](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/structnvinfer1_1_1plugin_1_1_n_m_s_parameters.html)
- [NMS algorithm](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


## Changelog

May 2019
This is the first release of this `README.md` file.


## Known issues

- When running `cub::DeviceSegmentedRadixSort::SortPairsDescending` with `cuda-memcheck --tool racecheck`, it will not work correctly.
- BatchedNMS plugin cannot handle greater than 4096 rectangles in the input.
