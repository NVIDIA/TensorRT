# **Model Conversion Guideline**
This is the general guideline to convert a keras model to .uff.

## **General Conversion steps and tips** 

- Build up the NCHW inference architecture:
    - If the model is trained in NHWC, we should make sure NCHW architecture can consume the pretrained weights. Generally, most layers could work well directly in NHWC -> NCHW conversion except **Reshape**, **Flatten**, **Dense** and **Softmax** applied to feature map
    - Input shape of model should be set.
- Using tensorflow's graph_utils, graph_io API to convert keras model to .pb.
- Converting .pb to .uff using uff converter:
    - We should map all the tf nodes which are not supported by TensorRT directly to plugin node.
    - graphsurgeon cannot handle situation like:

    ```
    input: "tf_node"
    input: "tf_node:1"
    input: "tf_node:2"
    ```
    although above inputs come from a same node "tf_node" but the graphsurgeon would regard them as output from 3 different nodes. So to get rid of this situation, we need to add "tf_node:1" and "tf_node:2" to plugin map dict (even they are not really nodes)
    - the convolution, deconvolution and dense layer's input orders matter. Generally, the inputs[0] should be the feature map input and inputs[1] of node should be the kernel weights. So in most cases, we should reverse the input list when connect the broken graph using `node.input.append(other_node.name)`(The example code is in `config.py`):
    - the **CPU** version tensorflow's NCHW convolution would introduce the nodes graph like:
    ```
    transpose_NCHW2NHWC -> Convolution_NHWC -> transpose_NHWC2NCHW
    ```
    Therefore we recommend installing tensorflow-gpu.
    - The UFF Parser cannot handle the inputs with more than one index dimension for convolution, deconvolution and softmax correctly. So a patched lib is needed to parse .uff with
      nodes whose input tensor has more than one index dimension (this will be the next TensorRT release).

## **Usage for MaskRCNN conversion scripts**

### Dependency:
 - [Matterport's Mask_RCNN v2.0 Release](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0)
 - graphsurgeon (provided with TensorRT binary release package)
 - uff (provided with TensorRT binary release package)

### Steps

- Update the Mask_RCNN model from NHWC to NCHW:
  
    1) Set default image data format :
    ```python
    import keras.backend as K
    K.set_image_data_format('channels_first')
    ```
    2) change all BN layers from NHWC to NCHW:
    ```python
    # In function: indentity_block, conv_block, fpn_classifier_graph, build_fpn_mask_graph 
    x = BatchNorm(name=bn_name_base + '2a', axis=1)(x, training=train_bn)
    x = KL.TimeDistributed(BatchNorm(axis=1), name='mrcnn_class_bn1')(x, training=train_bn)
    ```
    3) Modify class `PyramidROIAlign` to be compatible with NCHW format:
    
    - wrap `permutation` with `crop_and_resize` because `tf.crop_and_resize` only supports `NHWC` format:
    ```python
    def NCHW_crop_and_resize(feature_map, level_boxes, box_indices, crop_size, method="bilinear"):
        # NCHW(0,1,2,3) -> NHWC(0,2,3,1):
        feature_map = tf.transpose(feature_map, [0, 2, 3, 1])

        # crop_and_resize:
        box_feature = tf.image.crop_and_resize(feature_map, level_boxes,
                    box_indices, crop_size, method=method)
    
        # NHWC(0,1,2,3) -> NCHW(0,3,1,2)
        box_feature = tf.transpose(box_feature, [0, 3, 1, 2])
    
        return box_feature
    
    pooled.append(NCHW_crop_and_resize(feature_maps[i], level_boxes,
        box_indices, self.pool_shape, method="bilinear"))
    ```
    - Change the `compute_output_shape` to return `NCHW` shape:
    ```python
    return input_shape[0][:2]  + (input_shape[2][1], ) + self.pool_shape
    ```
    4) Change the input format in function `build_rpn_model`:
    ```python
    input_feature_map = KL.Input(shape=[depth, None, None],
                                 name="input_rpn_feature_map")
    ```
    5) Permute the feature in function `rpn_graph` and change 'lambda' to 'Reshape': 
    ```python
    x = KL.Permute((2,3,1))(x)
    rpn_class_logits = KL.Reshape((-1, 2))(x)
    
    x = KL.Permute((2,3,1))(x)
rpn_bbox = KL.Reshape((-1, 4))(x)
    ```
    6) Change squeeze axis in function `fpn_classifier_graph`:
    ```python
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 4), 3),
                       name="pool_squeeze")(x)
    ```
    7) Change the input format in function `build` of class `MaskRCNN:`
    ```python
shape=[config.IMAGE_SHAPE[2], 1024, 1024 ], name="input_image")
    ```
    8) (Optional) Change the input blob for prediction in function `detect` of class `MaskRCNN:`
    ```python
    molded_input_images = np.transpose(molded_images, (0, 3, 1, 2))
    detections, _, _, mrcnn_mask, _, _, _ =\
        self.keras_model.predict([molded_input_images, image_metas, anchors], verbose=0)
    mrcnn_mask = np.transpose(mrcnn_mask, (0, 1, 3, 4, 2))
    ```
- For conversion to UFF, please refer to [these instructions](https://github.com/NVIDIA/TensorRT/tree/main/samples/opensource/sampleUffMaskRCNN#generating-uff-model).

>  NOTE: For reference, the successful converted model should contain 3049 nodes.

