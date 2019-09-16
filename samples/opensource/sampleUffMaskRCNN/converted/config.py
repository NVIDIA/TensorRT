#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import graphsurgeon as gs
import tensorflow as tf

fpn_p5upsampled = gs.create_plugin_node("fpn_p5upsampled", op="ResizeNearest_TRT", dtype=tf.float32, scale=2.0)
fpn_p4upsampled = gs.create_plugin_node("fpn_p4upsampled", op="ResizeNearest_TRT", dtype=tf.float32, scale=2.0)
fpn_p3upsampled = gs.create_plugin_node("fpn_p3upsampled", op="ResizeNearest_TRT", dtype=tf.float32, scale=2.0)

roi = gs.create_plugin_node("ROI", op="ProposalLayer_TRT", prenms_topk=1024, keep_topk=1000, iou_threshold=0.7)
roi_align_classifier = gs.create_plugin_node("roi_align_classifier", op="PyramidROIAlign_TRT", pooled_size=7)
mrcnn_detection = gs.create_plugin_node("mrcnn_detection", op="DetectionLayer_TRT", num_classes=81, keep_topk=100, score_threshold=0.7, iou_threshold=0.3)
roi_align_mask = gs.create_plugin_node("roi_align_mask_trt", op="PyramidROIAlign_TRT", pooled_size=14)
mrcnn_detection_bboxes = gs.create_plugin_node("mrcnn_detection_bboxes", op="SpecialSlice_TRT")

namespace_plugin_map = {
"fpn_p5upsampled":fpn_p5upsampled,

"fpn_p4upsampled":fpn_p4upsampled,

"fpn_p3upsampled":fpn_p3upsampled,

"roi_align_classifier":roi_align_classifier,

"mrcnn_detection":mrcnn_detection,

"ROI":roi,

"roi_align_mask":roi_align_mask,

"lambda_1": mrcnn_detection_bboxes,

}

timedistributed_remove_list = [
        "mrcnn_class_conv1/Reshape/shape", "mrcnn_class_conv1/Reshape", "mrcnn_class_conv1/Reshape_1/shape", "mrcnn_class_conv1/Reshape_1",
        "mrcnn_class_bn1/Reshape/shape", "mrcnn_class_bn1/Reshape", "mrcnn_class_bn1/Reshape_5/shape", "mrcnn_class_bn1/Reshape_5",
        "mrcnn_class_conv2/Reshape/shape", "mrcnn_class_conv2/Reshape", "mrcnn_class_conv2/Reshape_1/shape", "mrcnn_class_conv2/Reshape_1",
        "mrcnn_class_bn2/Reshape/shape", "mrcnn_class_bn2/Reshape", "mrcnn_class_bn2/Reshape_5/shape", "mrcnn_class_bn2/Reshape_5",
        "mrcnn_class_logits/Reshape/shape", "mrcnn_class_logits/Reshape","mrcnn_class_logits/Reshape_1/shape", "mrcnn_class_logits/Reshape_1",
        "mrcnn_class/Reshape/shape", "mrcnn_class/Reshape","mrcnn_class/Reshape_1/shape", "mrcnn_class/Reshape_1",
        "mrcnn_bbox_fc/Reshape/shape", "mrcnn_bbox_fc/Reshape","mrcnn_bbox_fc/Reshape_1/shape", "mrcnn_bbox_fc/Reshape_1",

        "mrcnn_mask_conv1/Reshape/shape", "mrcnn_mask_conv1/Reshape", "mrcnn_mask_conv1/Reshape_1/shape", "mrcnn_mask_conv1/Reshape_1",
        "mrcnn_mask_bn1/Reshape/shape", "mrcnn_mask_bn1/Reshape", "mrcnn_mask_bn1/Reshape_5/shape", "mrcnn_mask_bn1/Reshape_5",
        "mrcnn_mask_conv2/Reshape/shape", "mrcnn_mask_conv2/Reshape", "mrcnn_mask_conv2/Reshape_1/shape", "mrcnn_mask_conv2/Reshape_1",
        "mrcnn_mask_bn2/Reshape/shape", "mrcnn_mask_bn2/Reshape", "mrcnn_mask_bn2/Reshape_5/shape", "mrcnn_mask_bn2/Reshape_5",
        "mrcnn_mask_conv3/Reshape/shape", "mrcnn_mask_conv3/Reshape", "mrcnn_mask_conv3/Reshape_1/shape", "mrcnn_mask_conv3/Reshape_1",
        "mrcnn_mask_bn3/Reshape/shape", "mrcnn_mask_bn3/Reshape", "mrcnn_mask_bn3/Reshape_5/shape", "mrcnn_mask_bn3/Reshape_5",
        "mrcnn_mask_conv4/Reshape/shape", "mrcnn_mask_conv4/Reshape", "mrcnn_mask_conv4/Reshape_1/shape", "mrcnn_mask_conv4/Reshape_1",
        "mrcnn_mask_bn4/Reshape/shape", "mrcnn_mask_bn4/Reshape", "mrcnn_mask_bn4/Reshape_5/shape", "mrcnn_mask_bn4/Reshape_5",
        "mrcnn_mask_deconv/Reshape/shape", "mrcnn_mask_deconv/Reshape", "mrcnn_mask_deconv/Reshape_1/shape", "mrcnn_mask_deconv/Reshape_1",
        "mrcnn_mask/Reshape/shape", "mrcnn_mask/Reshape", "mrcnn_mask/Reshape_1/shape", "mrcnn_mask/Reshape_1",
        ]

timedistributed_connect_pairs = [
        ("mrcnn_mask_deconv/Relu", "mrcnn_mask/convolution"), # mrcnn_mask_deconv -> mrcnn_mask
        ("activation_74/Relu", "mrcnn_mask_deconv/conv2d_transpose"), #active74 -> mrcnn_mask_deconv
        ("mrcnn_mask_bn4/batchnorm/add_1","activation_74/Relu"),  # mrcnn_mask_bn4 -> active74
        ("mrcnn_mask_conv4/BiasAdd", "mrcnn_mask_bn4/batchnorm/mul_1"), #mrcnn_mask_conv4 -> mrcnn_mask_bn4
        ("activation_73/Relu", "mrcnn_mask_conv4/convolution"), #active73 -> mrcnn_mask_conv4
        ("mrcnn_mask_bn3/batchnorm/add_1","activation_73/Relu"), #mrcnn_mask_bn3 -> active73
        ("mrcnn_mask_conv3/BiasAdd", "mrcnn_mask_bn3/batchnorm/mul_1"), #mrcnn_mask_conv3 -> mrcnn_mask_bn3
        ("activation_72/Relu", "mrcnn_mask_conv3/convolution"), #active72 -> mrcnn_mask_conv3
        ("mrcnn_mask_bn2/batchnorm/add_1","activation_72/Relu"), #mrcnn_mask_bn2 -> active72
        ("mrcnn_mask_conv2/BiasAdd", "mrcnn_mask_bn2/batchnorm/mul_1"), #mrcnn_mask_conv2 -> mrcnn_mask_bn2
        ("activation_71/Relu", "mrcnn_mask_conv2/convolution"), #active71 -> mrcnn_mask_conv2
        ("mrcnn_mask_bn1/batchnorm/add_1","activation_71/Relu"), #mrcnn_mask_bn1 -> active71
        ("mrcnn_mask_conv1/BiasAdd", "mrcnn_mask_bn1/batchnorm/mul_1"), #mrcnn_mask_conv1 -> mrcnn_mask_bn1
        ("roi_align_mask_trt", "mrcnn_mask_conv1/convolution"), #roi_align_mask -> mrcnn_mask_conv1


        ("mrcnn_class_bn2/batchnorm/add_1","activation_69/Relu"), # mrcnn_class_bn2 -> active 69
        ("mrcnn_class_conv2/BiasAdd", "mrcnn_class_bn2/batchnorm/mul_1"), # mrcnn_class_conv2 -> mrcnn_class_bn2
        ("activation_68/Relu", "mrcnn_class_conv2/convolution"), # active 68 -> mrcnn_class_conv2
        ("mrcnn_class_bn1/batchnorm/add_1","activation_68/Relu"), # mrcnn_class_bn1 -> active 68
        ("mrcnn_class_conv1/BiasAdd", "mrcnn_class_bn1/batchnorm/mul_1"), # mrcnn_class_conv1 -> mrcnn_class_bn1
        ("roi_align_classifier", "mrcnn_class_conv1/convolution"), # roi_align_classifier -> mrcnn_class_conv1
        ]

dense_compatible_patch =["pool_squeeze/Squeeze", "pool_squeeze/Squeeze_1", #No need to squeeze the dimensions for TRT Dense Layer
        "mrcnn_bbox/Shape", "mrcnn_bbox/strided_slice/stack", # mrcnn_bbox(Reshape): No need to reshape, cause we can process it as 1-D array in detectionlayer's kernel
        "mrcnn_bbox/strided_slice/stack_1", "mrcnn_bbox/strided_slice/stack_2",
        "mrcnn_bbox/strided_slice", "mrcnn_bbox/Reshape/shape/1",
        "mrcnn_bbox/Reshape/shape/2", "mrcnn_bbox/Reshape/shape/3",
        "mrcnn_bbox/Reshape/shape", "mrcnn_bbox/Reshape"]

dense_compatible_connect_pairs = [
        ("activation_69/Relu","mrcnn_bbox_fc/MatMul"), #activation_69 -> mrcnn_bbox_fc
        ("activation_69/Relu", "mrcnn_class_logits/MatMul"), #activation_69 -> mrcnn_class_logits
        ("mrcnn_class_logits/BiasAdd", "mrcnn_class/Softmax"), #mrcnn_class_logits -> mrcnn_class
        ("mrcnn_class/Softmax", "mrcnn_detection"), #mrcnn_class -> mrcnn_detection
        ("mrcnn_bbox_fc/BiasAdd", "mrcnn_detection"), #mrcnn_bbox_fc -> mrcnn_detection
        ]

def connect(dynamic_graph, connections_list):

    for node_a_name, node_b_name in connections_list:
        if node_a_name not in dynamic_graph.node_map[node_b_name].input:
            dynamic_graph.node_map[node_b_name].input.insert(0, node_a_name)

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map, unique_inputs=True)
    dynamic_graph.remove(timedistributed_remove_list)
    dynamic_graph.remove(dense_compatible_patch)
    dynamic_graph.remove(['input_anchors', 'input_image_meta'])

    connect(dynamic_graph, timedistributed_connect_pairs)
    connect(dynamic_graph, dense_compatible_connect_pairs)

