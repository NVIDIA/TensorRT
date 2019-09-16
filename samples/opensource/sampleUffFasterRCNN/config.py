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
import tensorflow as tf
import graphsurgeon as gs


CropAndResize = gs.create_plugin_node(name='roi_pooling_conv_1/CropAndResize_new', op="CropAndResize", inputs=['activation_7/Relu', 'proposal'], crop_height=7, crop_width=7)
Proposal = gs.create_plugin_node(name='proposal', op='Proposal', inputs=['rpn_out_class/Sigmoid', 'rpn_out_regress/BiasAdd'],     input_height=272, input_width=480, rpn_stride=16, roi_min_size=1.0, nms_iou_threshold=0.7, pre_nms_top_n=6000, post_nms_top_n=300, anchor_sizes=[32.0, 64.0, 128.0], anchor_ratios=[1.0, 0.5, 2.0])


namespace_plugin_map = {
"crop_and_resize_1/Reshape" : CropAndResize,
'crop_and_resize_1/CropAndResize' : CropAndResize,
"crop_and_resize_1/transpose" : CropAndResize,
"crop_and_resize_1/transpose_1" : CropAndResize
}


def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.append(Proposal)
    dynamic_graph.remove('input_2')
    dynamic_graph.collapse_namespaces(namespace_plugin_map)

