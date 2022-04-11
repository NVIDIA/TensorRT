#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
import sys
import argparse
import logging

import tensorflow as tf
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnx import shape_inference
from tf2onnx import tfonnx, optimizer, tf_loader

try:
    from object_detection.utils import config_util
except ImportError:
    print("Could not import TFOD modules. Maybe you did not install TFOD API")
    print("Please install TensorFlow 2 Object Detection API, check https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md")
    sys.exit(1)

import onnx_utils

logging.basicConfig(level=logging.INFO)
logging.getLogger("ModelHelper").setLevel(logging.INFO)
log = logging.getLogger("ModelHelper")


class TFODGraphSurgeon:
    def __init__(self, saved_model_path, pipeline_config_path):
        """
        Constructor of the Model Graph Surgeon object, to do the conversion of an TFOD saved model
        to an ONNX-TensorRT parsable model.
        :param saved_model_path: The path pointing to the TensorFlow saved model to load.
        :param pipeline_config_path: The path pointing to the TensorFlow Object Detection API pipeline.config to load.
        """

        saved_model_path = os.path.realpath(saved_model_path)
        assert os.path.exists(saved_model_path)

        # Use tf2onnx to convert saved model to an initial ONNX graph.
        graph_def, inputs, outputs = tf_loader.from_saved_model(saved_model_path, None, None, "serve",
                                                                ["serving_default"])
        log.info("Loaded saved model from {}".format(saved_model_path))
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")
        with tf_loader.tf_session(graph=tf_graph):
            onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
        onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
        self.graph = gs.import_onnx(onnx_model)
        assert self.graph
        log.info("TF2ONNX graph created successfully")

        # Fold constants via ONNX-GS that TF2ONNX may have missed.
        self.graph.fold_constants()
        
        # Pipeline config parsing.
        pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        # Get input resolution.
        self.height, self.width = config_util.get_spatial_image_size(config_util.get_image_resizer_config(pipeline_config["model"]))
        
        # If your model is SSD, get characteristics accordingly from pipeline.config file.
        if pipeline_config["model"].HasField("ssd"):
            # Getting model characteristics.
            self.model = str(pipeline_config["model"].ssd.feature_extractor.type)
            self.first_stage_nms_score_threshold = float(pipeline_config["model"].ssd.post_processing.batch_non_max_suppression.score_threshold)
            self.first_stage_nms_iou_threshold = float(pipeline_config["model"].ssd.post_processing.batch_non_max_suppression.iou_threshold)
            self.first_stage_max_proposals = int(pipeline_config["model"].ssd.post_processing.batch_non_max_suppression.max_detections_per_class)
        # If your model is Faster R-CNN get it's characteristics from pipeline.config file.
        elif pipeline_config["model"].HasField("faster_rcnn"):  
            # Getting model characteristics.
            self.model = str(pipeline_config["model"].faster_rcnn.feature_extractor.type) 
            self.num_classes = pipeline_config["model"].faster_rcnn.num_classes
            self.first_stage_nms_score_threshold = float(pipeline_config["model"].faster_rcnn.first_stage_nms_score_threshold)       
            self.first_stage_nms_iou_threshold = float(pipeline_config["model"].faster_rcnn.first_stage_nms_iou_threshold)
            self.first_stage_max_proposals = int(pipeline_config["model"].faster_rcnn.first_stage_max_proposals)
            self.first_stage_crop_size = int(pipeline_config["model"].faster_rcnn.initial_crop_size)
            self.second_stage_nms_score_threshold = float(pipeline_config["model"].faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold)
            self.second_stage_iou_threshold = float(pipeline_config["model"].faster_rcnn.second_stage_post_processing.batch_non_max_suppression.iou_threshold)
            self.mask_height = None
            self.mask_width = None
            self.matmul_crop_and_resize = False
            # Check what kind of Crop and Resize operation is used
            if pipeline_config["model"].faster_rcnn.HasField("use_matmul_crop_and_resize"):
                self.matmul_crop_and_resize = pipeline_config["model"].faster_rcnn.use_matmul_crop_and_resize
            # If model is Mask R-CNN, get final instance segmentation masks resolution.
            if pipeline_config["model"].faster_rcnn.second_stage_box_predictor.mask_rcnn_box_predictor.HasField("mask_height") and pipeline_config["model"].faster_rcnn.second_stage_box_predictor.mask_rcnn_box_predictor.HasField("mask_width"):
                self.mask_height = int(pipeline_config["model"].faster_rcnn.second_stage_box_predictor.mask_rcnn_box_predictor.mask_height)    
                self.mask_width = int(pipeline_config["model"].faster_rcnn.second_stage_box_predictor.mask_rcnn_box_predictor.mask_width)
        else: 
            log.info("Given Model type is not supported")
            sys.exit(1)

        # List of supported models.
        supported_models = ["ssd_mobilenet_v2_keras", "ssd_mobilenet_v1_fpn_keras", "ssd_mobilenet_v2_fpn_keras", "ssd_resnet50_v1_fpn_keras", 
                            "ssd_resnet101_v1_fpn_keras", "ssd_resnet152_v1_fpn_keras", "faster_rcnn_resnet50_keras", "faster_rcnn_resnet101_keras", 
                            "faster_rcnn_resnet152_keras", "faster_rcnn_inception_resnet_v2_keras"]
        assert self.model in supported_models

        # Model characteristics.
        log.info("Model is {}".format(self.model))
        log.info("Height is {}".format(self.height))
        log.info("Width is {}".format(self.width))
        log.info("First NMS score threshold is {}".format(self.first_stage_nms_score_threshold))
        log.info("First NMS iou threshold is {}".format(self.first_stage_nms_iou_threshold))
        log.info("First NMS max proposals is {}".format(self.first_stage_max_proposals))
        if "faster_rcnn" in self.model:
            log.info("Number of classes is {}".format(self.num_classes))
            log.info("Crop and Resize output size is {}".format(self.first_stage_crop_size))
            log.info("Second NMS score threshold is {}".format(self.second_stage_nms_score_threshold))
            log.info("Second NMS iou threshold is {}".format(self.second_stage_iou_threshold))
            log.info("Using MatMul Crop and Resize: {}".format(self.matmul_crop_and_resize))
            if not (self.mask_height is None and self.mask_width is None):
                log.info("Mask height is {}".format(self.mask_height))
                log.info("Mask width is {}".format(self.mask_width))
        
        self.batch_size = None

    def sanitize(self):
        """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort, and fold constant inputs values.
        When possible, run shape inference on the ONNX graph to determine tensor shapes.
        """

        # Type of model requires different amount of sanitize iterations
        if "ssd" in self.model:
            sanitize_num = 6
        elif "faster_rcnn" in self.model:
            sanitize_num = 3

        for i in range(sanitize_num):
            count_before = len(self.graph.nodes)
            self.graph.cleanup().toposort()
            try:
                for node in self.graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(self.graph)
                model = shape_inference.infer_shapes(model)
                self.graph = gs.import_onnx(model)
            except Exception as e:
                log.info("Shape inference could not be performed at this time:\n{}".format(e))
            try:
                self.graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                log.error("This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your "
                          "onnx_graphsurgeon module. Error:\n{}".format(e))
                raise

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break

    def save(self, output_path):
        """
        Save the ONNX model to the given location.
        :param output_path: Path pointing to the location where to write out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        output_path = os.path.realpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model, output_path)
        log.info("Saved ONNX model to {}".format(output_path))

    def add_debug_output(self, debug):
        """
        Add a debug output to a given node. 
        :param debug: Name of the output you would like to debug.
        """
        tensors = self.graph.tensors()
        for n, name in enumerate(debug):
            if name not in tensors:
                log.warning("Could not find tensor '{}'".format(name))
            debug_tensor = gs.Variable(name="debug:{}".format(n), dtype=tensors[name].dtype)
            debug_node = gs.Node(op="Identity", name="debug_{}".format(n), inputs=[tensors[name]], outputs=[debug_tensor])
            self.graph.nodes.append(debug_node)
            self.graph.outputs.append(debug_tensor)
            log.info("Adding debug output '{}' for graph tensor '{}'".format(debug_tensor.name, name))

    def update_preprocessor(self, batch_size, input_format):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.
        :param batch_size: The batch size to use for the ONNX graph.
        :param input_format: Input tensor format, either NCHW or NHWC.
        """
        # Update batch size.
        self.batch_size = batch_size

        # Set input tensor shape.
        assert input_format in ["NCHW", "NHWC"]
        input_shape = [None] * 4
        if input_format == "NHWC":
          input_shape = [self.batch_size, self.height, self.width, 3]
        if input_format == "NCHW":
          input_shape = [self.batch_size, 3, self.height, self.width]
        self.graph.inputs[0].shape = input_shape
        self.graph.inputs[0].dtype = np.float32
        self.graph.inputs[0].name = "input_tensor"

        self.sanitize()
        log.info("ONNX graph input shape: {} [NCHW format set]".format(self.graph.inputs[0].shape))

        # Find the initial nodes of the graph, whatever the input is first connected to, and disconnect them.
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Get input tensor.
        # Convert to NCHW format if needed.
        input_tensor = self.graph.inputs[0]
        if input_format == "NHWC":
            input_tensor = self.graph.transpose("preprocessor/transpose", input_tensor, [0, 3, 1, 2])

        # Mobilenets' and inception's backbones preprocessor.
        if 'mobilenet' in self.model or 'inception_resnet' in self.model:
            mul_const = np.expand_dims(np.asarray([2 / 255], dtype=np.float32), axis=(0, 2, 3))
            sub_const = np.expand_dims(np.asarray([1], dtype=np.float32), axis=(0, 2, 3))
            mul_out = self.graph.op_with_const("Mul", "preprocessor/scale", input_tensor, mul_const)
            sub_out = self.graph.op_with_const("Sub", "preprocessor/mean", mul_out, sub_const)

        # Resnet backbones' preprocessor.
        elif 'resnet' in self.model:
            sub_const = np.expand_dims(np.asarray([255 * 0.485, 255 * 0.456, 255 * 0.406], dtype=np.float32), axis=(0, 2, 3))
            sub_out = self.graph.op_with_const("Sub", "preprocessor/mean", input_tensor, sub_const)
        
        # Backbone is not supported.
        else:
            log.info("Given model's backbone is not supported, pre-processor algorithm can't be generated")
            sys.exit(1)

        # Find first Conv node and connect preprocessor directly to it.
        conv_node = self.graph.find_node_by_op("Conv")
        log.info("Found {} node '{}' as stem entry".format(conv_node.op, conv_node.name))
        conv_node.inputs[0] = sub_out[0]

        # Disconnect the last node in one of the preprocessing branches with first TensorListStack parent node.
        concat_node = self.graph.find_node_by_op("Concat")
        concat_node.outputs = []

        # Disconnect the last node in second preprocessing branch with parent second TensorListStack node.
        tile_node = self.graph.find_node_by_op("Tile")
        tile_node.outputs = []

        # Reshape nodes tend to update the batch dimension to a fixed value of 1, they should use the batch size instead.
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) == gs.Constant and node.inputs[1].values[0] == 1:
                node.inputs[1].values[0] = self.batch_size

        self.sanitize()

    def find_head_end(self, head_name, descendant, end_op):
        # This helper function finds ends of Class Net and Box Net, based on a model type. 
        # :param head_name: This is a common name that nodes in either Class or Box Nets start with.
        # :param descendant: Descendant of head_name, identified by operation (Transpose, MatMul, etc.).
        # :param end_op: Operation of a node you would like to get in the end of each Net.
        # These end_op nodes bring together prediction data based on type of model.
        # The Class Net end node will have shape [batch_size, num_anchors, num_classes],
        # and the Box Net end node has the shape [batch_size, num_anchors, 4].
        # These end nodes can be be found by searching for all end_op's operation nodes and checking if the node two
        # steps above in the graph has a name that begins with one of head_names for Class Net and Box Net respectively.
        for node in [node for node in self.graph.nodes if node.op == descendant and head_name in node.name]:
            target_node = self.graph.find_descendant_by_op(node, end_op)
            log.info("Found {} node '{}' as the tip of {}".format(target_node.op, target_node.name, head_name))
            return target_node

    def extract_anchors_tensor(self, split):
        # This will find the anchors that have been hardcoded somewhere within the ONNX graph.
        # The function will return a gs.Constant that can be directly used as an input to the NMS plugin.
        # The anchor tensor shape will be [1, num_anchors, 4]. Note that '1' is kept as first dim, regardless of
        # batch size, as it's not necessary to replicate the anchors for all images in the batch.

        # The anchors are available (one per coordinate) hardcoded as constants within certain box decoder nodes.
        # Each of these four constants have shape [1, num_anchors], so some numpy operations are used to expand the
        # dims and concatenate them as needed.

        # These constants can be found by starting from the Box Net's split operation , and for each coordinate,
        # walking down in the graph until either an Add or specific Mul node is found. The second input on this nodes will
        # be the anchor data required.
        def get_anchor(output_idx, op, depth=5):
            node = self.graph.find_descendant_by_op(split.o(0, output_idx), op)
            for i in range(depth):
                if node.op == op:
                    # Input of size 1 is not anchor data 
                    if (node.inputs[1].values).size == 1: 
                        node = node.o()
                    # Find the node that with anchor data, multielement input
                    elif (node.inputs[1].values).size > 1:
                        assert node
                        val = np.squeeze(node.inputs[1].values)
                        return np.expand_dims(val.flatten(), axis=(0, 2))
                else:
                    node = node.o()
            return None
           
        anchors_y = get_anchor(0, "Add")
        anchors_x = get_anchor(1, "Add")
        anchors_h = get_anchor(2, "Mul")
        anchors_w = get_anchor(3, "Mul")

        batched_anchors = np.concatenate([anchors_y, anchors_x, anchors_h, anchors_w], axis=2)
        # Identify num of anchors without repetitions.
        num_anchors = int(batched_anchors.shape[1]/self.batch_size)
        # Trim total number of anchors in order to not have copies introduced by growing number of batch_size.
        anchors = batched_anchors[0:num_anchors,0:num_anchors]
        return gs.Constant(name="nms/anchors:0", values=anchors)
        
    def NMS(self, box_net_tensor, class_net_tensor, anchors_tensor, background_class, score_activation, iou_threshold, nms_score_threshold, user_threshold, nms_name=None):
        # Helper function to create the NMS Plugin node with the selected inputs. 
        # EfficientNMS_TRT TensorRT Plugin is suitable for our use case.
        # :param box_net_tensor: The box predictions from the Box Net.      
        # :param class_net_tensor: The class predictions from the Class Net.
        # :param anchors_tensor: The default anchor coordinates (from the extracted anchor constants)
        # :param background_class: The label ID for the background class.
        # :param score_activation: If set to True - apply sigmoid activation to the confidence scores during NMS operation, 
        # if false - no activation, pass one from the graph.
        # :param iou_threshold: NMS intersection over union threshold, given by pipeline.config.
        # :param nms_score_threshold: NMS score threshold, given by pipeline.config.
        # :param user_threshold: User's given threshold to overwrite default NMS score threshold. 
        # :param nms_name: Name of NMS node in a graph, renames NMS elements accordingly in order to eliminate cycles.

        if nms_name is None:
            nms_name = ""
        else:
            nms_name = "_" + nms_name

        # Set score threshold.
        score_threshold = nms_score_threshold if user_threshold is None else user_threshold

        # NMS Outputs.
        nms_output_num_detections = gs.Variable(name="num_detections"+nms_name, dtype=np.int32, shape=[self.batch_size, 1])
        nms_output_boxes = gs.Variable(name="detection_boxes"+nms_name, dtype=np.float32,
                                       shape=[self.batch_size, self.first_stage_max_proposals, 4])
        nms_output_scores = gs.Variable(name="detection_scores"+nms_name, dtype=np.float32,
                                        shape=[self.batch_size, self.first_stage_max_proposals])
        nms_output_classes = gs.Variable(name="detection_classes"+nms_name, dtype=np.int32,
                                         shape=[self.batch_size, self.first_stage_max_proposals])

        nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

        # Plugin.
        self.graph.plugin(
            op="EfficientNMS_TRT",
            name="nms/non_maximum_suppression"+nms_name,
            inputs=[box_net_tensor, class_net_tensor, anchors_tensor],
            outputs=nms_outputs,
            attrs={
                'plugin_version': "1",
                'background_class': background_class,
                'max_output_boxes': self.first_stage_max_proposals,
                'score_threshold': max(0.01, score_threshold),
                'iou_threshold': iou_threshold,
                'score_activation': score_activation,
                'box_coding': 1,
            } 
        )
        log.info("Created 'nms/non_maximum_suppression{}' NMS plugin".format(nms_name))

        return nms_outputs

    def CropAndResize(self, unsqeeze_input, relu_node_outputs, cnr_num):
        # Helper function to create the NMS Plugin node with the selected inputs. 
        # CropAndResize TensorRT Plugin is suitable for our use case.
        # :param unsqeeze_input: NMS's bonding boxes output, clipped and normalized if this is first CropAndResize, this is a souce of rois for CropAndResize. 
        # :param relu_node_outputs: 1st backbone's last Relu node, this is a souce of feature_maps for CropAndResize
        # :param cnr_num: Positional number of CropAndResize node in a graph, renames CropAndResize elements accordingly in order to eliminate cycles. 
        
        # CropAndResizePlugin requires 4th dimension of 1: [N, B, 4, 1], so
        # we need to add unsqeeze node to make tensor 4 dimensional. 
        unsqueeze_node = self.graph.unsqueeze("CNR/detection_boxes_unsqueeze_"+cnr_num, unsqeeze_input)

        # CropAndResizePlugin's inputs 
        feature_maps = relu_node_outputs
        rois = unsqueeze_node[0]

        # CropAndResize Outputs.
        cnr_pfmap = gs.Variable(name="cnr/pfmap_"+cnr_num, dtype=np.float32,
                                shape=[self.batch_size, self.first_stage_max_proposals, feature_maps.shape[1], self.first_stage_crop_size, self.first_stage_crop_size])

        # Create the CropandResize Plugin node with the selected inputs. 
        # Two inputs are given to the CropAndResize TensorRT node:
        # - The feature_maps (from the relu_node_outputs): [batch_size, channel_num, height, width]
        # - The rois (clipped and normalized detection boxes resulting from NMS): [batch_size, featuremap, 4, 1]
        self.graph.plugin(
            op="CropAndResize",
            name="cnr/crop_and_resize_"+cnr_num,
            inputs=[feature_maps, rois],
            outputs=[cnr_pfmap],
            attrs={
                'crop_width': self.first_stage_crop_size,
                'crop_height': self.first_stage_crop_size,
            }
        )
        log.info("Created {} CropAndResize plugin".format(cnr_num))

        # Reshape node that is preparing CropAndResize's pfmap output shape for MaxPool node that comes next.
        reshape_shape = np.asarray([self.first_stage_max_proposals*self.batch_size, feature_maps.shape[1], self.first_stage_crop_size, self.first_stage_crop_size], dtype=np.int64)
        reshape_node = self.graph.op_with_const("Reshape", "cnr/reshape_"+cnr_num, cnr_pfmap, reshape_shape)

        return reshape_node[0]
            
    def process_graph(self, first_nms_threshold=None, second_nms_threshold=None):
        """
        Processes the graph to replace the NMS operations by EfficientNMS_TRT TensorRT plugin nodes and
        cropAndResize operations by CropAndResize plugin node.
        :param first_nms_threshold: Override the 1st NMS score threshold value. If set to None, use the value in the graph.
        :param second_nms_threshold: Override the 2nd NMS score threshold value. If set to None, use the value in the graph.
        """
        def first_nms(background_class, score_activation, first_nms_threshold, nms_name=None):
            """
            Updates the graph to replace the 1st NMS op by EfficientNMS_TRT TensorRT plugin node.
            :param background_class: Set EfficientNMS_TRT's background_class atribute.
            :param score_activation: Set EfficientNMS_TRT's score_activation atribute.
            :param first_nms_threshold: Override the NMS score threshold.
            :param nms_name: Set the NMS node name.
            """
            # Supported models
            ssd_models = ['ssd_mobilenet_v1_fpn_keras', 'ssd_mobilenet_v2_fpn_keras', 'ssd_resnet50_v1_fpn_keras', 'ssd_resnet101_v1_fpn_keras', 'ssd_resnet152_v1_fpn_keras']
            frcnn_models = ['faster_rcnn_resnet50_keras', 'faster_rcnn_resnet101_keras', 'faster_rcnn_resnet152_keras', 'faster_rcnn_inception_resnet_v2_keras']
            
            # Getting SSD's Class and Box Nets final tensors.
            if "ssd" in self.model:
                # Find the concat node at the end of the class net (multi-scale class predictor).
                class_net_head_name = 'BoxPredictor/ConvolutionalClassHead_' if self.model == 'ssd_mobilenet_v2_keras' else 'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead'
                class_net = self.find_head_end(class_net_head_name, "Transpose", "Concat")
                # Final Class Net tensor
                class_net_tensor = self.graph.slice(class_net_head_name+"/slicer", class_net.outputs[0], 1, 91, 2)[0]  # Remove background class

                # Find the concat or squeeze node at the end of the box net (multi-scale localization predictor).
                if self.model == 'ssd_mobilenet_v2_keras':
                    box_net_head_name = 'BoxPredictor/ConvolutionalBoxHead_'
                    box_net = self.find_head_end(box_net_head_name, "Transpose", "Squeeze")
                else:
                    box_net_head_name = 'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead'
                    box_net = self.find_head_end(box_net_head_name, "Transpose", "Concat")

                box_net_output = box_net.outputs[0]
                # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale box_net_output in order to get accurate coordinates.
                variance_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
                # Final Box Net tensor.
                box_net_tensor = self.graph.op_with_const("Mul", box_net_head_name+"/scale", box_net_output, variance_adj)[0]
    
            # Getting Faster R-CNN's 1st Class and Box Nets tensors.
            elif "faster_rcnn" in self.model:
                # Identify Class Net and Box Net head names
                head_names = ['FirstStageBoxPredictor/ConvolutionalClassHead_0/ClassPredictor','FirstStageBoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor']

                # Find the softmax node at the end of the class net (multi-scale class predictor).
                class_net = self.find_head_end(head_names[0], "Transpose", "Softmax")
                # Final Class Net tensor
                class_net_tensor = class_net.outputs[0] 

                # Find the reshape node at the end of the box net (multi-scale localization predictor).
                box_net = self.find_head_end(head_names[1], "Transpose", "Reshape")
                # Final Box Net tensor.
                box_net_output = box_net.outputs[0]

                #Insert a squeeze node
                squeeze_node = self.graph.squeeze(head_names[1]+"/squeeze", box_net_output)
                # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale box_net_output, in order to get accurate coordinates.
                variance_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
                # Final Box Net tensor.
                box_net_tensor = self.graph.op_with_const("Mul", head_names[1]+"/scale", squeeze_node, variance_adj)[0]

            # Find the split node that separates the box net coordinates and feeds them into the box decoder.
            box_net_split = self.graph.find_descendant_by_op(box_net, "Split")
            assert box_net_split and len(box_net_split.outputs) == 4

            # Get anchors tensor.
            anchors_tensor = self.extract_anchors_tensor(box_net_split)

            # Create NMS node.
            nms_outputs = self.NMS(box_net_tensor, class_net_tensor, anchors_tensor, background_class, score_activation, self.first_stage_nms_iou_threshold, self.first_stage_nms_score_threshold, first_nms_threshold, nms_name)

            # Return NMS's outputs.
            return nms_outputs

        def first_cnr(input):
            """
            Updates the graph to replace the 1st cropAndResize op by CropAndResize TensorRT plugin node.
            :param input: Input tensor is the output from previous first_nms() step. 
            """

            # Locate the last Relu node of the first backbone (pre 1st NMS). Relu node contains feature maps
            # necessary for CropAndResize plugin.
            relu_name = "StatefulPartitionedCall/model/"
            relu_node = [node for node in self.graph.nodes if node.op == "Relu" and relu_name in node.name][-1]

            # Before passing 1st NMS's detection boxes (rois) to CropAndResize, we need to clip and normalize them.
            # Clipping happens for coordinates that are less than 0 and more than self.height.
            # Normalization is just divison of every coordinate by self.height.
            clip_out = self.graph.clip("FirstNMS/detection_boxes_clipper", input, 0, self.height)
            div_const = np.expand_dims(np.asarray([self.height, self.width, self.height, self.width], dtype=np.float32), axis=(0, 1))
            div_out = self.graph.op_with_const("Div", "FirstNMS/detection_boxes_normalizer", clip_out[0], div_const)

            # Linear transformation to convert box coordinates from (TopLeft, BottomRight) Corner encoding
            # to CenterSize encoding. 1st NMS boxes are multiplied by transformation matrix in order to 
            # encode it into CenterSize format.
            matmul_const = np.matrix('0.5 0 -1 0; 0 0.5 0 -1; 0.5 0 1 0; 0 0.5 0 1', dtype=np.float32)
            matmul_out = self.graph.matmul("FirstNMS/detection_boxes_conversion", div_out[0], matmul_const)

            # Create Crop and Resize node.
            cnr_output = self.CropAndResize(div_out, relu_node.outputs[0], "first")

            # Find MaxPool node that summarizes CropAndResize structure.
            maxpool_node = [node for node in self.graph.nodes if node.op == "MaxPool" and "MaxPool2D/MaxPool" in node.name][0]
            maxpool_node.inputs[0] = cnr_output

            # Return linear transformation node, it will be located between 1st and 2nd NMS, 
            # so we need to pass and connect it to 2nd NMS.
            # In case you are converting Mask R-CNN, feature maps are required for 2nd CropAndResize.
            return matmul_out[0], relu_node.outputs[0]

        def second_nms(background_class, score_activation, encoded_boxes, second_nms_threshold, nms_name=None):
            """
            Updates the graph to replace the 2nd (or final) NMS op by EfficientNMS_TRT TensorRT plugin node.
            :param background_class: Set EfficientNMS_TRT's background_class atribute. 
            :param score_activation: Set EfficientNMS_TRT's score_activation atribute. 
            :param encoded_boxes: The boxes to use as input. 
            :param second_nms_threshold: Override the NMS score threshold.
            :param nms_name: Set the NMS node name.
            """

            # Identify Class Net and Box Net head names.
            second_head_names = ['StatefulPartitionedCall/mask_rcnn_keras_box_predictor/mask_rcnn_class_head/ClassPredictor_dense',
                'StatefulPartitionedCall/mask_rcnn_keras_box_predictor/mask_rcnn_box_head/BoxEncodingPredictor_dense']

            # Find the softmax node at the end of the 2nd class net (multi-scale class predictor).
            second_class_net = self.find_head_end(second_head_names[0], "MatMul", "Softmax")

            # Faster R-CNN's slice operation to adjust third dimension of Class Net's last node tensor (adjusting class values).
            slice_out = self.graph.slice(second_head_names[0]+"/slicer", second_class_net.outputs[0], 1, 91, 2)

            # Final Class Net tensor.
            second_class_net_tensor = slice_out[0]
        
            # Find the add node at the end of the box net (multi-scale localization predictor).
            second_box_net = self.find_head_end(second_head_names[1], "MatMul", "Add")
            # Final Box Net tensor.
            second_box_net_output = second_box_net.outputs[0]

            # Reshape node that is preparing second_box_net_output's output shape for Mul scaling node that comes next.
            # Based on type of Crop and Resize operation, second_box_net_output can be of two types, example:
            # If use_matmul_crop_and_resize in pipeline.config is set to True, expect: [batch_size, first_stage_max_proposals, 4].
            # Else use_matmul_crop_and_resize is either False or absent, expect: [batch_size, first_stage_max_proposals, num_classes, 4]
            if self.matmul_crop_and_resize: 
                reshape_shape_second = np.asarray([self.batch_size, self.first_stage_max_proposals, second_box_net.outputs[0].shape[1]], dtype=np.int64)
            else:
                reshape_shape_second = np.asarray([self.batch_size, self.first_stage_max_proposals, self.num_classes, second_box_net.outputs[0].shape[1]/self.num_classes], dtype=np.int64)
            reshape_node_second = self.graph.op_with_const("Reshape", second_head_names[1]+"/reshape", second_box_net_output, reshape_shape_second)
            # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale second_box_net_output, in order to get accurate coordinates.
            second_scale_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
            second_scale_out = self.graph.op_with_const("Mul", second_head_names[1]+"/scale_second", reshape_node_second[0], second_scale_adj)

            # Final Box Net tensor.
            second_box_net_tensor = second_scale_out[0]

            # Create NMS node.
            nms_outputs = self.NMS(second_box_net_tensor, second_class_net_tensor, encoded_boxes, background_class, score_activation, self.second_stage_iou_threshold, self.second_stage_nms_score_threshold, second_nms_threshold, nms_name)
            
            return nms_outputs

        def second_cnr(feature_maps, second_nms_outputs):
            """
            Updates the graph to replace the 2nd cropAndResize op by CropAndResize TensorRT plugin node.
            :param input: Input tensor is the output from previous first_nms() step. 
            """

            # Before passing 2nd NMS's detection boxes (rois) to second CropAndResize, we need to clip them.
            # Clipping happens for coordinates that are less than 0 and more than 1 (binary).
            clip_out = self.graph.clip("SecondNMS/detection_boxes_clipper", second_nms_outputs[1], 0, 1)

            # Create Crop and Resize node.
            cnr_output = self.CropAndResize(clip_out, feature_maps, "second")

            # Find MaxPool node that summarizes CropAndResize structure 
            maxpool_node = [node for node in self.graph.nodes if node.op == "MaxPool" and "MaxPool2D/MaxPool_1" in node.name][0]
            maxpool_node.inputs[0] = cnr_output

            # Reshape node that is preparing 2nd NMS class outputs for Add node that comes next.
            # [self.batch_size, self.first_stage_max_proposals] -> [self.first_stage_max_proposals*self.batch_size]
            class_reshape_shape = np.asarray([self.first_stage_max_proposals*self.batch_size], dtype=np.int64)
            class_reshape_node = self.graph.op_with_const("Reshape", "Reshape_Class", second_nms_outputs[3], class_reshape_shape)

            # Find sigmoid node in the end of the network, applies sigmoid to get instance segmentation masks
            last_sigmoid_node = self.graph.find_descendant_by_op(maxpool_node, "Sigmoid", 40)

            if (self.num_classes > 1):
                # Find first ancestor of Sigmoid of operation type Add. This Add node is one of the Gather node inputs,
                # Gather node performs gather on 0th axis of data tensor and requires indices that set tesnors to be withing bounds,
                # this Add node provides the bounds for Gather.
                add_node = self.graph.find_ancestor_by_op(last_sigmoid_node, "Add")
                add_node.inputs[1] = class_reshape_node[0]

            # Final Reshape node, reshapes output of Sigmoid, important for various batch_size support.
            final_reshape_shape = np.asarray([self.batch_size, self.first_stage_max_proposals, self.mask_height, self.mask_width], dtype=np.int64)
            final_reshape_node = self.graph.op_with_const("Reshape", "Reshape_Final_Masks", last_sigmoid_node.outputs[0], final_reshape_shape)
            final_reshape_node[0].dtype = np.float32
            final_reshape_node[0].name = "detection_masks"

            return final_reshape_node[0]
        
        # If you model is SSD, you need only one NMS and nothin else.
        if "ssd" in self.model:
            # Set graph outputs.
            self.graph.outputs = first_nms(-1, True, first_nms_threshold)
            self.sanitize()
        # If your model is Faster R-CNN, you will need 2 NMS nodes with CropAndResize in between.
        elif "faster_rcnn" in self.model and self.mask_height is None and self.mask_width is None:
            first_nms_outputs = first_nms(0, False, first_nms_threshold, "rpn")
            first_cnr_output, feature_maps = first_cnr(first_nms_outputs[1])
            # Set graph outputs.
            self.graph.outputs = second_nms(-1, False, first_cnr_output, second_nms_threshold)
            self.sanitize()
        # Mask R-CNN
        elif "faster_rcnn" in self.model and not (self.mask_height is None and self.mask_width is None):
            first_nms_outputs = first_nms(0, False, first_nms_threshold, "rpn")
            first_cnr_output, feature_maps = first_cnr(first_nms_outputs[1])
            second_nms_outputs = second_nms(-1, False, first_cnr_output, second_nms_threshold)
            second_cnr_output = second_cnr(feature_maps, second_nms_outputs)
            # Append segmentation head output.
            second_nms_outputs.append(second_cnr_output)
            # Set graph outputs, both bbox and segmentation heads.
            self.graph.outputs = second_nms_outputs
            self.sanitize()


def main(args):
    effdet_gs = TFODGraphSurgeon(args.saved_model, args.pipeline_config)
    if args.tf2onnx:
        effdet_gs.save(args.tf2onnx)
    effdet_gs.update_preprocessor(args.batch_size, args.input_format)
    effdet_gs.process_graph(args.first_nms_threshold, args.second_nms_threshold)
    if args.debug:
        effdet_gs.add_debug_output(args.debug)
    effdet_gs.save(args.onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline_config", help="Pipeline configuration file to load", type=str)
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model directory to load", type=str)
    parser.add_argument("-o", "--onnx", help="The output ONNX model file to write", type=str)
    parser.add_argument("-b", "--batch_size", help="Batch size for the model", type=int, default=1)
    parser.add_argument("-t1", "--first_nms_threshold", help="Override the score threshold for the 1st NMS operation", type=float)
    parser.add_argument("-t2", "--second_nms_threshold", help="Override the score threshold for the 2nd NMS operation", type=float)
    parser.add_argument("-d", "--debug", action='append', help="Add an extra output to debug a particular node")
    parser.add_argument("-f", "--input_format", default="NHWC", choices=["NHWC", "NCHW"], 
                        help="Set the input shape of the graph, as comma-separated dimensions in NCHW or NHWC format, default: NHWC")
    parser.add_argument("--tf2onnx", help="The path where to save the intermediate ONNX graph generated by tf2onnx, "
                                          "useful for debugging purposes, default: not saved", type=str)
    args = parser.parse_args()
    if not all([args.pipeline_config, args.saved_model, args.onnx]):
        parser.print_help()
        print("\nThese arguments are required: --pipeline_config, --saved_model and --onnx")
        sys.exit(1)
    main(args)

