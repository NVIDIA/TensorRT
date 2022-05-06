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
import sys
import argparse
import logging

import tensorflow as tf
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnx import shape_inference
from tf2onnx import tfonnx, optimizer, tf_loader

import onnx_utils

logging.basicConfig(level=logging.INFO)
logging.getLogger("EfficientDetGraphSurgeon").setLevel(logging.INFO)
log = logging.getLogger("EfficientDetGraphSurgeon")


class EfficientDetGraphSurgeon:
    def __init__(self, saved_model_path):
        """
        Constructor of the EfficientDet Graph Surgeon object, to do the conversion of an EfficientDet TF saved model
        to an ONNX-TensorRT parsable model.
        :param saved_model_path: The path pointing to the TensorFlow saved model to load.
        """
        saved_model_path = os.path.realpath(saved_model_path)
        assert os.path.exists(saved_model_path)

        # Use tf2onnx to convert saved model to an initial ONNX graph.
        graph_def, inputs, outputs = tf_loader.from_saved_model(
            saved_model_path, None, None, "serve", ["serving_default"]
        )
        log.info("Loaded saved model from {}".format(saved_model_path))
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")
        with tf_loader.tf_session(graph=tf_graph):
            onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
        onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
        self.graph = gs.import_onnx(onnx_model)
        assert self.graph
        log.info("TF2ONNX graph created successfully")

        # Fold constants via ONNX-GS that TF2ONNX may have missed
        self.graph.fold_constants()

        # Try to auto-detect by finding if nodes match a specific name pattern expected for either of the APIs.
        self.api = None
        if len([node for node in self.graph.nodes if "class_net/" in node.name]) > 0:
            self.api = "AutoML"
        elif len([node for node in self.graph.nodes if "/WeightSharedConvolutionalClassHead/" in node.name]) > 0:
            self.api = "TFOD"
        assert self.api
        log.info("Graph was detected as {}".format(self.api))

    def sanitize(self):
        """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort, and fold constant inputs values.
        When possible, run shape inference on the ONNX graph to determine tensor shapes.
        """
        for i in range(3):
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
                log.error(
                    "This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your "
                    "onnx_graphsurgeon module. Error:\n{}".format(e)
                )
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

    def update_preprocessor(self, input_format, input_size, preprocessor="imagenet"):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.
        :param input_format: The input data format, either "NCHW" or "NHWC".
        :param input_size: The input size as a comma-separated string in H,W format, e.g. "512,512".
        :param preprocessor: The preprocessor to use, either "imagenet" for imagenet mean and stdev normalization,
        or "scale_range" for uniform [-1,+1] range normalization.
        """
        # Update the input and output tensors shape
        input_size = input_size.split(",")
        assert len(input_size) == 2
        for i in range(len(input_size)):
            input_size[i] = int(input_size[i])
            assert input_size[i] >= 1
        assert input_format in ["NCHW", "NHWC"]
        if input_format == "NCHW":
            self.graph.inputs[0].shape = ['N', 3, input_size[0], input_size[1]]
        if input_format == "NHWC":
            self.graph.inputs[0].shape = ['N', input_size[0], input_size[1], 3]
        self.graph.inputs[0].dtype = np.float32
        self.graph.inputs[0].name = "input"
        log.info("ONNX graph input shape: {} [{} format]".format(self.graph.inputs[0].shape, input_format))
        self.sanitize()

        # Find the initial nodes of the graph, whatever the input is first connected to, and disconnect them
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Convert to NCHW format if needed
        input_tensor = self.graph.inputs[0]
        if input_format == "NHWC":
            input_tensor = self.graph.transpose("preprocessor/transpose", input_tensor, [0, 3, 1, 2])

        assert preprocessor in ["imagenet", "scale_range"]
        preprocessed_tensor = None
        if preprocessor == "imagenet":
            # RGB Normalizers. The per-channel values are given with shape [1, 3, 1, 1] for proper NCHW shape broadcasting
            scale_val = 1 / np.asarray([255], dtype=np.float32)
            mean_val = -1 * np.expand_dims(np.asarray([0.485, 0.456, 0.406], dtype=np.float32), axis=(0, 2, 3))
            stddev_val = 1 / np.expand_dims(np.asarray([0.229, 0.224, 0.225], dtype=np.float32), axis=(0, 2, 3))
            # y = (x * scale + mean) * stddev   -->   y = x * scale * stddev + mean * stddev
            scale_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, scale_val * stddev_val)
            mean_out = self.graph.elt_const("Add", "preprocessor/mean", scale_out, mean_val * stddev_val)
            preprocessed_tensor = mean_out[0]
        if preprocessor == "scale_range":
            # RGB Normalizers. The per-channel values are given with shape [1, 3, 1, 1] for proper NCHW shape broadcasting
            scale_val = 2 / np.asarray([255], dtype=np.float32)
            offset_val = np.expand_dims(np.asarray([-1, -1, -1], dtype=np.float32), axis=(0, 2, 3))
            # y = (x * scale + mean) * stddev   -->   y = x * scale * stddev + mean * stddev
            scale_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, scale_val)
            range_out = self.graph.elt_const("Add", "preprocessor/range", scale_out, offset_val)
            preprocessed_tensor = range_out[0]

        # Find the first stem conv node of the graph, and connect the normalizer directly to it
        stem_name = None
        if self.api == "AutoML":
            stem_name = "/stem/"
        if self.api == "TFOD":
            stem_name = "/stem_conv2d/"
        stem = [node for node in self.graph.nodes if node.op == "Conv" and stem_name in node.name][0]
        log.info("Found {} node '{}' as stem entry".format(stem.op, stem.name))
        stem.inputs[0] = preprocessed_tensor

        self.sanitize()

    def update_shapes(self):
        # Reshape nodes have the batch dimension as a fixed value of 1, they should use the batch size instead
        # Output-Head reshapes use [1, -1, C], corrected reshape value should be [-1, V, C]
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            shape_in = node.inputs[0].shape
            if shape_in is None or len(shape_in) not in [4,5]: # TFOD graphs have 5-dim inputs on this Reshape
                continue
            if type(node.inputs[1]) != gs.Constant:
                continue
            shape_out = node.inputs[1].values
            if len(shape_out) != 3 or shape_out[0] != 1 or shape_out[1] != -1:
                continue
            volume = shape_in[1] * shape_in[2] * shape_in[3] / shape_out[2]
            if len(shape_in) == 5:
                volume *= shape_in[4]
            shape_corrected = np.asarray([-1, volume, shape_out[2]], dtype=np.int64)
            node.inputs[1] = gs.Constant("{}_shape".format(node.name), values=shape_corrected)
            log.info("Updating Output-Head Reshape node {} to {}".format(node.name, node.inputs[1].values))

        # Other Reshapes only need to change the first dim to -1, as long as there are no -1's already
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) != gs.Constant or node.inputs[1].values[0] != 1 or -1 in node.inputs[1].values:
                continue
            node.inputs[1].values[0] = -1
            log.info("Updating Reshape node {} to {}".format(node.name, node.inputs[1].values))

        # Resize nodes try to calculate the output shape dynamically, it's more optimal to pre-compute the shape
        if self.api == "AutoML":
            # Resize on a BiFPN will always be 2x, but grab it from the graph just in case
            for node in [node for node in self.graph.nodes if node.op == "Resize"]:
                if len(node.inputs) < 4 or node.inputs[0].shape is None:
                    continue
                scale_h, scale_w = None, None
                if type(node.inputs[3]) == gs.Constant:
                    # The sizes input is already folded
                    if len(node.inputs[3].values) != 4:
                        continue
                    scale_h = node.inputs[3].values[2] / node.inputs[0].shape[2]
                    scale_w = node.inputs[3].values[3] / node.inputs[0].shape[3]
                if type(node.inputs[3]) == gs.Variable:
                    # The sizes input comes from Shape+Slice+Concat
                    concat = node.i(3)
                    if concat.op != "Concat":
                        continue
                    if type(concat.inputs[1]) != gs.Constant or len(concat.inputs[1].values) != 2:
                        continue
                    scale_h = concat.inputs[1].values[0] / node.inputs[0].shape[2]
                    scale_w = concat.inputs[1].values[1] / node.inputs[0].shape[3]
                scales = np.asarray([1, 1, scale_h, scale_w], dtype=np.float32)
                del node.inputs[3]
                node.inputs[2] = gs.Constant(name="{}_scales".format(node.name), values=scales)
                log.info("Updating Resize node {} to {}".format(node.name, scales))

        self.sanitize()

    def update_network(self):
        """
        Updates the graph to replace certain nodes in the main EfficientDet network:
        - the global average pooling nodes are optimized when running for TFOD models.
        """

        if self.api == "TFOD":
            for reduce in [node for node in self.graph.nodes if node.op == "ReduceMean"]:
                # TFOD models have their ReduceMean nodes applied with some redundant transposes that can be
                # optimized away for better performance
                # Make sure the correct subgraph is being replaced, basically search for this:
                # X > Transpose (0,2,3,1) > ReduceMean (1,2) > Reshape (?,1,1,?) > Reshape (?,?,1,1) > Conv > Y
                # And change to this:
                # X > ReduceMean (2,3) > Conv > Y
                transpose = reduce.i()
                if transpose.op != "Transpose" or transpose.attrs['perm'] != [0, 2, 3, 1]:
                    continue
                if len(reduce.attrs['axes']) != 2 or reduce.attrs['axes'] != [1, 2]:
                    continue
                reshape1 = reduce.o()
                if reshape1.op != "Reshape" or len(reshape1.inputs[1].values) != 4:
                    continue
                if reshape1.inputs[1].values[1] != 1 or reshape1.inputs[1].values[2] != 1:
                    continue
                reshape2 = reshape1.o()
                if reshape2.op != "Reshape" or len(reshape2.inputs[1].values) != 4:
                    continue
                if reshape2.inputs[1].values[2] != 1 or reshape2.inputs[1].values[3] != 1:
                    continue
                conv = reshape2.o()
                if conv.op != "Conv":
                    continue
                # If all the checks above pass, then this node sequence can be optimized by just the ReduceMean itself
                # operating on a different set of axes
                input_tensor = transpose.inputs[0]  # Input tensor to the Transpose
                reduce.inputs[0] = input_tensor  # Forward the Transpose input to the ReduceMean node
                output_tensor = reduce.outputs[0]  # Output tensor of the ReduceMean
                conv.inputs[0] = output_tensor  # Forward the ReduceMean output to the Conv node
                reduce.attrs["axes"] = [2, 3]  # Update the axes that ReduceMean operates on
                reduce.attrs["keepdims"] = 1  # Keep the reduced dimensions
                log.info("Optimized subgraph around ReduceMean node '{}'".format(reduce.name))

    def update_nms(self, threshold=None, detections=None):
        """
        Updates the graph to replace the NMS op by BatchedNMS_TRT TensorRT plugin node.
        :param threshold: Override the score threshold attribute. If set to None, use the value in the graph.
        :param detections: Override the max detections attribute. If set to None, use the value in the graph.
        """

        def find_head_concat(name_scope):
            # This will find the concatenation node at the end of either Class Net or Box Net. These concatenation nodes
            # bring together prediction data for each of 5 scales.
            # The concatenated Class Net node will have shape [batch_size, num_anchors, num_classes],
            # and the concatenated Box Net node has the shape [batch_size, num_anchors, 4].
            # These concatenation nodes can be be found by searching for all Concat's and checking if the node two
            # steps above in the graph has a name that begins with either "box_net/..." or "class_net/...".
            for node in [node for node in self.graph.nodes if node.op == "Transpose" and name_scope in node.name]:
                concat = self.graph.find_descendant_by_op(node, "Concat")
                assert concat and len(concat.inputs) == 5
                log.info("Found {} node '{}' as the tip of {}".format(concat.op, concat.name, name_scope))
                return concat

        def extract_anchors_tensor(split):
            # This will find the anchors that have been hardcoded somewhere within the ONNX graph.
            # The function will return a gs.Constant that can be directly used as an input to the NMS plugin.
            # The anchor tensor shape will be [1, num_anchors, 4]. Note that '1' is kept as first dim, regardless of
            # batch size, as it's not necessary to replicate the anchors for all images in the batch.

            # The anchors are available (one per coordinate) hardcoded as constants within certain box decoder nodes.
            # Each of these four constants have shape [1, num_anchors], so some numpy operations are used to expand the
            # dims and concatenate them as needed.

            # These constants can be found by starting from the Box Net's split operation , and for each coordinate,
            # walking down in the graph until either an Add or Mul node is found. The second input on this nodes will
            # be the anchor data required.
            def get_anchor_np(output_idx, op):
                node = self.graph.find_descendant_by_op(split.o(0, output_idx), op)
                assert node
                val = np.squeeze(node.inputs[1].values)
                return np.expand_dims(val.flatten(), axis=(0, 2))

            anchors_y = get_anchor_np(0, "Add")
            anchors_x = get_anchor_np(1, "Add")
            anchors_h = get_anchor_np(2, "Mul")
            anchors_w = get_anchor_np(3, "Mul")
            anchors = np.concatenate([anchors_y, anchors_x, anchors_h, anchors_w], axis=2)
            return gs.Constant(name="nms/anchors:0", values=anchors)

        self.sanitize()

        head_names = []
        if self.api == "AutoML":
            head_names = ["class_net/", "box_net/"]
        if self.api == "TFOD":
            head_names = ["/WeightSharedConvolutionalClassHead/", "/WeightSharedConvolutionalBoxHead/"]

        # There are five nodes at the bottom of the graph that provide important connection points:

        # 1. Find the concat node at the end of the class net (multi-scale class predictor)
        class_net = find_head_concat(head_names[0])
        class_net_tensor = class_net.outputs[0]

        # 2. Find the concat node at the end of the box net (multi-scale localization predictor)
        box_net = find_head_concat(head_names[1])
        box_net_tensor = box_net.outputs[0]

        # 3. Find the split node that separates the box net coordinates and feeds them into the box decoder.
        box_net_split = self.graph.find_descendant_by_op(box_net, "Split")
        assert box_net_split and len(box_net_split.outputs) == 4

        # 4. Find the concat node at the end of the box decoder.
        box_decoder = self.graph.find_descendant_by_op(box_net_split, "Concat")
        assert box_decoder and len(box_decoder.inputs) == 4
        box_decoder_tensor = box_decoder.outputs[0]

        # 5. Find the NMS node.
        nms_node = self.graph.find_node_by_op("NonMaxSuppression")

        # Extract NMS Configuration
        num_detections = int(nms_node.inputs[2].values) if detections is None else detections
        iou_threshold = float(nms_node.inputs[3].values)
        score_threshold = float(nms_node.inputs[4].values) if threshold is None else threshold
        num_classes = class_net.i().inputs[1].values[-1]
        normalized = True if self.api == "TFOD" else False

        # NMS Inputs and Attributes
        # NMS expects these shapes for its input tensors:
        # box_net: [batch_size, number_boxes, 4]
        # class_net: [batch_size, number_boxes, number_classes]
        # anchors: [1, number_boxes, 4] (if used)
        nms_op = None
        nms_attrs = None
        nms_inputs = None

        # EfficientNMS TensorRT Plugin
        # Fusing the decoder will always be faster, so this is the default NMS method supported. In this case,
        # three inputs are given to the NMS TensorRT node:
        # - The box predictions (from the Box Net node found above)
        # - The class predictions (from the Class Net node found above)
        # - The default anchor coordinates (from the extracted anchor constants)
        # As the original tensors from EfficientDet will be used, the NMS code type is set to 1 (Center+Size),
        # because this is the internal box coding format used by the network.
        anchors_tensor = extract_anchors_tensor(box_net_split)
        nms_inputs = [box_net_tensor, class_net_tensor, anchors_tensor]
        nms_op = "EfficientNMS_TRT"
        nms_attrs = {
            'plugin_version': "1",
            'background_class': -1,
            'max_output_boxes': num_detections,
            'score_threshold': max(0.01, score_threshold),  # Keep threshold to at least 0.01 for better efficiency
            'iou_threshold': iou_threshold,
            'score_activation': True,
            'box_coding': 1,
        }
        nms_output_classes_dtype = np.int32

        # NMS Outputs
        nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=['N', 1])
        nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32,
                                       shape=['N', num_detections, 4])
        nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32,
                                        shape=['N', num_detections])
        nms_output_classes = gs.Variable(name="detection_classes", dtype=nms_output_classes_dtype,
                                         shape=['N', num_detections])

        nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

        # Create the NMS Plugin node with the selected inputs. The outputs of the node will also become the final
        # outputs of the graph.
        self.graph.plugin(
            op=nms_op,
            name="nms/non_maximum_suppression",
            inputs=nms_inputs,
            outputs=nms_outputs,
            attrs=nms_attrs)
        log.info("Created NMS plugin '{}' with attributes: {}".format(nms_op, nms_attrs))

        self.graph.outputs = nms_outputs

        self.sanitize()


def main(args):
    effdet_gs = EfficientDetGraphSurgeon(args.saved_model)
    if args.tf2onnx:
        effdet_gs.save(args.tf2onnx)
    effdet_gs.update_preprocessor(args.input_format, args.input_size, args.preprocessor)
    effdet_gs.update_shapes()
    effdet_gs.update_network()
    effdet_gs.update_nms(args.nms_threshold, args.nms_detections)
    effdet_gs.save(args.onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--saved_model", required=True,
                        help="The TensorFlow saved model directory to load")
    parser.add_argument("-o", "--onnx", required=True,
                        help="The output ONNX model file to write")
    parser.add_argument("-f", "--input_format", default="NHWC", choices=["NHWC", "NCHW"],
                        help="Set the input data format of the graph, either NCHW or NHWC, default: NHWC")
    parser.add_argument("-i", "--input_size", default="512,512",
                        help="Set the input shape of the graph, as a comma-separated dimensions in H,W format, "
                             "default: 512,512")
    parser.add_argument("-p", "--preprocessor", default="imagenet", choices=["imagenet", "scale_range"],
                        help="Set the preprocessor to apply on the graph, either 'imagenet' for standard mean "
                             "subtraction and stdev normalization, or 'scale_range' for uniform [-1,+1] "
                             "normalization as is used in the AdvProp models, default: imagenet")
    parser.add_argument("-t", "--nms_threshold", type=float,
                        help="Override the NMS score threshold, default: use the original value in the model")
    parser.add_argument("-d", "--nms_detections", type=int,
                        help="Override the NMS max detections, default: use the original value in the model")
    parser.add_argument("--tf2onnx",
                        help="The path where to save the intermediate ONNX graph generated by tf2onnx, useful"
                             "for graph debugging purposes, default: not saved")
    args = parser.parse_args()
    main(args)
