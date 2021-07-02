#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
    def __init__(self, saved_model_path, legacy_plugins=False):
        """
        Constructor of the EfficientDet Graph Surgeon object, to do the conversion of an EfficientDet TF saved model
        to an ONNX-TensorRT parsable model.
        :param saved_model_path: The path pointing to the TensorFlow saved model to load.
        :param legacy_plugins: If using TensorRT version < 8.0.1, set this to True to use older (but slower) plugins.
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

        self.batch_size = None
        self.legacy_plugins = legacy_plugins

    def infer(self):
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

    def update_preprocessor(self, input_shape):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.
        :param input_shape: The input tensor shape to use for the ONNX graph.
        """
        # Update the input and output tensors shape
        input_shape = input_shape.split(",")
        assert len(input_shape) == 4
        for i in range(len(input_shape)):
            input_shape[i] = int(input_shape[i])
            assert input_shape[i] >= 1
        input_format = None
        if input_shape[1] == 3:
            input_format = "NCHW"
        if input_shape[3] == 3:
            input_format = "NHWC"
        assert input_format in ["NCHW", "NHWC"]
        self.batch_size = input_shape[0]
        self.graph.inputs[0].shape = input_shape
        self.graph.inputs[0].dtype = np.float32
        if self.api == "TFOD" and self.batch_size > 1 and self.legacy_plugins:
            log.error("TFOD models with a batch size larger than 1 are not currently supported in legacy plugin mode. "
                      "Please upgrade to TensorRT >= 8.0.1 or use batch size 1 for now.")
            sys.exit(1)
        self.infer()
        log.info("ONNX graph input shape: {} [{} format detected]".format(self.graph.inputs[0].shape, input_format))

        # Find the initial nodes of the graph, whatever the input is first connected to, and disconnect them
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Convert to NCHW format if needed
        input_tensor = self.graph.inputs[0]
        if input_format == "NHWC":
            input_tensor = self.graph.transpose("preprocessor/transpose", input_tensor, [0, 3, 1, 2])

        # RGB Normalizers. The per-channel values are given with shape [1, 3, 1, 1] for proper NCHW shape broadcasting
        scale_val = 1 / np.asarray([255], dtype=np.float32)
        mean_val = -1 * np.expand_dims(np.asarray([0.485, 0.456, 0.406], dtype=np.float32), axis=(0, 2, 3))
        stddev_val = 1 / np.expand_dims(np.asarray([0.229, 0.224, 0.225], dtype=np.float32), axis=(0, 2, 3))
        # y = (x * scale + mean) * stddev   -->   y = x * scale * stddev + mean * stddev
        scale_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, scale_val * stddev_val)
        mean_out = self.graph.elt_const("Add", "preprocessor/mean", scale_out, mean_val * stddev_val)

        # Find the first stem conv node of the graph, and connect the normalizer directly to it
        stem_name = None
        if self.api == "AutoML":
            stem_name = "/stem/"
        if self.api == "TFOD":
            stem_name = "/stem_conv2d/"
        stem = [node for node in self.graph.nodes if node.op == "Conv" and stem_name in node.name][0]
        log.info("Found {} node '{}' as stem entry".format(stem.op, stem.name))
        stem.inputs[0] = mean_out[0]

        # Reshape nodes tend to update the batch dimension to a fixed value of 1, they should use the batch size instead
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) == gs.Constant and node.inputs[1].values[0] == 1:
                node.inputs[1].values[0] = self.batch_size

        self.infer()

    def update_network(self):
        """
        Updates the graph to replace certain nodes in the main EfficientDet network:
        - the global average pooling nodes are optimized when running for TFOD models.
        - the nearest neighbor resize ops in the FPN are replaced by a TRT plugin nodes when running in legacy mode.
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
                reduce.attrs['axes'] = [2, 3]  # Update the axes that ReduceMean operates on
                reduce.attrs['keepdims'] = 1  # Keep the reduced dimensions
                log.info("Optimized subgraph around ReduceMean node '{}'".format(reduce.name))

        if self.legacy_plugins:
            self.infer()
            count = 1
            for node in [node for node in self.graph.nodes if node.op == "Resize" and node.attrs['mode'] == "nearest"]:
                # Older versions of TensorRT do not understand nearest neighbor resize ops, so a plugin is used to
                # perform this operation.
                self.graph.plugin(
                    op="ResizeNearest_TRT",
                    name="resize_nearest_{}".format(count),
                    inputs=[node.inputs[0]],
                    outputs=node.outputs,
                    attrs={
                        'plugin_version': "1",
                        'scale': 2.0,  # All resize ops in the EfficientDet FPN should have an upscale factor of 2.0
                    })
                node.outputs.clear()
                log.info(
                    "Replaced '{}' ({}) with a ResizeNearest_TRT plugin node".format(node.name, count))
                count += 1

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

        self.infer()

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
        if not self.legacy_plugins:
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
        else:
            # BatchedNMS TensorRT Plugin
            # Alternatively, the ONNX box decoder can be used. This will be slower, as more element-wise and non-fused
            # operations will need to be performed by TensorRT. However, it's easier to implement, so it is shown here
            # for reference. In this case, only two inputs are given to the NMS TensorRT node:
            # - The box predictions (already decoded through the ONNX Box Decoder node)
            # - The class predictions (from the Class Net node found above, but also needs to pass through a sigmoid)
            # This time, the box predictions will have the coordinate coding from the ONNX box decoder, which matches
            # what the BatchedNMS plugin uses.

            if self.api == "AutoML":
                # The default boxes tensor has shape [batch_size, number_boxes, 4]. This will insert a "1" dimension
                # in the second axis, to become [batch_size, number_boxes, 1, 4], the shape that BatchedNMS expects.
                box_decoder_tensor = self.graph.unsqueeze("nms/box_net_reshape", box_decoder_tensor, axes=[2])[0]
            if self.api == "TFOD":
                # The default boxes tensor has shape [4, number_boxes]. This will transpose and insert a "1" dimension
                # in the 0 and 2 axes, to become [1, number_boxes, 1, 4], the shape that BatchedNMS expects.
                box_decoder_tensor = self.graph.transpose("nms/box_decoder_transpose", box_decoder_tensor, perm=[1, 0])
                box_decoder_tensor = self.graph.unsqueeze("nms/box_decoder_reshape", box_decoder_tensor, axes=[0, 2])[0]

            # BatchedNMS also expects the classes tensor to be already activated, in the case of EfficientDet, this is
            # through a Sigmoid op.
            class_net_tensor = self.graph.sigmoid("nms/class_net_sigmoid", class_net_tensor)[0]

            nms_inputs = [box_decoder_tensor, class_net_tensor]
            nms_op = "BatchedNMS_TRT"
            nms_attrs = {
                'plugin_version': "1",
                'shareLocation': True,
                'backgroundLabelId': -1,
                'numClasses': num_classes,
                'topK': 1024,
                'keepTopK': num_detections,
                'scoreThreshold': score_threshold,
                'iouThreshold': iou_threshold,
                'isNormalized': normalized,
                'clipBoxes': False,
                # 'scoreBits': 10, # Some versions of the plugin may need this parameter. If so, uncomment this line.
            }
            nms_output_classes_dtype = np.float32

        # NMS Outputs
        nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[self.batch_size, 1])
        nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32,
                                       shape=[self.batch_size, num_detections, 4])
        nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32,
                                        shape=[self.batch_size, num_detections])
        nms_output_classes = gs.Variable(name="detection_classes", dtype=nms_output_classes_dtype,
                                         shape=[self.batch_size, num_detections])

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

        self.infer()


def main(args):
    effdet_gs = EfficientDetGraphSurgeon(args.saved_model, args.legacy_plugins)
    if args.tf2onnx:
        effdet_gs.save(args.tf2onnx)
    effdet_gs.update_preprocessor(args.input_shape)
    effdet_gs.update_network()
    effdet_gs.update_nms(args.nms_threshold, args.nms_detections)
    effdet_gs.save(args.onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model directory to load")
    parser.add_argument("-o", "--onnx", help="The output ONNX model file to write")
    parser.add_argument("-i", "--input_shape", default="1,512,512,3",
                        help="Set the input shape of the graph, as comma-separated dimensions in NCHW or NHWC format, "
                             "default: 1,512,512,3")
    parser.add_argument("-t", "--nms_threshold", type=float, help="Override the score threshold for the NMS op, "
                                                                  "default: use the original value in the model")
    parser.add_argument("-d", "--nms_detections", type=int, help="Override the max detections for the NMS op, "
                                                                 "default: use the original value in the model")
    parser.add_argument("--legacy_plugins", action="store_true", help="Use legacy plugins for support on TensorRT "
                                                                      "versions lower than 8.0.1")
    parser.add_argument("--tf2onnx", help="The path where to save the intermediate ONNX graph generated by tf2onnx, "
                                          "useful for debugging purposes, default: not saved")
    args = parser.parse_args()
    if not all([args.saved_model, args.onnx]):
        parser.print_help()
        print("\nThese arguments are required: --saved_model and --onnx")
        sys.exit(1)
    main(args)
