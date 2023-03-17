#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cv2
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnx import shape_inference
import torch

try:
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.modeling import build_model
    from detectron2.config import get_cfg
    from detectron2.structures import ImageList
except ImportError:
    print("Could not import Detectron 2 modules. Maybe you did not install Detectron 2")
    print("Please install Detectron 2, check https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md")
    sys.exit(1)

import onnx_utils

logging.basicConfig(level=logging.INFO)
logging.getLogger("ModelHelper").setLevel(logging.INFO)
log = logging.getLogger("ModelHelper")


class DET2GraphSurgeon:
    def __init__(self, saved_model_path, config_file, weights):
        """
        Constructor of the Model Graph Surgeon object, to do the conversion of a Detectron 2 Mask R-CNN exported model
        to an ONNX-TensorRT parsable model.
        :param saved_model_path: The path pointing to the exported Detectron 2 Mask R-CNN ONNX model.
        :param config_file: The path pointing to the Detectron 2 yaml file which describes the model.
        :param config_file: Weights to load for the Detectron 2 model.
        """

        def det2_setup(config_file, weights):
            """
            Create configs and perform basic setups.
            """
            cfg = get_cfg()
            cfg.merge_from_file(config_file)
            cfg.merge_from_list(["MODEL.WEIGHTS", weights])
            cfg.freeze()
            return cfg

        # Import exported Detectron 2 Mask R-CNN ONNX model as GraphSurgeon object.
        self.graph = gs.import_onnx(onnx.load(saved_model_path))
        assert self.graph
        log.info("ONNX graph loaded successfully")

        # Fold constants via ONNX-GS that exported script might've missed.
        self.graph.fold_constants()

        # Set up Detectron 2 model configuration.
        self.det2_cfg = det2_setup(config_file, weights)

        # Getting model characteristics.
        self.fpn_out_channels = self.det2_cfg.MODEL.FPN.OUT_CHANNELS
        self.num_classes = self.det2_cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.first_NMS_max_proposals = self.det2_cfg.MODEL.RPN.POST_NMS_TOPK_TEST
        self.first_NMS_iou_threshold = self.det2_cfg.MODEL.RPN.NMS_THRESH
        self.first_NMS_score_threshold = 0.01
        self.first_ROIAlign_pooled_size = self.det2_cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.first_ROIAlign_sampling_ratio = self.det2_cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.first_ROIAlign_type = self.det2_cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.second_NMS_max_proposals = self.det2_cfg.TEST.DETECTIONS_PER_IMAGE
        self.second_NMS_iou_threshold = self.det2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.second_NMS_score_threshold = self.det2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.second_ROIAlign_pooled_size = self.det2_cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.second_ROIAlign_sampling_ratio = self.det2_cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        self.second_ROIAlign_type = self.det2_cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        self.mask_out_res = 28

        # Model characteristics.
        log.info("Number of FPN output channels is {}".format(self.fpn_out_channels))
        log.info("Number of classes is {}".format(self.num_classes))
        log.info("First NMS max proposals is {}".format(self.first_NMS_max_proposals))
        log.info("First NMS iou threshold is {}".format(self.first_NMS_iou_threshold))
        log.info("First NMS score threshold is {}".format(self.first_NMS_score_threshold))
        log.info("First ROIAlign type is {}".format(self.first_ROIAlign_type))
        log.info("First ROIAlign pooled size is {}".format(self.first_ROIAlign_pooled_size))
        log.info("First ROIAlign sampling ratio is {}".format(self.first_ROIAlign_sampling_ratio))
        log.info("Second NMS max proposals is {}".format(self.second_NMS_max_proposals))
        log.info("Second NMS iou threshold is {}".format(self.second_NMS_iou_threshold))
        log.info("Second NMS score threshold is {}".format(self.second_NMS_score_threshold))
        log.info("Second ROIAlign type is {}".format(self.second_ROIAlign_type))
        log.info("Second ROIAlign pooled size is {}".format(self.second_ROIAlign_pooled_size))
        log.info("Second ROIAlign sampling ratio is {}".format(self.second_ROIAlign_sampling_ratio))
        log.info("Individual mask output resolution is {}x{}".format(self.mask_out_res, self.mask_out_res))

        self.batch_size = None

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
                log.error("This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your "
                          "onnx_graphsurgeon module. Error:\n{}".format(e))
                raise

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break

    def get_anchors(self, sample_image):
        """
        Detectron 2 exported ONNX does not contain anchors required for efficientNMS plug-in, so they must be generated
        "offline" by calling actual Detectron 2 model and getting anchors from it.
        :param sample_image: Sample image required to run through the model and obtain anchors.
        Can be any image from a dataset. Make sure listed here Detectron 2 preprocessing steps
        actually match your preprocessing steps. Otherwise, behavior can be unpredictable.
        Additionally, anchors have to be generated for a fixed input dimensions,
        meaning as soon as image leaves a preprocessor and enters predictor.model.backbone() it must have
        a fixed dimension (1344x1344 in my case) that every single image in dataset must follow, since currently
        TensorRT plug-ins do not support dynamic shapes.
        """
        # Get Detectron 2 model config and build it.
        predictor = DefaultPredictor(self.det2_cfg)
        model = build_model(self.det2_cfg)

        # Image preprocessing.
        input_im = cv2.imread(sample_image)
        raw_height, raw_width = input_im.shape[:2]
        image = predictor.aug.get_transform(input_im).apply_image(input_im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # Model preprocessing.
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = [x["image"].to(model.device) for x in inputs]
        images = [(x - model.pixel_mean) / model.pixel_std for x in images]
        imagelist_images = ImageList.from_tensors(images, 1344)

        # Get feature maps from backbone.
        features = predictor.model.backbone(imagelist_images.tensor)

        # Get proposals from Region Proposal Network and obtain anchors from anchor generator.
        features = [features[f] for f in predictor.model.proposal_generator.in_features]
        det2_anchors = predictor.model.proposal_generator.anchor_generator(features)

        # Extract anchors based on feature maps in ascending order (P2->P6).
        p2_anchors = det2_anchors[0].tensor.detach().cpu().numpy()
        p3_anchors = det2_anchors[1].tensor.detach().cpu().numpy()
        p4_anchors = det2_anchors[2].tensor.detach().cpu().numpy()
        p5_anchors = det2_anchors[3].tensor.detach().cpu().numpy()
        p6_anchors = det2_anchors[4].tensor.detach().cpu().numpy()
        final_anchors = np.concatenate((p2_anchors,p3_anchors,p4_anchors,p5_anchors,p6_anchors))

        return final_anchors

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

    def update_preprocessor(self, batch_size):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.
        :param batch_size: The batch size to use for the ONNX graph.
        """
        # Set graph inputs.
        self.batch_size = batch_size
        self.height = self.graph.inputs[0].shape[1]
        self.width = self.graph.inputs[0].shape[2]

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
        input_tensor = self.graph.inputs[0]

        # Create preprocessing Sub node and connect input tensor to it.
        sub_const = np.expand_dims(np.asarray([255 * 0.406, 255 * 0.456, 255 * 0.485], dtype=np.float32), axis=(1, 2))
        sub_out = self.graph.op_with_const("Sub", "preprocessor/mean", input_tensor, sub_const)

        # Find first Div node and connect to output of Sub node.
        div_node = self.graph.find_node_by_op("Div")
        log.info("Found {} node".format(div_node.op))
        div_node.inputs[0] = sub_out[0]

        # Find first Conv and connect preprocessor directly to it.
        conv_node = self.graph.find_node_by_op("Conv")
        log.info("Found {} node".format(conv_node.op))
        conv_node.inputs[0] = div_node.outputs[0]

        # Reshape nodes tend to update the batch dimension to a fixed value of 1, they should use the batch size instead.
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) == gs.Constant and node.inputs[1].values[0] == 1:
                node.inputs[1].values[0] = self.batch_size

    def NMS(self, boxes, scores, anchors, background_class, score_activation, max_proposals, iou_threshold, nms_score_threshold, user_threshold, nms_name=None):
        # Helper function to create the NMS Plugin node with the selected inputs.
        # EfficientNMS_TRT TensorRT Plugin is suitable for our use case.
        # :param boxes: The box predictions from the Box Net.
        # :param scores: The class predictions from the Class Net.
        # :param anchors: The default anchor coordinates.
        # :param background_class: The label ID for the background class.
        # :param max_proposals: Number of proposals made by NMS.
        # :param score_activation: If set to True - apply sigmoid activation to the confidence scores during NMS operation,
        # if false - no activation.
        # :param iou_threshold: NMS intersection over union threshold, given by self.det2_cfg.
        # :param nms_score_threshold: NMS score threshold, given by self.det2_cfg.
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
                                       shape=[self.batch_size, max_proposals, 4])
        nms_output_scores = gs.Variable(name="detection_scores"+nms_name, dtype=np.float32,
                                        shape=[self.batch_size, max_proposals])
        nms_output_classes = gs.Variable(name="detection_classes"+nms_name, dtype=np.int32,
                                         shape=[self.batch_size, max_proposals])

        nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

        # Plugin.
        self.graph.plugin(
            op="EfficientNMS_TRT",
            name="nms"+nms_name,
            inputs=[boxes, scores, anchors],
            outputs=nms_outputs,
            attrs={
                'plugin_version': "1",
                'background_class': background_class,
                'max_output_boxes': max_proposals,
                'score_threshold': max(0.01, score_threshold),
                'iou_threshold': iou_threshold,
                'score_activation': score_activation,
                'class_agnostic': False,
                'box_coding': 1,
            }
        )
        log.info("Created nms{} with EfficientNMS_TRT plugin".format(nms_name))

        return nms_outputs

    def ROIAlign(self, rois, p2, p3, p4, p5, pooled_size, sampling_ratio, roi_align_type, num_rois, ra_name):
        # Helper function to create the ROIAlign Plugin node with the selected inputs.
        # PyramidROIAlign_TRT TensorRT Plugin is suitable for our use case.
        # :param rois: Regions of interest/detection boxes outputs from preceding NMS node.
        # :param p2: Output of p2 feature map.
        # :param p3: Output of p3 feature map.
        # :param p4: Output of p4 feature map.
        # :param p5: Output of p5 feature map.
        # :param pooled_size: Pooled output dimensions.
        # :param sampling_ratio: Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.
        # :param roi_align_type: Type of Detectron 2 ROIAlign op, either ROIAlign (vanilla) or ROIAlignV2 (0.5 coordinate offset).
        # :param num_rois: Number of ROIs resulting from ROIAlign operation.
        # :param ra_name: Name of ROIAlign node in a graph, renames ROIAlign elements accordingly in order to eliminate cycles.

        # Different types of Detectron 2's ROIAlign ops require coordinate offset that is supported by PyramidROIAlign_TRT.
        if roi_align_type == "ROIAlignV2":
            roi_coords_transform = 2
        elif roi_align_type == "ROIAlign":
            roi_coords_transform = 0

        # ROIAlign outputs.
        roi_align_output = gs.Variable(name="roi_align/output_"+ra_name, dtype=np.float32,
                                shape=[self.batch_size, num_rois, self.fpn_out_channels, pooled_size, pooled_size])

        # Plugin.
        self.graph.plugin(
            op="PyramidROIAlign_TRT",
            name="roi_align_"+ra_name,
            inputs=[rois, p2, p3, p4, p5],
            outputs=[roi_align_output],
            attrs={
                'plugin_version': "1",
                'fpn_scale': 224,
                'pooled_size': pooled_size,
                'image_size': [self.height, self.width],
                'roi_coords_absolute': 0,
                'roi_coords_swap': 0,
                'roi_coords_transform': roi_coords_transform,
                'sampling_ratio': sampling_ratio,
            }
        )
        log.info("Created {} with PyramidROIAlign_TRT plugin".format(ra_name))

        return roi_align_output

    def process_graph(self, anchors, first_nms_threshold=None, second_nms_threshold=None):
        """
        Processes the graph to replace the GenerateProposals and BoxWithNMSLimit operations with EfficientNMS_TRT
        TensorRT plugin nodes and ROIAlign operations with PyramidROIAlign_TRT plugin nodes.
        :param anchors: Anchors generated from sample image "offline" by Detectron 2, since anchors are not provided
        inside the graph.
        :param first_nms_threshold: Override the 1st NMS score threshold value. If set to None, use the value in the graph.
        :param second_nms_threshold: Override the 2nd NMS score threshold value. If set to None, use the value in the graph.
        """
        def backbone():
            """
            Updates the graph to replace all ResizeNearest ops with ResizeNearest plugins in backbone.
            """
            # Get final backbone outputs.
            p2 = self.graph.find_node_by_op_name("Conv", "/backbone/fpn_output2/Conv")
            p3 = self.graph.find_node_by_op_name("Conv", "/backbone/fpn_output3/Conv")
            p4 = self.graph.find_node_by_op_name("Conv", "/backbone/fpn_output4/Conv")
            p5 = self.graph.find_node_by_op_name("Conv", "/backbone/fpn_output5/Conv")


            return p2.outputs[0], p3.outputs[0], p4.outputs[0], p5.outputs[0]

        def proposal_generator(anchors, first_nms_threshold):
            """
            Updates the graph to replace all GenerateProposals Caffe ops with one single NMS for proposals generation.
            :param anchors: Anchors generated from sample image "offline" by Detectron 2, since anchors are not provided
            inside the graph
            :param first_nms_threshold: Override the 1st NMS score threshold value. If set to None, use the value in the graph.
            """
            # Get nodes containing final objectness logits.
            p2_logits = self.graph.find_node_by_op_name("Flatten", "/proposal_generator/Flatten")
            p3_logits = self.graph.find_node_by_op_name("Flatten", "/proposal_generator/Flatten_1")
            p4_logits = self.graph.find_node_by_op_name("Flatten", "/proposal_generator/Flatten_2")
            p5_logits = self.graph.find_node_by_op_name("Flatten", "/proposal_generator/Flatten_3")
            p6_logits = self.graph.find_node_by_op_name("Flatten", "/proposal_generator/Flatten_4")

            # Get nodes containing final anchor_deltas.
            p2_anchors = self.graph.find_node_by_op_name("Reshape", "/proposal_generator/Reshape_1")
            p3_anchors = self.graph.find_node_by_op_name("Reshape", "/proposal_generator/Reshape_3")
            p4_anchors = self.graph.find_node_by_op_name("Reshape", "/proposal_generator/Reshape_5")
            p5_anchors = self.graph.find_node_by_op_name("Reshape", "/proposal_generator/Reshape_7")
            p6_anchors = self.graph.find_node_by_op_name("Reshape", "/proposal_generator/Reshape_9")

            # Concatenate all objectness logits/scores data.
            scores_inputs = [p2_logits.outputs[0], p3_logits.outputs[0], p4_logits.outputs[0], p5_logits.outputs[0], p6_logits.outputs[0]]
            scores_tensor = self.graph.layer(name="scores", op="Concat", inputs=scores_inputs, outputs=['scores'], attrs={'axis': 1})[0]
            # Unsqueeze to add 3rd dimension of 1 to match tensor dimensions of boxes tensor.
            scores = self.graph.unsqueeze("scores_unsqueeze", scores_tensor, [2])[0]

            # Concatenate all boxes/anchor_delta data.
            boxes_inputs = [p2_anchors.outputs[0], p3_anchors.outputs[0], p4_anchors.outputs[0], p5_anchors.outputs[0], p6_anchors.outputs[0]]
            boxes = self.graph.layer(name="boxes", op="Concat", inputs=boxes_inputs, outputs=['anchors'], attrs={'axis': 1})[0]

            # Convert the anchors from Corners to CenterSize encoding.
            anchors = np.matmul(anchors, [[0.5, 0, -1, 0], [0, 0.5, 0, -1], [0.5, 0, 1, 0], [0, 0.5, 0, 1]])
            anchors = anchors / [self.width, self.height, self.width, self.height] # Normalize anchors to [0-1] range
            anchors = np.expand_dims(anchors, axis=0)
            anchors = anchors.astype(np.float32)
            anchors = gs.Constant(name="default_anchors", values=anchors)

            # Create NMS node.
            nms_outputs = self.NMS(boxes, scores, anchors, -1, False, self.first_NMS_max_proposals, self.first_NMS_iou_threshold, self.first_NMS_score_threshold, first_nms_threshold, 'rpn')

            return nms_outputs

        def roi_heads(rpn_outputs, p2, p3, p4, p5, second_nms_threshold):
            """
            Updates the graph to replace all ROIAlign Caffe ops with one single pyramid ROIAlign. Eliminates CollectRpnProposals
            DistributeFpnProposals and BatchPermutation nodes that are not supported by TensorRT. Connects pyramid ROIAlign to box_head
            and connects box_head to final box head outputs in a form of second NMS. In order to implement mask head outputs,
            similar steps as in box_pooler are performed to replace mask_pooler. Finally, reimplemented mask_pooler is connected to
            mask_head and mask head outputs are produced.
            :param rpn_outputs: Outputs of the first NMS/proposal generator.
            :param p2: Output of p2 feature map, required for ROIAlign operation.
            :param p3: Output of p3 feature map, required for ROIAlign operation.
            :param p4: Output of p4 feature map, required for ROIAlign operation.
            :param p5: Output of p5 feature map, required for ROIAlign operation.
            :param second_nms_threshold: Override the 2nd NMS score threshold value. If set to None, use the value in the graph.
            """
            # Create ROIAlign node.
            box_pooler_output = self.ROIAlign(rpn_outputs[1], p2, p3, p4, p5, self.first_ROIAlign_pooled_size, self.first_ROIAlign_sampling_ratio, self.first_ROIAlign_type, self.first_NMS_max_proposals, 'box_pooler')

            # Reshape node that prepares ROIAlign/box pooler output for Gemm node that comes next.
            box_pooler_shape = np.asarray([-1, self.fpn_out_channels*self.first_ROIAlign_pooled_size*self.first_ROIAlign_pooled_size], dtype=np.int64)
            box_pooler_reshape = self.graph.op_with_const("Reshape", "box_pooler/reshape", box_pooler_output, box_pooler_shape)

            # Get first Gemm op of box head and connect box pooler to it.
            first_box_head_gemm = self.graph.find_node_by_op_name("Gemm", "/roi_heads/box_head/fc1/Gemm")
            first_box_head_gemm.inputs[0] = box_pooler_reshape[0]

            # Get final two nodes of box predictor. Softmax op for cls_score, Gemm op for bbox_pred.
            cls_score = self.graph.find_node_by_op_name("Softmax", "/roi_heads/Softmax")
            bbox_pred = self.graph.find_node_by_op_name("Gemm", "/roi_heads/box_predictor/bbox_pred/Gemm")

            # Linear transformation to convert box coordinates from (TopLeft, BottomRight) Corner encoding
            # to CenterSize encoding. 1st NMS boxes are multiplied by transformation matrix in order to
            # encode it into CenterSize format.
            matmul_const = np.matrix('0.5 0 -1 0; 0 0.5 0 -1; 0.5 0 1 0; 0 0.5 0 1', dtype=np.float32)
            matmul_out = self.graph.matmul("RPN_NMS/detection_boxes_conversion", rpn_outputs[1], matmul_const)

            # Reshape node that prepares bbox_pred for scaling and second NMS.
            bbox_pred_shape = np.asarray([self.batch_size, self.first_NMS_max_proposals, self.num_classes, 4], dtype=np.int64)
            bbox_pred_reshape = self.graph.op_with_const("Reshape", "bbox_pred/reshape", bbox_pred.outputs[0], bbox_pred_shape)

            # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale bbox_pred_reshape, in order to get accurate coordinates.
            scale_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
            final_bbox_pred = self.graph.op_with_const("Mul", "bbox_pred/scale", bbox_pred_reshape[0], scale_adj)

            # Reshape node that prepares cls_score for slicing and second NMS.
            cls_score_shape = np.array([self.batch_size, self.first_NMS_max_proposals, self.num_classes+1], dtype=np.int64)
            cls_score_reshape = self.graph.op_with_const("Reshape", "cls_score/reshape", cls_score.outputs[0], cls_score_shape)

            # Slice operation to adjust third dimension of cls_score tensor, deletion of background class (81 in Detectron 2).
            final_cls_score = self.graph.slice("cls_score/slicer", cls_score_reshape[0], 0, self.num_classes, 2)

            # Create NMS node.
            nms_outputs = self.NMS(final_bbox_pred[0], final_cls_score[0], matmul_out[0], -1, False, self.second_NMS_max_proposals, self.second_NMS_iou_threshold, self.second_NMS_score_threshold, second_nms_threshold, 'box_outputs')

            # Create ROIAlign node.
            mask_pooler_output = self.ROIAlign(nms_outputs[1], p2, p3, p4, p5, self.second_ROIAlign_pooled_size, self.second_ROIAlign_sampling_ratio, self.second_ROIAlign_type, self.second_NMS_max_proposals, 'mask_pooler')

            # Reshape mask pooler output.
            mask_pooler_shape = np.asarray([self.second_NMS_max_proposals*self.batch_size, self.fpn_out_channels, self.second_ROIAlign_pooled_size, self.second_ROIAlign_pooled_size], dtype=np.int64)
            mask_pooler_reshape_node = self.graph.op_with_const("Reshape", "mask_pooler/reshape", mask_pooler_output, mask_pooler_shape)

            # Get first Conv op in mask head and connect ROIAlign's squeezed output to it.
            mask_head_conv = self.graph.find_node_by_op_name("Conv", "/roi_heads/mask_head/mask_fcn1/Conv")
            mask_head_conv.inputs[0] = mask_pooler_reshape_node[0]

            # Reshape node that is preparing 2nd NMS class outputs for Add node that comes next.
            classes_reshape_shape = np.asarray([self.second_NMS_max_proposals*self.batch_size], dtype=np.int64)
            classes_reshape_node = self.graph.op_with_const("Reshape", "box_outputs/reshape_classes", nms_outputs[3], classes_reshape_shape)

            # This loop will generate an array used in Add node, which eventually will help Gather node to pick the single
            # class of interest per bounding box, instead of creating 80 masks for every single bounding box.
            add_array = []
            for i in range(self.second_NMS_max_proposals*self.batch_size):
                if i == 0:
                    start_pos = 0
                else:
                    start_pos = i * self.num_classes
                add_array.append(start_pos)

            # This Add node is one of the Gather node inputs, Gather node performs gather on 0th axis of data tensor
            # and requires indices that set tensors to be withing bounds, this Add node provides the bounds for Gather.
            add_array = np.asarray(add_array, dtype=np.int32)
            classes_add_node = self.graph.op_with_const("Add", "box_outputs/add", classes_reshape_node[0], add_array)

            # Get the last Conv op in mask head and reshape it to correctly gather class of interest's masks.
            last_conv = self.graph.find_node_by_op_name("Conv", "/roi_heads/mask_head/predictor/Conv")
            last_conv_reshape_shape = np.asarray([self.second_NMS_max_proposals*self.num_classes*self.batch_size, self.mask_out_res, self.mask_out_res], dtype=np.int64)
            last_conv_reshape_node = self.graph.op_with_const("Reshape", "mask_head/reshape_all_masks", last_conv.outputs[0], last_conv_reshape_shape)

            # Gather node that selects only masks belonging to detected class, 79 other masks are discarded.
            final_gather = self.graph.gather("mask_head/final_gather", last_conv_reshape_node[0], classes_add_node[0], 0)

            # Get last Sigmoid node and connect Gather node to it.
            mask_head_sigmoid = self.graph.find_node_by_op_name("Sigmoid", "/roi_heads/mask_head/Sigmoid")
            mask_head_sigmoid.inputs[0] = final_gather[0]

            # Final Reshape node, reshapes output of Sigmoid, important for various batch_size support (not tested yet).
            final_graph_reshape_shape = np.asarray([self.batch_size, self.second_NMS_max_proposals, self.mask_out_res, self.mask_out_res], dtype=np.int64)
            final_graph_reshape_node = self.graph.op_with_const("Reshape", "mask_head/final_reshape", mask_head_sigmoid.outputs[0], final_graph_reshape_shape)
            final_graph_reshape_node[0].dtype = np.float32
            final_graph_reshape_node[0].name = "detection_masks"

            return nms_outputs, final_graph_reshape_node[0]

        # Only Detectron 2's Mask-RCNN R50-FPN 3x is supported currently.
        p2, p3, p4, p5 = backbone()
        rpn_outputs = proposal_generator(anchors, first_nms_threshold)
        box_head_outputs, mask_head_output = roi_heads(rpn_outputs, p2, p3, p4, p5, second_nms_threshold)
        # Append segmentation head output.
        box_head_outputs.append(mask_head_output)
        # Set graph outputs, both bbox and segmentation heads.
        self.graph.outputs = box_head_outputs
        self.sanitize()


def main(args):
    det2_gs = DET2GraphSurgeon(args.exported_onnx, args.det2_config, args.det2_weights)
    det2_gs.update_preprocessor(args.batch_size)
    anchors = det2_gs.get_anchors(args.sample_image)
    det2_gs.process_graph(anchors, args.first_nms_threshold, args.second_nms_threshold)
    det2_gs.save(args.onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--exported_onnx", help="The exported to ONNX Detectron 2 Mask R-CNN", type=str)
    parser.add_argument("-o", "--onnx", help="The output ONNX model file to write", type=str)
    parser.add_argument("-c", "--det2_config", help="The Detectron 2 config file (.yaml) for the model", type=str)
    parser.add_argument("-w", "--det2_weights", help="The Detectron 2 model weights (.pkl)", type=str)
    parser.add_argument("-s", "--sample_image", help="Sample image for anchors generation", type=str)
    parser.add_argument("-b", "--batch_size", help="Batch size for the model", type=int, default=1)
    parser.add_argument("-t1", "--first_nms_threshold", help="Override the score threshold for the 1st NMS operation", type=float)
    parser.add_argument("-t2", "--second_nms_threshold", help="Override the score threshold for the 2nd NMS operation", type=float)
    args = parser.parse_args()
    if not all([args.exported_onnx, args.onnx, args.det2_config, args.det2_weights, args.sample_image]):
        parser.print_help()
        print("\nThese arguments are required: --exported_onnx --onnx --det2_config --det2_weights and --sample_image")
        sys.exit(1)
    main(args)
