#!/usr/bin/env python3
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

from __future__ import print_function
from collections import OrderedDict
import sys
import os

import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from downloader import getFilePath


class DarkNetParser(object):
    """Definition of a parser for DarkNet-based YOLOv3-608 (only tested for this topology)."""

    def __init__(self, supported_layers):
        """Initializes a DarkNetParser object.

        Keyword argument:
        supported_layers -- a string list of supported layers in DarkNet naming convention,
        parameters are only added to the class dictionary if a parsed layer is included.
        """

        # A list of YOLOv3 layers containing dictionaries with all layer
        # parameters:
        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers
        self.layer_counter = 0

    def parse_cfg_file(self, cfg_file_path):
        """Takes the yolov3.cfg file and parses it layer by layer,
        appending each layer's parameters as a dictionary to layer_configs.

        Keyword argument:
        cfg_file_path -- path to the yolov3.cfg file as string
        """
        with open(cfg_file_path) as cfg_file:
            remainder = cfg_file.read()
            while remainder is not None:
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        return self.layer_configs

    def _next_layer(self, remainder):
        """Takes in a string and segments it by looking for DarkNet delimiters.
        Returns the layer parameters and the remaining string after the last delimiter.
        Example for the first Conv layer in yolo.cfg ...

        [convolutional]
        batch_normalize=1
        filters=32
        size=3
        stride=1
        pad=1
        activation=leaky

        ... becomes the following layer_dict return value:
        {'activation': 'leaky', 'stride': 1, 'pad': 1, 'filters': 32,
        'batch_normalize': 1, 'type': 'convolutional', 'size': 3}.

        '001_convolutional' is returned as layer_name, and all lines that follow in yolo.cfg
        are returned as the next remainder.

        Keyword argument:
        remainder -- a string with all raw text after the previously parsed layer
        """
        remainder = remainder.split("[", 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            return None, None, None
        remainder = remainder.split("]", 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder
        else:
            return None, None, None
        if remainder.replace(" ", "")[0] == "#":
            remainder = remainder.split("\n", 1)[1]

        layer_param_block, remainder = remainder.split("\n\n", 1)
        layer_param_lines = layer_param_block.split("\n")[1:]
        layer_name = str(self.layer_counter).zfill(3) + "_" + layer_type
        layer_dict = dict(type=layer_type)
        if layer_type in self.supported_layers:
            for param_line in layer_param_lines:
                if param_line[0] == "#":
                    continue
                param_type, param_value = self._parse_params(param_line)
                layer_dict[param_type] = param_value
        self.layer_counter += 1
        return layer_dict, layer_name, remainder

    def _parse_params(self, param_line):
        """Identifies the parameters contained in one of the cfg file and returns
        them in the required format for each parameter type, e.g. as a list, an int or a float.

        Keyword argument:
        param_line -- one parsed line within a layer block
        """
        param_line = param_line.replace(" ", "")
        param_type, param_value_raw = param_line.split("=")
        param_value = None
        if param_type == "layers":
            layer_indexes = list()
            for index in param_value_raw.split(","):
                layer_indexes.append(int(index))
            param_value = layer_indexes
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = param_value_raw[0] == "-" and param_value_raw[1:].isdigit()
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            else:
                param_value = float(param_value_raw)
        else:
            param_value = str(param_value_raw)
        return param_type, param_value


class MajorNodeSpecs(object):
    """Helper class used to store the names of ONNX output names,
    corresponding to the output of a DarkNet layer and its output channels.
    Some DarkNet layers are not created and there is no corresponding ONNX node,
    but we still need to track them in order to set up skip connections.
    """

    def __init__(self, name, channels):
        """Initialize a MajorNodeSpecs object.

        Keyword arguments:
        name -- name of the ONNX node
        channels -- number of output channels of this node
        """
        self.name = name
        self.channels = channels
        self.created_onnx_node = False
        if name is not None and isinstance(channels, int) and channels > 0:
            self.created_onnx_node = True


class ConvParams(object):
    """Helper class to store the hyper parameters of a Conv layer,
    including its prefix name in the ONNX graph and the expected dimensions
    of weights for convolution, bias, and batch normalization.

    Additionally acts as a wrapper for generating safe names for all
    weights, checking on feasible combinations.
    """

    def __init__(self, node_name, batch_normalize, conv_weight_dims):
        """Constructor based on the base node name (e.g. 101_convolutional), the batch
        normalization setting, and the convolutional weights shape.

        Keyword arguments:
        node_name -- base name of this YOLO convolutional layer
        batch_normalize -- bool value if batch normalization is used
        conv_weight_dims -- the dimensions of this layer's convolutional weights
        """
        self.node_name = node_name
        self.batch_normalize = batch_normalize
        assert len(conv_weight_dims) == 4
        self.conv_weight_dims = conv_weight_dims

    def generate_param_name(self, param_category, suffix):
        """Generates a name based on two string inputs,
        and checks if the combination is valid."""
        assert suffix
        assert param_category in ["bn", "conv"]
        assert suffix in ["scale", "mean", "var", "weights", "bias"]
        if param_category == "bn":
            assert self.batch_normalize
            assert suffix in ["scale", "bias", "mean", "var"]
        elif param_category == "conv":
            assert suffix in ["weights", "bias"]
            if suffix == "bias":
                assert not self.batch_normalize
        param_name = self.node_name + "_" + param_category + "_" + suffix
        return param_name


class ResizeParams(object):
    # Helper class to store the scale parameter for an Resize node.

    def __init__(self, node_name, value):
        """Constructor based on the base node name (e.g. 86_Resize),
        and the value of the scale input tensor.

        Keyword arguments:
        node_name -- base name of this YOLO Resize layer
        value -- the value of the scale input to the Resize layer as numpy array
        """
        self.node_name = node_name
        self.value = value

    def generate_param_name(self):
        """Generates the scale parameter name for the Resize node."""
        param_name = self.node_name + "_" + "scale"
        return param_name

    def generate_roi_name(self):
        """Generates the roi input name for the Resize node."""
        param_name = self.node_name + "_" + "roi"
        return param_name


class WeightLoader(object):
    """Helper class used for loading the serialized weights of a binary file stream
    and returning the initializers and the input tensors required for populating
    the ONNX graph with weights.
    """

    def __init__(self, weights_file_path):
        """Initialized with a path to the YOLOv3 .weights file.

        Keyword argument:
        weights_file_path -- path to the weights file.
        """
        self.weights_file = self._open_weights_file(weights_file_path)

    def load_resize_scales(self, resize_params):
        """Returns the initializers with the value of the scale input
        tensor given by resize_params.

        Keyword argument:
        resize_params -- a ResizeParams object
        """
        initializer = list()
        inputs = list()
        name = resize_params.generate_param_name()
        shape = resize_params.value.shape
        data = resize_params.value
        scale_init = helper.make_tensor(name, TensorProto.FLOAT, shape, data)
        scale_input = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        initializer.append(scale_init)
        inputs.append(scale_input)

        # In opset 11 an additional input named roi is required. Create a dummy tensor to satisfy this.
        # It is a 1D tensor of size of the rank of the input (4)
        rank = 4
        roi_name = resize_params.generate_roi_name()
        roi_input = helper.make_tensor_value_info(roi_name, TensorProto.FLOAT, [rank])
        roi_init = helper.make_tensor(roi_name, TensorProto.FLOAT, [rank], [0, 0, 0, 0])
        initializer.append(roi_init)
        inputs.append(roi_input)

        return initializer, inputs

    def load_conv_weights(self, conv_params):
        """Returns the initializers with weights from the weights file and
        the input tensors of a convolutional layer for all corresponding ONNX nodes.

        Keyword argument:
        conv_params -- a ConvParams object
        """
        initializer = list()
        inputs = list()
        if conv_params.batch_normalize:
            bias_init, bias_input = self._create_param_tensors(conv_params, "bn", "bias")
            bn_scale_init, bn_scale_input = self._create_param_tensors(conv_params, "bn", "scale")
            bn_mean_init, bn_mean_input = self._create_param_tensors(conv_params, "bn", "mean")
            bn_var_init, bn_var_input = self._create_param_tensors(conv_params, "bn", "var")
            initializer.extend([bn_scale_init, bias_init, bn_mean_init, bn_var_init])
            inputs.extend([bn_scale_input, bias_input, bn_mean_input, bn_var_input])
        else:
            bias_init, bias_input = self._create_param_tensors(conv_params, "conv", "bias")
            initializer.append(bias_init)
            inputs.append(bias_input)
        conv_init, conv_input = self._create_param_tensors(conv_params, "conv", "weights")
        initializer.append(conv_init)
        inputs.append(conv_input)
        return initializer, inputs

    def _open_weights_file(self, weights_file_path):
        """Opens a YOLOv3 DarkNet file stream and skips the header.

        Keyword argument:
        weights_file_path -- path to the weights file.
        """
        weights_file = open(weights_file_path, "rb")
        length_header = 5
        np.ndarray(shape=(length_header,), dtype="int32", buffer=weights_file.read(length_header * 4))
        return weights_file

    def _create_param_tensors(self, conv_params, param_category, suffix):
        """Creates the initializers with weights from the weights file together with
        the input tensors.

        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
        """
        param_name, param_data, param_data_shape = self._load_one_param_type(conv_params, param_category, suffix)

        initializer_tensor = helper.make_tensor(param_name, TensorProto.FLOAT, param_data_shape, param_data)
        input_tensor = helper.make_tensor_value_info(param_name, TensorProto.FLOAT, param_data_shape)
        return initializer_tensor, input_tensor

    def _load_one_param_type(self, conv_params, param_category, suffix):
        """Deserializes the weights from a file stream in the DarkNet order.

        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
        """
        param_name = conv_params.generate_param_name(param_category, suffix)
        channels_out, channels_in, filter_h, filter_w = conv_params.conv_weight_dims
        if param_category == "bn":
            param_shape = [channels_out]
        elif param_category == "conv":
            if suffix == "weights":
                param_shape = [channels_out, channels_in, filter_h, filter_w]
            elif suffix == "bias":
                param_shape = [channels_out]
        param_size = np.product(np.array(param_shape))
        param_data = np.ndarray(shape=param_shape, dtype="float32", buffer=self.weights_file.read(param_size * 4))
        param_data = param_data.flatten().astype(float)
        return param_name, param_data, param_shape


class GraphBuilderONNX(object):
    """Class for creating an ONNX graph from a previously generated list of layer dictionaries."""

    def __init__(self, output_tensors):
        """Initialize with all DarkNet default parameters used creating YOLOv3,
        and specify the output tensors as an OrderedDict for their output dimensions
        with their names as keys.

        Keyword argument:
        output_tensors -- the output tensors as an OrderedDict containing the keys'
        output dimensions
        """
        self.output_tensors = output_tensors
        self._nodes = list()
        self.graph_def = None
        self.input_tensor = None
        self.epsilon_bn = 1e-5
        self.momentum_bn = 0.99
        self.alpha_lrelu = 0.1
        self.param_dict = OrderedDict()
        self.major_node_specs = list()
        self.batch_size = 1

    def build_onnx_graph(self, layer_configs, weights_file_path, verbose=True):
        """Iterate over all layer configs (parsed from the DarkNet representation
        of YOLOv3-608), create an ONNX graph, populate it with weights from the weights
        file and return the graph definition.

        Keyword arguments:
        layer_configs -- an OrderedDict object with all parsed layers' configurations
        weights_file_path -- location of the weights file
        verbose -- toggles if the graph is printed after creation (default: True)
        """
        for layer_name in layer_configs.keys():
            layer_dict = layer_configs[layer_name]
            major_node_specs = self._make_onnx_node(layer_name, layer_dict)
            if major_node_specs.name is not None:
                self.major_node_specs.append(major_node_specs)
        outputs = list()
        for tensor_name in self.output_tensors.keys():
            output_dims = [
                self.batch_size,
            ] + self.output_tensors[tensor_name]
            output_tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, output_dims)
            outputs.append(output_tensor)
        inputs = [self.input_tensor]
        weight_loader = WeightLoader(weights_file_path)
        initializer = list()
        # If a layer has parameters, add them to the initializer and input lists.
        for layer_name in self.param_dict.keys():
            _, layer_type = layer_name.split("_", 1)
            params = self.param_dict[layer_name]
            if layer_type == "convolutional":
                initializer_layer, inputs_layer = weight_loader.load_conv_weights(params)
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
            elif layer_type == "upsample":
                initializer_layer, inputs_layer = weight_loader.load_resize_scales(params)
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
        del weight_loader
        self.graph_def = helper.make_graph(
            nodes=self._nodes, name="YOLOv3-608", inputs=inputs, outputs=outputs, initializer=initializer
        )
        if verbose:
            print(helper.printable_graph(self.graph_def))
        model_def = helper.make_model(self.graph_def, producer_name="NVIDIA TensorRT sample")
        return model_def

    def _make_onnx_node(self, layer_name, layer_dict):
        """Take in a layer parameter dictionary, choose the correct function for
        creating an ONNX node and store the information important to graph creation
        as a MajorNodeSpec object.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        layer_type = layer_dict["type"]
        if self.input_tensor is None:
            if layer_type == "net":
                major_node_output_name, major_node_output_channels = self._make_input_tensor(layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name, major_node_output_channels)
            else:
                raise ValueError('The first node has to be of type "net".')
        else:
            node_creators = dict()
            node_creators["convolutional"] = self._make_conv_node
            node_creators["shortcut"] = self._make_shortcut_node
            node_creators["route"] = self._make_route_node
            node_creators["upsample"] = self._make_resize_node

            if layer_type in node_creators.keys():
                major_node_output_name, major_node_output_channels = node_creators[layer_type](layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name, major_node_output_channels)
            else:
                print("Layer of type %s not supported, skipping ONNX node generation." % layer_type)
                major_node_specs = MajorNodeSpecs(layer_name, None)
        return major_node_specs

    def _make_input_tensor(self, layer_name, layer_dict):
        """Create an ONNX input tensor from a 'net' layer and store the batch size.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        batch_size = layer_dict["batch"]
        channels = layer_dict["channels"]
        height = layer_dict["height"]
        width = layer_dict["width"]
        self.batch_size = batch_size
        input_tensor = helper.make_tensor_value_info(
            str(layer_name), TensorProto.FLOAT, [batch_size, channels, height, width]
        )
        self.input_tensor = input_tensor
        return layer_name, channels

    def _get_previous_node_specs(self, target_index=-1):
        """Get a previously generated ONNX node (skip those that were not generated).
        Target index can be passed for jumping to a specific index.

        Keyword arguments:
        target_index -- optional for jumping to a specific index (default: -1 for jumping
        to previous element)
        """
        previous_node = None
        for node in self.major_node_specs[target_index::-1]:
            if node.created_onnx_node:
                previous_node = node
                break
        assert previous_node is not None
        return previous_node

    def _make_conv_node(self, layer_name, layer_dict):
        """Create an ONNX Conv node with optional batch normalization and
        activation nodes.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        previous_node_specs = self._get_previous_node_specs()
        inputs = [previous_node_specs.name]
        previous_channels = previous_node_specs.channels
        kernel_size = layer_dict["size"]
        stride = layer_dict["stride"]
        filters = layer_dict["filters"]
        batch_normalize = False
        if "batch_normalize" in layer_dict.keys() and layer_dict["batch_normalize"] == 1:
            batch_normalize = True

        kernel_shape = [kernel_size, kernel_size]
        weights_shape = [filters, previous_channels] + kernel_shape
        conv_params = ConvParams(layer_name, batch_normalize, weights_shape)

        strides = [stride, stride]
        dilations = [1, 1]
        weights_name = conv_params.generate_param_name("conv", "weights")
        inputs.append(weights_name)
        if not batch_normalize:
            bias_name = conv_params.generate_param_name("conv", "bias")
            inputs.append(bias_name)

        conv_node = helper.make_node(
            "Conv",
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            auto_pad="SAME_LOWER",
            dilations=dilations,
            name=layer_name,
        )
        self._nodes.append(conv_node)
        inputs = [layer_name]
        layer_name_output = layer_name

        if batch_normalize:
            layer_name_bn = layer_name + "_bn"
            bn_param_suffixes = ["scale", "bias", "mean", "var"]
            for suffix in bn_param_suffixes:
                bn_param_name = conv_params.generate_param_name("bn", suffix)
                inputs.append(bn_param_name)
            batchnorm_node = helper.make_node(
                "BatchNormalization",
                inputs=inputs,
                outputs=[layer_name_bn],
                epsilon=self.epsilon_bn,
                momentum=self.momentum_bn,
                name=layer_name_bn,
            )
            self._nodes.append(batchnorm_node)
            inputs = [layer_name_bn]
            layer_name_output = layer_name_bn

        if layer_dict["activation"] == "leaky":
            layer_name_lrelu = layer_name + "_lrelu"

            lrelu_node = helper.make_node(
                "LeakyRelu", inputs=inputs, outputs=[layer_name_lrelu], name=layer_name_lrelu, alpha=self.alpha_lrelu
            )
            self._nodes.append(lrelu_node)
            inputs = [layer_name_lrelu]
            layer_name_output = layer_name_lrelu
        elif layer_dict["activation"] == "linear":
            pass
        else:
            print("Activation not supported.")

        self.param_dict[layer_name] = conv_params
        return layer_name_output, filters

    def _make_shortcut_node(self, layer_name, layer_dict):
        """Create an ONNX Add node with the shortcut properties from
        the DarkNet-based graph.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        shortcut_index = layer_dict["from"]
        activation = layer_dict["activation"]
        assert activation == "linear"

        first_node_specs = self._get_previous_node_specs()
        second_node_specs = self._get_previous_node_specs(target_index=shortcut_index)
        assert first_node_specs.channels == second_node_specs.channels
        channels = first_node_specs.channels
        inputs = [first_node_specs.name, second_node_specs.name]
        shortcut_node = helper.make_node(
            "Add",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(shortcut_node)
        return layer_name, channels

    def _make_route_node(self, layer_name, layer_dict):
        """If the 'layers' parameter from the DarkNet configuration is only one index, continue
        node creation at the indicated (negative) index. Otherwise, create an ONNX Concat node
        with the route properties from the DarkNet-based graph.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        route_node_indexes = layer_dict["layers"]
        if len(route_node_indexes) == 1:
            split_index = route_node_indexes[0]
            assert split_index < 0
            # Increment by one because we skipped the YOLO layer:
            split_index += 1
            self.major_node_specs = self.major_node_specs[:split_index]
            layer_name = None
            channels = None
        else:
            inputs = list()
            channels = 0
            for index in route_node_indexes:
                if index > 0:
                    # Increment by one because we count the input as a node (DarkNet
                    # does not)
                    index += 1
                route_node_specs = self._get_previous_node_specs(target_index=index)
                inputs.append(route_node_specs.name)
                channels += route_node_specs.channels
            assert inputs
            assert channels > 0

            route_node = helper.make_node(
                "Concat",
                axis=1,
                inputs=inputs,
                outputs=[layer_name],
                name=layer_name,
            )
            self._nodes.append(route_node)
        return layer_name, channels

    def _make_resize_node(self, layer_name, layer_dict):
        """Create an ONNX Resize node with the properties from
        the DarkNet-based graph.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        resize_scale_factors = float(layer_dict["stride"])
        # Create the scale factor array with node parameters
        scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
        previous_node_specs = self._get_previous_node_specs()
        inputs = [previous_node_specs.name]

        channels = previous_node_specs.channels
        assert channels > 0
        resize_params = ResizeParams(layer_name, scales)

        # roi input is the second input, so append it before scales
        roi_name = resize_params.generate_roi_name()
        inputs.append(roi_name)

        scales_name = resize_params.generate_param_name()
        inputs.append(scales_name)

        resize_node = helper.make_node(
            "Resize",
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(resize_node)
        self.param_dict[layer_name] = resize_params
        return layer_name, channels


def main():
    """Run the DarkNet-to-ONNX conversion for YOLOv3-608."""
    cfg_file_path = getFilePath("samples/python/yolov3_onnx/yolov3.cfg")
    # These are the only layers DarkNetParser will extract parameters from. The three layers of
    # type 'yolo' are not parsed in detail because they are included in the post-processing later:
    supported_layers = ["net", "convolutional", "shortcut", "route", "upsample"]

    # Create a DarkNetParser object, and the use it to generate an OrderedDict with all
    # layer's configs from the cfg file:
    parser = DarkNetParser(supported_layers)
    layer_configs = parser.parse_cfg_file(cfg_file_path)
    # We do not need the parser anymore after we got layer_configs:
    del parser

    # In above layer_config, there are three outputs that we need to know the output
    # shape of (in CHW format):
    output_tensor_dims = OrderedDict()
    output_tensor_dims["082_convolutional"] = [255, 19, 19]
    output_tensor_dims["094_convolutional"] = [255, 38, 38]
    output_tensor_dims["106_convolutional"] = [255, 76, 76]

    # Create a GraphBuilderONNX object with the known output tensor dimensions:
    builder = GraphBuilderONNX(output_tensor_dims)

    weights_file_path = getFilePath("samples/python/yolov3_onnx/yolov3.weights")

    # Now generate an ONNX graph with weights from the previously parsed layer configurations
    # and the weights file:
    yolov3_model_def = builder.build_onnx_graph(
        layer_configs=layer_configs, weights_file_path=weights_file_path, verbose=True
    )
    # Once we have the model definition, we do not need the builder anymore:
    del builder

    # Perform a sanity check on the ONNX model definition:
    onnx.checker.check_model(yolov3_model_def)

    # Serialize the generated ONNX graph to this file:
    output_file_path = "yolov3.onnx"
    onnx.save(yolov3_model_def, output_file_path)


if __name__ == "__main__":
    main()
