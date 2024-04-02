#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
TensorRT Engine Exploration API - Layer
"""


from .activations import *
from typing import Dict, List
import numpy as np


class Layer:
    def __init__(self, raw_dict: Dict):
        self.raw_dict = raw_dict
        self.name = raw_dict['Name']
        try:
            self.type = raw_dict['ParameterType']
        except KeyError:
            self.type = raw_dict['LayerType']
        try:
            self.metadata = raw_dict['Metadata']
        except KeyError:
            self.metadata = None
        self.subtype = raw_dict['LayerType']
        self.inputs = [Activation(tensor) for tensor in raw_dict['Inputs']]
        self.outputs = [Activation(tensor) for tensor in raw_dict['Outputs']]
        self.outputs_size_bytes = np.sum([i.size_bytes for i in self.outputs])
        if self.inputs:
            self.precision = self.inputs[0].precision
            self.inputs_size_bytes = np.sum([i.size_bytes for i in self.inputs])
        else:
            self.inputs_size_bytes = 0
            self.precision = None

        self.total_io_size_bytes = self.inputs_size_bytes + self.outputs_size_bytes
        self._parse_weights()
        self.total_footprint_bytes = self.total_io_size_bytes + self.weights_size

    def _parse_constant(const):
        cnt = const['Count']
        data_type = const['Type']
        try:
            data_size = dict(
                {"Int8": 1, "Half": 2, "Float": 4, "Int32": 4})[data_type]
        except KeyError:
            # Backward compatbility.
            data_size = dict(
                {"Int8": 1, "FP16": 2, "FP32": 4, "Int32": 4})[data_type]
        return cnt, data_type, cnt * data_size

    def _parse_weights(self):
        try:
            self.weights_cnt, self.weights_type, self.weights_size = \
                Layer._parse_constant(self.raw_dict['Weights'])
        except KeyError:
            self.weights_cnt = 0
            self.weights_type = None
            self.weights_size = 0
        try:
            self.bias_cnt, self.bias_type, self.bias_size = \
                Layer._parse_constant(self.raw_dict['Bias'])
        except KeyError:
            self.bias_cnt = 0
            self.bias_type = None
            self.bias_size = 0

    def tooltip(self):
        tip = ""
        for key, value in sorted(self.raw_dict.items()):
            if key not in ['InputRegions', 'OutputRegions', 'Inputs', 'Outputs',
                    'ParameterType', 'LayerName']:
                tip += f"{key}:{value}\\n"
        return tip

    def __repr__(self):
        rep = f"Layer({self.name})"
        return rep


def fold_no_ops(layers: List, bindings: List) -> List:
    """Remove layers of type No-Op"""

    def consumers_producers_dict(layers) -> Dict[Activation, List[Layer]]:
        """Return a dictionary of consumer-layers per activation tensor"""
        consumers, producers = {}, {}
        for layer in layers:
            for loc, input in enumerate(layer.inputs):
                try:
                    consumers[input.name].append((layer.name, loc))
                except KeyError:
                    consumers[input.name] = [(layer.name, loc)]
            for loc, output in enumerate(layer.outputs):
                try:
                    producers[output.name].append((layer.name, loc))
                except KeyError:
                    producers[output.name] = [(layer.name, loc)]
        return consumers, producers

    def move_input(src: Layer, dst: Layer, loc: int=0):
        # Can safely assume src input port #0 because NoOp has only a single input.
        dst.inputs[loc] = src.inputs[0]

    def move_output(src: Layer, dst: Layer, loc: int=0):
        # Can safely assume src output port #0 because NoOp has only a single output.
        dst.outputs[loc] = src.outputs[0]

    def fold(no_op: Layer):
        try:
            successors = activation_consumers[no_op.outputs[0].name]
            for successor in successors:
                successor_name, successor_in_port = successor
                move_input(
                    src=no_op, dst=ret[successor_name], loc=successor_in_port)
        except KeyError:
            # A leaf NoOp layer: it's output is a binding so we need to move the
            # NoOp output to the output of the previous layer.
            if no_op.outputs[0].name in bindings :
                predecessors = activation_producers[no_op.inputs[0].name]
                for predecessor in predecessors:
                    predecessor_name, predecessor_out_port = predecessor
                    move_output(
                        src=no_op, dst=ret[predecessor_name], loc=predecessor_out_port)
            pass

    ret = {layer.name: layer for layer in layers}
    activation_consumers, activation_producers = consumers_producers_dict(layers)

    for layer in layers:
        if layer.type == 'NoOp':
            fold(layer)

    # Remove the No-Op layers from the final list.
    ret = [layer for layer in ret.values() if layer.type != 'NoOp']
    return ret
