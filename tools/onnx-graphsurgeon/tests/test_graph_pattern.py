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

import os

import onnx

import pytest

from onnx_graphsurgeon import GraphPattern, PatternMapping
from onnx_graphsurgeon.importers.onnx_importer import import_onnx
from onnx_graphsurgeon.logger import G_LOGGER

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE


class TestGraphPatternMatching:
    def get_plugin_io_and_attrs(self, subgraph: PatternMapping):
        inputs = []
        inputs.append(subgraph.get("Anode").inputs[0])
        inputs.append(subgraph.get("Bnode").inputs[0])

        attrs = dict()
        attrs["x"] = subgraph.get("Cnode").attrs["x"]

        outputs = []
        outputs.append(subgraph.get("Dnode").outputs[0])
        outputs.append(subgraph.get("Enode").outputs[0])

        return inputs, outputs, attrs

    def get_plugin_pattern(self):
        """
        Toy plugin pattern:
            A     B
              \   /
                C, attrs['x'] < 2.0
              /   \
            D     E
        """
        pattern = GraphPattern()
        # in_0, in_1 = pattern.set_input_tensors(2)
        in_0 = pattern.variable()
        in_1 = pattern.variable()
        a_out = pattern.add("Anode", "A", inputs=[in_0])
        b_out = pattern.add("Bnode", "B", inputs=[in_1])
        check_function = lambda node: node.attrs["x"] < 2.0
        c_out = pattern.add(
            "Cnode", "C", inputs=[a_out, b_out], check_func=check_function
        )
        d_out = pattern.add("Dnode", "D", inputs=[c_out])
        e_out = pattern.add("Enode", "E", inputs=[c_out])
        pattern.set_output_tensors([d_out, e_out])

        return pattern

    def test_base_match(self):
        path = os.path.join(TEST_ROOT, "models", "test_toyPlugin_base_match_case.onnx")
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern()

        matched_subgraphs = graph_pattern.match_all(graph)
        assert len(matched_subgraphs) == 1
        sg = matched_subgraphs[0]

        inputs, outputs, attrs = self.get_plugin_io_and_attrs(sg)

        # node-to-node mapping
        assert sg.get("Anode").name == "n2"
        assert sg.get("Bnode").name == "n3"
        assert sg.get("Cnode").name == "n4"
        assert sg.get("Dnode").name == "n5"
        assert sg.get("Enode").name == "n6"

        # I/O mapping
        assert inputs[0].name == "i1" and inputs[1].name == "i1"
        assert outputs[0].name == "o1" and outputs[1].name == "o2"

        # attrs mapping
        assert attrs["x"] == 1.0

    def test_callback_check_unmatch(self):
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_callback_check_unmatch_case.onnx"
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern()

        matched_subgraphs = graph_pattern.match_all(graph)

        # No matched subgraph due to the callback check failure for attrs.
        assert len(matched_subgraphs) == 0

    def test_intermediate_output_unmatch(self):
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_intermediate_output_unmatch_case.onnx"
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern()

        matched_subgraphs = graph_pattern.match_all(graph)

        # No matched subgraph due to the callback check failure for attrs.
        assert len(matched_subgraphs) == 0

    def test_intermediate_output_to_other_node_unmatch(self):
        path = os.path.join(
            TEST_ROOT,
            "models",
            "test_toyPlugin_intermediate_output_to_other_node_unmatch_case.onnx",
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern()

        matched_subgraphs = graph_pattern.match_all(graph)

        # No matched subgraph due to the callback check failure for attrs.
        assert len(matched_subgraphs) == 0


class TestGraphPatternBuilding:
    def get_plugin_io_and_attrs(self, subgraph: PatternMapping):
        inputs = []
        inputs.append(subgraph.get("left").get("Anode").inputs[0])
        inputs.append(subgraph.get("right").get("Anode").inputs[0])

        attrs = dict()

        outputs = []
        outputs.append(subgraph.get("Cnode").outputs[0])

        return inputs, outputs, attrs

    def get_plugin_pattern(self):
        """
        Graph pattern:
             A     A
             |     |
             B     B
              \   /
                C
        """
        subpattern = GraphPattern()
        # i0 = subpattern.set_input_tensors(1)
        i0 = subpattern.variable()
        a_node = subpattern.add("Anode", "A", inputs=[i0], num_output_tensors=1)
        b_out = subpattern.add("Bnode", "B", inputs=[a_node], num_output_tensors=1)
        subpattern.set_output_tensors([b_out])

        pattern = GraphPattern()
        # in_0, in_1 = pattern.set_input_tensors(2)
        in_0 = pattern.variable()
        in_1 = pattern.variable()
        left = pattern.add("left", subpattern, inputs=[in_0], num_output_tensors=1)
        right = pattern.add("right", subpattern, inputs=[in_1], num_output_tensors=1)
        c_out = pattern.add("Cnode", "C", inputs=[left, right], num_output_tensors=1)
        pattern.set_output_tensors([c_out])

        return pattern

    def test_recursive_pattern_building(self):
        path = os.path.join(TEST_ROOT, "models", "test_recursive_pattern_building.onnx")
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern()

        matched_subgraphs = graph_pattern.match_all(graph)
        assert len(matched_subgraphs) == 1
        sg = matched_subgraphs[0]
        print(sg)

        inputs, outputs, attrs = self.get_plugin_io_and_attrs(sg)

        # node-to-node mapping
        assert sg.get("left").get("Anode").name == "n1"
        assert sg.get("left").get("Bnode").name == "n3"
        assert sg.get("Cnode").name == "n5"
        assert sg.get("right").get("Anode").name == "n2"
        assert sg.get("right").get("Bnode").name == "n4"

        # I/O mapping
        assert inputs[0].name == "i0" and inputs[1].name == "i1"
        assert outputs[0].name == "i6"


class TestOutputNodes:
    def get_plugin_io_and_attrs(self, subgraph: PatternMapping):
        inputs = []
        inputs.append(subgraph.get("Anode").inputs[0])
        inputs.append(subgraph.get("Bnode").inputs[0])

        attrs = dict()
        attrs["x"] = subgraph.get("Cnode").attrs["x"]

        outputs = []
        outputs.append(subgraph.get("Dnode").outputs[0])
        outputs.append(subgraph.get("Enode").outputs[0])
        outputs.append(subgraph.get("Bnode").outputs[0])

        return inputs, outputs, attrs

    def get_plugin_pattern(self):
        r"""
        Toy plugin pattern:
            A     B
              \   / \
                C    |
              /   \  |
            D     E  |
        """
        pattern = GraphPattern()
        # in_0, in_1 = pattern.set_input_tensors(2)
        in_0 = pattern.variable()
        in_1 = pattern.variable()
        a_out = pattern.add("Anode", "A", inputs=[in_0])
        b_out = pattern.add("Bnode", "B", inputs=[in_1])
        c_out = pattern.add("Cnode", "C", inputs=[a_out, b_out])
        d_out = pattern.add("Dnode", "D", inputs=[c_out])
        e_out = pattern.add("Enode", "E", inputs=[c_out])
        pattern.set_output_tensors([d_out, e_out, b_out])

        return pattern

    def get_plugin_pattern_with_multiple_output_node(self):
        r"""
        Toy plugin pattern: B has two different outputs.
            A     B
              \   / \
                C    |
              /   \  |
            D     E  |
        """

        pattern = GraphPattern()
        # in_0, in_1 = pattern.set_input_tensors(2)
        in_0 = pattern.variable()
        in_1 = pattern.variable()
        a_out = pattern.add("Anode", "A", inputs=[in_0])
        b_out_0, b_out_1 = pattern.add(
            "Bnode", "B", inputs=[in_1], num_output_tensors=2
        )
        c_out = pattern.add("Cnode", "C", inputs=[a_out, b_out_0])
        d_out = pattern.add("Dnode", "D", inputs=[c_out])
        e_out = pattern.add("Enode", "E", inputs=[c_out])
        pattern.set_output_tensors([d_out, e_out, b_out_1])

        return pattern

    def test_outbound_node_with_consumer_match(self):
        # special case: B has consumers but it is an outbound node.
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_intermediate_output_unmatch_case.onnx"
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern()

        matched_subgraphs = graph_pattern.match_all(graph)
        assert len(matched_subgraphs) == 1
        sg = matched_subgraphs[0]

        inputs, outputs, attrs = self.get_plugin_io_and_attrs(sg)

        # node-to-node mapping
        assert sg.get("Anode").name == "n2"
        assert sg.get("Bnode").name == "n3"
        assert sg.get("Cnode").name == "n4"
        assert sg.get("Dnode").name == "n5"
        assert sg.get("Enode").name == "n6"

        # I/O mapping
        assert inputs[0].name == "i1" and inputs[1].name == "i1"
        assert (
            outputs[0].name == "o1"
            and outputs[1].name == "o2"
            and outputs[2].name == "i3"
        )

    def test_multiple_output_node_unmatch(self):
        # special case: B has 2 outputs in pattern, but onnx model only has one output.
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_intermediate_output_unmatch_case.onnx"
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern_with_multiple_output_node()

        matched_subgraphs = graph_pattern.match_all(graph)
        assert len(matched_subgraphs) == 0


class TestConstantCases:
    def get_plugin_pattern_constant_node(self):
        r"""
        Toy plugin pattern:
            A      Constant
              \   /
               B
        """
        pattern = GraphPattern()
        in_0 = pattern.variable()
        a_out = pattern.add("Anode", "A", inputs=[in_0])
        c_out = pattern.add("ConstantNode", "Constant")
        b_out = pattern.add("Bnode", "B", inputs=[a_out, c_out])
        pattern.set_output_tensors([b_out])

        return pattern

    def get_plugin_pattern_constant_tensor(self):
        r"""
        Toy plugin pattern:
            A      Constant
              \   /
               B
        """
        pattern = GraphPattern()
        in_0 = pattern.variable()
        a_out = pattern.add("Anode", "A", inputs=[in_0])
        c_out = pattern.constant()
        b_out = pattern.add("Bnode", "B", inputs=[a_out, c_out])
        pattern.set_output_tensors([b_out])

        return pattern

    def get_plugin_pattern_no_constant(self):
        r"""
        Toy plugin pattern:
            A
            |
            B
        """
        pattern = GraphPattern()
        in_0 = pattern.variable()
        a_out = pattern.add("Anode", "A", inputs=[in_0])
        b_out = pattern.add("Bnode", "B", inputs=[a_out])
        pattern.set_output_tensors([b_out])

        return pattern

    def test_constant_initializer_match(self):
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_constant_initializer_match_case.onnx"
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern_constant_tensor()

        matched_subgraphs = graph_pattern.match_all(graph)
        assert len(matched_subgraphs) == 1

    def test_constant_node_match(self):
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_constant_node_match_case.onnx"
        )
        graph = import_onnx(onnx.load(path))

        graph_pattern = self.get_plugin_pattern_constant_node()
        matched_subgraphs = graph_pattern.match_all(graph)

        assert len(matched_subgraphs) == 1

    def test_constant_initializer_unmatch(self):
        path = os.path.join(
            TEST_ROOT, "models", "test_toyPlugin_constant_initializer_match_case.onnx"
        )
        graph = import_onnx(onnx.load(path))
        graph_pattern = self.get_plugin_pattern_no_constant()
        matched_subgraphs = graph_pattern.match_all(graph)

        assert len(matched_subgraphs) == 0
