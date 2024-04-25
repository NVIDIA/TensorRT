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

import pytest
import os
import yaml
from tests.models.meta import ONNX_MODELS
import onnx
import tempfile


class TestMatch:

    TOY_MODEL_PATH = ONNX_MODELS["graph_with_subgraph_matching_toy_plugin"].path
    PLUGINS_PATH = os.path.join(os.path.dirname(TOY_MODEL_PATH), "plugins")

    @pytest.mark.script_launch_mode("subprocess")
    def test_match_toy(self, poly_plugin_match):

        assert os.path.exists(self.TOY_MODEL_PATH)

        with tempfile.TemporaryDirectory() as outdir:
            config_yaml_loc = os.path.join(outdir, "config.yaml")
            poly_plugin_match(
                [
                    self.TOY_MODEL_PATH,
                    "--plugin-dir",
                    self.PLUGINS_PATH,
                    "-o",
                    config_yaml_loc,
                ]
            )

            assert os.path.exists(config_yaml_loc)

            with open(config_yaml_loc, "r") as stream:
                config_yaml = yaml.safe_load_all(stream)

                num_plugins = 0
                for plugin in config_yaml:
                    num_plugins += 1
                    assert plugin["name"] == "toyPlugin"
                    assert len(plugin["instances"]) == 1
                    assert len(plugin["instances"][0]["inputs"]) == 1
                    assert len(plugin["instances"][0]["outputs"]) == 2
                    assert plugin["instances"][0]["attributes"]["ToyX"] == 2

                assert num_plugins == 1

    @pytest.mark.script_launch_mode("subprocess")
    def test_list_toy(self, poly_plugin_list_plugins):
        status = poly_plugin_list_plugins(
            [self.TOY_MODEL_PATH, "--plugin-dir", self.PLUGINS_PATH]
        )

        assert "{'toyPlugin': 1}" in status.stdout

    @pytest.mark.script_launch_mode("subprocess")
    def test_replace_toy(self, poly_plugin_replace, poly_plugin_match):
        with tempfile.TemporaryDirectory() as outdir:
            config_yaml_loc = os.path.join(outdir, "config.yaml")

            poly_plugin_match(
                [
                    self.TOY_MODEL_PATH,
                    "--plugin-dir",
                    self.PLUGINS_PATH,
                    "-o",
                    config_yaml_loc,
                ]
            )

            replaced_loc = os.path.join(outdir, "replaced.onnx")
            poly_plugin_replace(
                [
                    self.TOY_MODEL_PATH,
                    "--plugin-dir",
                    self.PLUGINS_PATH,
                    "--config",
                    config_yaml_loc,
                    "-o",
                    replaced_loc,
                ]
            )

            model = onnx.load(replaced_loc)
            assert len(model.graph.node) == 2
            node_names = {node.name for node in model.graph.node}

            assert "n1" in node_names
            assert not node_names.intersection({"n2", "n3", "n4", "n5", "n6"})
            assert model.graph.node[1].op_type == "CustomToyPlugin"
            assert model.graph.node[1].attribute[0].name == "ToyX"
            assert model.graph.node[1].attribute[0].i == 2
