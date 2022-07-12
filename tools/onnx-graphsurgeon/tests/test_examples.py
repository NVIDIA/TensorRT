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
import subprocess as sp
import sys
import tempfile

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime
import pytest
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.util import misc

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))
EXAMPLES_ROOT = os.path.join(ROOT_DIR, "examples")


class Artifact(object):
    def __init__(self, name, infer=True):
        self.name = name
        self.infer = infer


EXAMPLES = [
    ("01_creating_a_model", [Artifact("test_globallppool.onnx")]),
    ("02_creating_a_model_with_initializer", [Artifact("test_conv.onnx")]),
    ("03_isolating_a_subgraph", [Artifact("model.onnx"), Artifact("subgraph.onnx")]),
    ("04_modifying_a_model", [Artifact("model.onnx"), Artifact("modified.onnx")]),
    ("05_folding_constants", [Artifact("model.onnx"), Artifact("folded.onnx")]),
    ("06_removing_nodes", [Artifact("model.onnx", infer=False), Artifact("removed.onnx")]),
    ("07_creating_a_model_with_the_layer_api", [Artifact("model.onnx")]),
    ("08_replacing_a_subgraph", [Artifact("model.onnx"), Artifact("replaced.onnx")]),
    ("09_shape_operations_with_the_layer_api", [Artifact("model.onnx")]),
    ("10_dynamic_batch_size", [Artifact("model.onnx"), Artifact("dynamic.onnx")]),
]

# Extract any ``` blocks from the README
def load_commands_from_readme(readme):
    def ignore_command(cmd):
        return "pip" in cmd

    commands = []
    with open(readme, "r") as f:
        in_command_block = False
        for line in f.readlines():
            if not in_command_block and "```bash" in line:
                in_command_block = True
            elif in_command_block:
                if "```" in line:
                    in_command_block = False
                elif not ignore_command(line):
                    commands.append(line.strip())
    return commands


def infer_model(path):
    model = onnx.load(path)
    graph = gs.import_onnx(model)

    feed_dict = {}
    for tensor in graph.inputs:
        shape = tuple(dim if not misc.is_dynamic_dimension(dim) else 1 for dim in tensor.shape)
        feed_dict[tensor.name] = np.random.random_sample(size=shape).astype(tensor.dtype)

    output_names = [out.name for out in graph.outputs]

    sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = sess.run(output_names, feed_dict)
    G_LOGGER.info("Inference outputs: {:}".format(outputs))
    return outputs


@pytest.mark.parametrize("example_dir,artifacts", EXAMPLES)
def test_examples(example_dir, artifacts):
    example_dir = os.path.join(EXAMPLES_ROOT, example_dir)
    readme = os.path.join(example_dir, "README.md")
    commands = load_commands_from_readme(readme)
    for command in commands:
        G_LOGGER.info(command)
        assert sp.run(["bash", "-c", command], cwd=example_dir, env={"PYTHONPATH": ROOT_DIR}).returncode == 0

    for artifact in artifacts:
        artifact_path = os.path.join(example_dir, artifact.name)
        assert os.path.exists(artifact_path)
        if artifact.infer:
            assert infer_model(artifact_path)
        os.remove(artifact_path)
