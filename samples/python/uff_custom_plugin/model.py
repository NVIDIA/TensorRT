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
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import graphsurgeon as gs
import uff

# lenet5.py
from lenet5 import ModelData

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path to which trained model will be saved (check README.md)
MODEL_PATH = os.path.join(WORKING_DIR, "models/trained_lenet5.pb")

# Generates mappings from unsupported TensorFlow operations to TensorRT plugins
def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.relu6, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.relu6.

    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_relu6 = gs.create_plugin_node(name="trt_relu6", op="CustomClipPlugin", clipMin=0.0, clipMax=6.0)
    namespace_plugin_map = {ModelData.RELU6_NAME: trt_relu6}
    return namespace_plugin_map


# Transforms model path to uff path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path


# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
    # Normally the relu6 layer's output is placed at index 0 of the consumer layer's input list.
    # But when running in Windows, the relu6 layer's output might be placed in a wrong index due to different implementation of tensorflow.NodeDef.
    for node in dynamic_graph:
        if len(node.input) == 2 and node.input[1] == "trt_relu6":
            # Reorder the input list to make sure that the relu6 layer's output is always placed at index 0.
            node.input.reverse()
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(), [ModelData.OUTPUT_NAME], output_filename=output_uff_path, text=True
    )
    return output_uff_path


def main():
    # Load pretrained model
    if not os.path.isfile(MODEL_PATH):
        raise IOError(
            "\n{}\n{}\n{}\n".format(
                "Failed to load model file ({}).".format(MODEL_PATH),
                "Please use 'python lenet5.py' to train and save the model.",
                "For more information, see the included README.md",
            )
        )

    uff_path = model_to_uff(MODEL_PATH)
    print("Saved converted UFF model to: " + uff_path)


if __name__ == "__main__":
    main()
