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

import onnx_graphsurgeon as gs
import onnx
import numpy as np
import json

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from downloader import getFilePath


def drop_category_mapper_nodes(graph):
    new_inputs = []
    for org_input in graph.inputs:
        # head node, simply disconnect it with others
        assert len(org_input.outputs) == 1
        category_mapper_node = org_input.outputs[0]
        assert category_mapper_node.op == "CategoryMapper"
        assert len(category_mapper_node.outputs) == 1
        new_inputs.append(category_mapper_node.outputs[0])
        category_mapper_node.inputs.clear()
        category_mapper_node.outputs.clear()

        # Save mapping info to preprocess inputs.
        with open(category_mapper_node.name + ".json", "w") as fp:
            json.dump(category_mapper_node.attrs, fp)

    graph.inputs = new_inputs


def replace_unsupported_ops(graph):
    # replace hardmax with ArgMax
    hardmaxes = [node for node in graph.nodes if node.op == "Hardmax"]
    assert len(hardmaxes) == 1
    hardmax = hardmaxes[0]
    hardmax.op = "ArgMax"
    hardmax.name = "ArgMax(org:" + hardmax.name + ")"
    hardmax.attrs["axis"] = 1
    hardmax.attrs["keepdims"] = 0

    cast = hardmax.o()
    reshape = cast.o()

    hardmax.outputs = reshape.outputs
    assert len(hardmax.outputs) == 1
    hardmax.outputs[0].dtype = np.int64
    hardmax.outputs[0].shape = [1]

    compress = reshape.o()
    compress.op = "Gather"
    compress.name = "Gather(org:" + compress.name + ")"
    compress.attrs["axis"] = 1

    cast.outputs.clear()
    reshape.outputs.clear()
    # Remove the node from the graph completely
    graph.cleanup().toposort()


def save_weights_for_refitting(graph):
    # Save weights for refitting
    tmap = graph.tensors()
    np.save("Parameter576_B_0.npy", tmap["Parameter576_B_0"].values)
    np.save("W_0.npy", tmap["W_0"].values)


def main():
    org_model_file_path = getFilePath("samples/python/engine_refit_onnx_bidaf/bidaf-original.onnx")

    print("Modifying the ONNX model ...")
    original_model = onnx.load(org_model_file_path)
    graph = gs.import_onnx(original_model)

    drop_category_mapper_nodes(graph)
    replace_unsupported_ops(graph)
    save_weights_for_refitting(graph)

    new_model = gs.export_onnx(graph)

    modified_model_name = "bidaf-modified.onnx"
    onnx.checker.check_model(new_model)
    onnx.save(new_model, modified_model_name)
    print("Modified ONNX model saved as {}".format(modified_model_name))
    print("Done.")


if __name__ == "__main__":
    main()
