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

"""
A totally fake checker script that lets us test the functionality
of `debug reduce`, which reduces ONNX models to minimal failing subgraphs,
by simulating failures in arbitrary nodes.
"""
import argparse
import sys
import onnx


def main():
    parser = argparse.ArgumentParser(description="Makes Polygraphy think a node in a model is failing")
    parser.add_argument("model", help="The ONNX model")
    parser.add_argument(
        "--fail-node",
        help="The name(s) of the node(s) that 'fails'. "
        "If multiple nodes are specified, they must all be present to cause a failure.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--default-return-code", help="The return code to use when there are no failures. ", default=0, type=int
    )
    parser.add_argument(
        "--fail-return-code", help="The return code to use when there is a failure. ", default=1, type=int
    )

    args = parser.parse_args()

    print("Running fake_reduce_checker.py")

    model = onnx.load(args.model)
    print(model)
    node_names = [node.name for node in model.graph.node]

    if all(fail_node in node_names for fail_node in args.fail_node):
        # Alternate error messages to test --fail-regex
        if "onnx_graphsurgeon_node_1" in node_names:
            print("REALLY BAD!")
        else:
            print("FOUND A BAD NODE!")
        return args.fail_return_code
    return args.default_return_code


if __name__ == "__main__":
    sys.exit(main())
