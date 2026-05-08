#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def to_1d(arr):
    return np.asarray(arr).reshape(-1)


def fold_groupnorm(graph):
    folded = 0

    for inst in [n for n in graph.nodes if n.op == "InstanceNormalization"]:
        pre_input = inst.inputs[0]
        if not pre_input.inputs or pre_input.inputs[0].op != "Reshape":
            continue
        pre = pre_input.inputs[0]

        post_consumers = inst.outputs[0].outputs
        if len(post_consumers) != 1 or post_consumers[0].op != "Reshape":
            continue
        post = post_consumers[0]

        if not isinstance(pre.inputs[1], gs.Constant):
            continue
        target_shape = pre.inputs[1].values.tolist()
        if len(target_shape) < 2 or int(target_shape[1]) <= 0:
            continue
        num_groups = int(target_shape[1])

        mul_consumers = post.outputs[0].outputs
        if len(mul_consumers) != 1 or mul_consumers[0].op != "Mul":
            continue
        mul = mul_consumers[0]

        add_consumers = mul.outputs[0].outputs
        if len(add_consumers) != 1 or add_consumers[0].op != "Add":
            continue
        addn = add_consumers[0]

        gamma_t = mul.inputs[0] if mul.inputs[1] is post.outputs[0] else mul.inputs[1]
        beta_t = addn.inputs[0] if addn.inputs[1] is mul.outputs[0] else addn.inputs[1]
        if not isinstance(gamma_t, gs.Constant) or not isinstance(beta_t, gs.Constant):
            continue

        gamma = to_1d(gamma_t.values).astype(np.float32)
        beta = to_1d(beta_t.values).astype(np.float32)
        if gamma.shape != beta.shape:
            continue

        epsilon = inst.attrs.get("epsilon", 1e-5)

        gn_scale = gs.Constant(name=inst.name + "_gn_scale", values=gamma)
        gn_bias = gs.Constant(name=inst.name + "_gn_bias", values=beta)
        x_tensor = pre.inputs[0]
        gn_out = addn.outputs[0]

        gn_out.inputs.clear()
        gn_node = gs.Node(
            op="GroupNormalization",
            name=inst.name + "_folded",
            attrs={"num_groups": num_groups, "epsilon": float(epsilon)},
            inputs=[x_tensor, gn_scale, gn_bias],
            outputs=[gn_out],
        )
        graph.nodes.append(gn_node)
        folded += 1

    if folded:
        graph.cleanup().toposort()
    return folded


def bump_opset(model, min_version=21):
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < min_version:
            op.version = min_version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="model.onnx")
    ap.add_argument("output", nargs="?", default="folded.onnx")
    args = ap.parse_args()

    graph = gs.import_onnx(onnx.load(args.input))
    n = fold_groupnorm(graph)
    print(f"folded {n} pattern(s)")

    model = gs.export_onnx(graph)
    bump_opset(model, 21)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
