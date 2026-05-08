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


def make_legacy_groupnorm_pattern():
    pat = gs.GraphPattern()

    x = pat.variable()
    target_shape = pat.variable()
    inst_scale = pat.variable()
    inst_bias = pat.variable()
    gamma = pat.variable()
    beta = pat.variable()

    reshape_in = pat.add("pre", "Reshape", inputs=[x, target_shape])
    inst_out = pat.add("inst", "InstanceNormalization", inputs=[reshape_in, inst_scale, inst_bias])
    shape_out = pat.add("shape", "Shape", inputs=[x])
    reshape_back = pat.add("post", "Reshape", inputs=[inst_out, shape_out])
    mul_out = pat.add("mul", "Mul", inputs=[reshape_back, gamma])
    add_out = pat.add("add", "Add", inputs=[mul_out, beta])

    pat.set_output_tensors([add_out])
    return pat


def fold_groupnorm(graph):
    pattern = make_legacy_groupnorm_pattern()
    matches = pattern.match_all(graph)

    folded = 0
    for match in matches:
        pre = match.get("pre")
        inst = match.get("inst")
        mul = match.get("mul")
        addn = match.get("add")

        target = pre.inputs[1]
        if not isinstance(target, gs.Constant):
            continue
        target_shape = target.values.tolist()
        if len(target_shape) < 2 or int(target_shape[1]) <= 0:
            continue
        num_groups = int(target_shape[1])

        gamma_t = mul.inputs[1]
        beta_t = addn.inputs[1]
        if not isinstance(gamma_t, gs.Constant) or not isinstance(beta_t, gs.Constant):
            continue
        gamma = to_1d(gamma_t.values).astype(np.float32)
        beta = to_1d(beta_t.values).astype(np.float32)
        if gamma.shape != beta.shape:
            continue

        epsilon = inst.attrs.get("epsilon", 1e-5)

        x_tensor = pre.inputs[0]
        gn_out = addn.outputs[0]
        gn_out.inputs.clear()

        graph.nodes.append(
            gs.Node(
                op="GroupNormalization",
                name=inst.name + "_folded",
                attrs={"num_groups": num_groups, "epsilon": float(epsilon)},
                inputs=[
                    x_tensor,
                    gs.Constant(name=inst.name + "_gn_scale", values=gamma),
                    gs.Constant(name=inst.name + "_gn_bias", values=beta),
                ],
                outputs=[gn_out],
            )
        )
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
