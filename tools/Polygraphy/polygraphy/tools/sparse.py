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

import sys
from collections import namedtuple

from polygraphy import mod, util
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER

onnx = mod.lazy_import("onnx")
onnx_numpy_helper = mod.lazy_import("onnx.numpy_helper")
np = mod.lazy_import("numpy")


PruneInfo = namedtuple("PruneInfo", ["name", "axis"])


class SparsityPruner:
    def __init__(self, model):
        self.model = model
        g = model.graph
        self.g = g
        # map: initializer name -> object
        self.w_name2obj = {t.name: t for t in g.initializer}
        # map: tensor name -> producer node object
        self.tname2producer = dict()
        for n in g.node:
            for t in n.output:
                self.tname2producer[t] = n

        self.prune_infos = dict()
        self.sparse_tensors = set()
        self.weights_skip = set()

    # Look back through Q/DQ/Cast nodes
    def __tensor(self, t, axis):
        if t in self.w_name2obj:
            G_LOGGER.super_verbose(f"Tracking weight: ({t})")
            self.prune_infos[t] = PruneInfo(t, axis)
            return

        axis_insensitive_op_type = [
            "QuantizeLinear",
            "DequantizeLinear",
            "TRT_FP8QuantizeLinear",
            "TRT_FP8DequantizeLinear",
            "Cast",
        ]
        stop_op_type = [
            "LayerNormalization",
            "Reshape",
            "Concat",
            "Slice",
            "Shape",
            "Unsqueeze",
            "Gather",
            "Mul",
            "Add",
        ]
        if t in self.tname2producer:
            producer = self.tname2producer[t]
            if producer.op_type in axis_insensitive_op_type:
                G_LOGGER.ultra_verbose(
                    f"({t}) is produced by {producer.op_type}, looking back"
                )
                self.__tensor(producer.input[0], axis)
            elif producer.op_type == "Transpose":
                G_LOGGER.ultra_verbose(
                    f"({t}) is produced by {producer.op_type}, checking attributes"
                )
                for attr in producer.attribute:
                    if attr.name == "perm":
                        perm = list(attr.ints)
                        new_axis = perm.index(axis)
                        G_LOGGER.ultra_verbose(
                            f"attribute <perm> is {perm}, axis {axis} -> {new_axis}"
                        )
                        self.__tensor(producer.input[0], new_axis)
                        return
                G_LOGGER.warning(f"{producer.op_type} doesn't have <perm> attribute!")
            elif producer.op_type in stop_op_type:
                G_LOGGER.ultra_verbose(
                    f"({t}) produced by {producer.name} type {producer.op_type}. Stopping backward analysis."
                )
            else:
                G_LOGGER.warning(
                    f"({t}) produced by {producer.name} type: {producer.op_type} is unsupported!"
                )

    def __conv(self, node):
        assert node.op_type == "Conv"
        w = node.input[1]
        self.__tensor(w, 1)

    def __matmul(self, node):
        assert node.op_type == "MatMul"
        a = node.input[0]
        b = node.input[1]
        self.__tensor(a, 1)
        self.__tensor(b, 0)

    def __gemm(self, node):
        assert node.op_type == "Gemm"
        a = node.input[0]
        b = node.input[1]

        # get attributes
        trans_a = False
        trans_b = False
        attrs = node.attribute
        for attr in attrs:
            if attr.name == "transA":
                trans_a = attr.i == 1
            elif attr.name == "transB":
                trans_b = attr.i == 1

        # check
        axis = 0 if trans_a else 1
        self.__tensor(a, axis)
        axis = 1 if trans_b else 0
        self.__tensor(b, axis)

    def _walk_nodes(self):
        G_LOGGER.verbose(f"Walking graph to collect weights candidates.")
        assert len(self.prune_infos) == 0
        count = len(self.g.node)
        for i in range(count):
            n = self.g.node[i]
            G_LOGGER.super_verbose(
                f"Processing node {i}/{count} ({n.op_type}): {n.name}"
            )
            if n.op_type == "MatMul":
                self.__matmul(n)
            elif n.op_type == "Gemm":
                self.__gemm(n)
            elif n.op_type == "Conv":
                self.__conv(n)
            else:
                pass
        G_LOGGER.verbose(f"Collected {len(self.prune_infos)} weights candidates.")

        G_LOGGER.verbose("Skipping tensors that are not eligible for pruning.")
        prune_infos = list(self.prune_infos.values())
        count = len(prune_infos)
        final_prune_infos = []
        for i in range(count):
            pinfo = prune_infos[i]
            G_LOGGER.super_verbose(f"Processing tensor {i + 1}/{count}: {pinfo}")
            t = self.w_name2obj[pinfo.name]
            if t.name in self.weights_skip:
                G_LOGGER.warning(
                    f"Skipping tensor: {t.name} since it was marked to skip pruning"
                )
                continue
            supported_dtypes = [
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.BFLOAT16,
            ]
            if not t.data_type in supported_dtypes:
                G_LOGGER.warning(
                    f"Skipping tensor: {t.name} due to unsupported type: {DataType.from_dtype(t.data_type, 'onnx')}"
                )
                continue
            assert pinfo.axis < len(t.dims)
            dim = t.dims[pinfo.axis]
            if dim % 4 != 0:
                G_LOGGER.verbose(
                    f"Skipping {t.name} since the length of axis {pinfo.axis} ({dim} in {t.dims}) is not a multiple of 4. "
                )
                continue
            final_prune_infos.append(pinfo)

        new_count = len(final_prune_infos)
        G_LOGGER.extra_verbose(
            f"Skipped {count - new_count} of {count} tensor(s) since they are not eligible for pruning. "
        )
        G_LOGGER.info(f"Found: {new_count} weight tensor(s) eligible for pruning.")
        return final_prune_infos

    def process(self, check):
        # Walk nodes to collect the tensors (initializers) that need to be pruned and the axis.
        prune_infos = self._walk_nodes()
        count = len(prune_infos)

        if check:
            G_LOGGER.start(f"Checking the sparsity pattern of {count} tensors.")
            for i in range(count):
                pinfo = prune_infos[i]
                tensor = self.w_name2obj[pinfo.name]
                G_LOGGER.extra_verbose(f"Checking tensor {i + 1}/{count}: {pinfo.name}")
                is_sparse = process_tensor(pinfo, tensor, True)
                if is_sparse:
                    self.sparse_tensors.add(tensor.name)
            G_LOGGER.finish(f"Finished checking {count} tensors. ")
            return None
        else:
            G_LOGGER.start(f"Pruning {count} tensors.")
            new_w_name2obj = dict()
            for i in range(count):
                pinfo = prune_infos[i]
                tensor = self.w_name2obj[pinfo.name]
                G_LOGGER.extra_verbose(f"Pruning tensor {i+ 1}/{count}: {pinfo.name}")
                new_t = process_tensor(pinfo, tensor, False)
                new_w_name2obj[new_t.name] = new_t
            G_LOGGER.finish(f"Finished pruning {count} tensors. ")

            return build_new_model(self.model, new_w_name2obj)

    def prune(self, weights_skip=set()):
        self.weights_skip = weights_skip
        return self.process(False)

    def check(self):
        self.process(True)


def process_bf16_tensor(tensor, outer, pdim, pstride, check):
    G_LOGGER.super_verbose("Processing BF16 tensor")
    assert tensor.data_type == onnx.TensorProto.BFLOAT16
    is_raw_data = len(tensor.int32_data) == 0
    data = bytearray(tensor.raw_data) if is_raw_data else tensor.int32_data
    step = 4 if is_raw_data else 2

    ostride = pdim * pstride
    for o in range(outer):
        for i in range(pstride):
            for piter in range(0, pdim, step):

                def short2long(idx):
                    return o * ostride + (piter + idx) * pstride + i

                if check:
                    zeros = 0
                    if is_raw_data:
                        for i in range(step):
                            if (
                                data[short2long(i) * 2] == 0
                                and data[short2long(i) * 2 + 1] == 0
                            ):
                                zeros += 1
                    else:
                        i32_data_0 = data[short2long(0)]

                        def bf16_zeros_in_int32(v):
                            bf16_data_0 = v & 0xFF
                            bf16_data_1 = (v >> 8) & 0xFF
                            v0_zero = 1 if bf16_data_0 == 0 else 0
                            v1_zero = 1 if bf16_data_1 == 0 else 0
                            return v0_zero + v1_zero

                        zeros = bf16_zeros_in_int32(i32_data_0) + bf16_zeros_in_int32(
                            i32_data_0
                        )
                    if zeros < 2:
                        G_LOGGER.warning(f"Found non-sparse tensor: {tensor.name}")
                        return False
                else:
                    if is_raw_data:
                        # data is 8bit array, bf16 is 16bit
                        # the index is doubled, and we need twice change for one bf16 value
                        data[short2long(1) * 2] = 0
                        data[short2long(1) * 2 + 1] = 0
                        data[short2long(2) * 2] = 0
                        data[short2long(2) * 2 + 1] = 0
                    else:
                        # data is 32bit array, bf16 is 16bit
                        # We use the index but only need to change one value
                        data[short2long(0)] = 0

    if check:
        G_LOGGER.info(f"Found sparse tensor: {tensor.name}")
        return True
    else:
        if is_raw_data:
            tensor.raw_data = bytes(data)
        return tensor


def process_tensor(pinfo, tensor, check):
    axis = pinfo.axis
    dims = tensor.dims
    pdim = tensor.dims[axis]

    # figure out the stride
    outer = 1
    pstride = 1
    for i in range(0, axis, 1):
        outer *= dims[i]
    for i in range(axis + 1, len(tensor.dims), 1):
        pstride *= dims[i]
    G_LOGGER.ultra_verbose(
        f"axis {axis} of dims {dims} has stride {pstride} and outer {outer}"
    )

    # We need hacks since BF16 has not been fully enabled in Numpy or ONNX.
    if tensor.data_type is onnx.TensorProto.BFLOAT16:
        return process_bf16_tensor(tensor, outer, pdim, pstride, check)

    # prune/check alongside the axis
    ostride = pdim * pstride
    data = np.array(onnx_numpy_helper.to_array(tensor)).reshape(util.volume(dims))
    for o in range(outer):
        for i in range(pstride):
            for piter in range(0, pdim, 4):

                def short2long(idx):
                    """Convert the short-index to the location in the buffer"""
                    return o * ostride + (piter + idx) * pstride + i

                short_idx = range(4)
                long_idx = [short2long(si) for si in short_idx]
                vals = [data[li] for li in long_idx]
                vals_abs = [abs(v) for v in vals]
                min0_vabs = min(vals_abs)
                min0_idx = vals_abs.index(min0_vabs)
                vals_abs[min0_idx] = sys.float_info.max
                min1_vabs = min(vals_abs)
                min1_idx = vals_abs.index(min1_vabs)

                if check:
                    if min0_vabs != 0 or min1_vabs != 0:
                        G_LOGGER.warning(f"Found non-sparse tensor: {tensor.name}")
                        return False
                else:
                    min0_idx = short2long(min0_idx)
                    min1_idx = short2long(min1_idx)
                    np.put(data, min0_idx, 0)
                    np.put(data, min1_idx, 0)

    if check:
        G_LOGGER.info(f"Found sparse tensor: {tensor.name}")
        return True
    else:
        # pack raw data pack and then push to the model
        data = data.reshape(dims)
        return onnx_numpy_helper.from_array(data, name=tensor.name)


def build_new_model(m, new_w_name2obj):
    if len(new_w_name2obj) == 0:
        G_LOGGER.verbose("No need to build new model object")
        return m

    G_LOGGER.info("Replacing weights to build new model object...")
    g = m.graph
    new_initializer = list()
    n = len(g.initializer)
    for i in range(n):
        t = g.initializer[i]
        G_LOGGER.extra_verbose(f"Processing {i}/{n} {t.name}")
        if t.name in new_w_name2obj:
            new_t = new_w_name2obj[t.name]
            new_initializer.append(new_t)
        else:
            new_initializer.append(t)

    new_g = onnx.helper.make_graph(
        nodes=g.node,
        name=g.name,
        inputs=g.input,
        outputs=g.output,
        initializer=new_initializer,
        doc_string=g.doc_string,
        value_info=g.value_info,
    )

    attrs = {
        "ir_version": m.ir_version,
        "producer_name": "polygraphy surgeon prune",
        "opset_imports": [m.opset_import[0]],
    }
    return onnx.helper.make_model(new_g, **attrs)
