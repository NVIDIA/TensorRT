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

import numpy as np
import onnx
import pytest

from polygraphy.tools.sparse import PruneInfo, process_tensor

BF16_TWO = 0x4000  # 2.0 in BF16. Its low byte is zero, but the value is not.


def bf16_bits(values):
    """Returns the BF16 bit patterns of the given float values as uint16."""
    as_uint32 = np.asarray(values, dtype=np.float32).view(np.uint32)
    return (as_uint32 >> 16).astype(np.uint16)


def make_bf16_tensor(values, dims, use_int32_data):
    bits = bf16_bits(values).reshape(-1)
    tensor = onnx.TensorProto()
    tensor.name = "w"
    tensor.data_type = onnx.TensorProto.BFLOAT16
    tensor.dims.extend(dims)
    if use_int32_data:
        # ONNX stores one BF16 value per int32_data entry, in the low 16 bits.
        tensor.int32_data.extend(int(b) for b in bits)
    else:
        tensor.raw_data = bits.astype("<u2").tobytes()
    return tensor


def read_bf16_bits(tensor):
    if len(tensor.int32_data):
        return np.array(tensor.int32_data, dtype=np.int64) & 0xFFFF
    return np.frombuffer(tensor.raw_data, dtype="<u2").astype(np.int64)


def make_2_4_sparse(dims, axis):
    """Nonzero values, except the first two of every group of 4 along `axis`."""
    values = np.arange(1, np.prod(dims) + 1, dtype=np.float32).reshape(dims)
    index = [slice(None)] * len(dims)
    for group_start in range(0, dims[axis], 4):
        for offset in (0, 1):
            index[axis] = group_start + offset
            values[tuple(index)] = 0
    return values


STORAGE = pytest.mark.parametrize(
    "use_int32_data", [False, True], ids=["raw_data", "int32_data"]
)
# (dims, axis): axis 0 with a stride > 1, and the last axis where the stride is 1.
LAYOUTS = pytest.mark.parametrize(
    "dims, axis", [([8, 3], 0), ([3, 8], 1)], ids=["pstride3", "pstride1"]
)


class TestBF16SparsityCheck:
    @STORAGE
    @LAYOUTS
    def test_sparse_tensor_is_sparse(self, dims, axis, use_int32_data):
        tensor = make_bf16_tensor(make_2_4_sparse(dims, axis), dims, use_int32_data)
        assert process_tensor(PruneInfo("w", axis), tensor, check=True) is True

    @STORAGE
    @LAYOUTS
    def test_dense_tensor_is_not_sparse(self, dims, axis, use_int32_data):
        values = np.full(dims, 2.0, dtype=np.float32)
        tensor = make_bf16_tensor(values, dims, use_int32_data)
        assert bf16_bits(2.0) == BF16_TWO
        assert process_tensor(PruneInfo("w", axis), tensor, check=True) is False

    @STORAGE
    @LAYOUTS
    def test_one_zero_per_group_is_not_sparse(self, dims, axis, use_int32_data):
        values = make_2_4_sparse(dims, axis)
        # Restore one of the two zeros in the last group only.
        index = [slice(None)] * len(dims)
        index[axis] = dims[axis] - 4
        values[tuple(index)] = 1.0
        tensor = make_bf16_tensor(values, dims, use_int32_data)
        assert process_tensor(PruneInfo("w", axis), tensor, check=True) is False


class TestBF16SparsityPrune:
    @STORAGE
    @LAYOUTS
    def test_pruned_tensor_is_2_4_sparse(self, dims, axis, use_int32_data):
        values = np.full(dims, 2.0, dtype=np.float32)
        tensor = make_bf16_tensor(values, dims, use_int32_data)
        pruned = process_tensor(PruneInfo("w", axis), tensor, check=False)

        bits = read_bf16_bits(pruned).reshape(dims)
        groups = np.moveaxis(bits, axis, -1).reshape(-1, 4)
        assert np.all(np.count_nonzero(groups == 0, axis=1) == 2)
        assert process_tensor(PruneInfo("w", axis), pruned, check=True) is True

    @LAYOUTS
    def test_int32_data_prune_zeroes_alternate_values(self, dims, axis):
        values = np.full(dims, 2.0, dtype=np.float32)
        tensor = make_bf16_tensor(values, dims, use_int32_data=True)
        pruned = process_tensor(PruneInfo("w", axis), tensor, check=False)

        bits = np.moveaxis(read_bf16_bits(pruned).reshape(dims), axis, -1)
        assert np.all(bits[..., 0::2] == 0)
        assert np.all(bits[..., 1::2] == BF16_TWO)
