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
from collections import OrderedDict

import numpy as np
import torch
import pytest

from polygraphy import constants, util
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader
from polygraphy.comparator.data_loader import DataLoaderCache
from polygraphy.datatype import DataType
from tests.models.meta import ONNX_MODELS
from polygraphy.exception import PolygraphyException


def meta(dtype):
    return (
        TensorMetadata()
        .add("X", dtype=dtype, shape=(4, 4))
        .add("Y", dtype=dtype, shape=(5, 5))
    )


class TestDataLoader:
    @pytest.mark.parametrize("dtype", [np.int32, bool, np.float32, np.int64])
    def test_default_ranges(self, dtype):
        data_loader = DataLoader(input_metadata=meta(dtype))
        x, y = data_loader[0].values()
        assert np.all((x >= 0) & (x <= 1))
        assert np.all((y >= 0) & (y <= 1))

    def test_can_override_shape(self):
        model = ONNX_MODELS["dynamic_identity"]

        shape = (1, 1, 4, 5)
        custom_input_metadata = TensorMetadata().add("X", dtype=None, shape=shape)
        data_loader = DataLoader(input_metadata=custom_input_metadata)
        # Simulate what the comparator does
        data_loader.input_metadata = model.input_metadata

        feed_dict = data_loader[0]
        assert tuple(feed_dict["X"].shape) == shape

    @pytest.mark.parametrize(
        "min_shape, max_shape, expected",
        [
            # When both min/max are set, use min.
            ((2, 3, 2, 2), (4, 3, 2, 2), (2, 3, 2, 2)),
            # When only one of min/max are set, use whichever one is set.
            ((2, 3, 2, 2), None, (2, 3, 2, 2)),
            (None, (4, 3, 2, 2), (4, 3, 2, 2)),
            # When min/max are not set, override with the default shape value.
            (None, None, (constants.DEFAULT_SHAPE_VALUE, 3, 2, 2)),
        ],
    )
    def test_can_use_min_max_shape(self, min_shape, max_shape, expected):
        shape = (-1, 3, 2, 2)

        data_loader = DataLoader()
        data_loader.input_metadata = TensorMetadata().add(
            "X", dtype=np.float32, shape=shape, min_shape=min_shape, max_shape=max_shape
        )

        feed_dict = data_loader[0]
        assert tuple(feed_dict["X"].shape) == expected

    @pytest.mark.parametrize("dtype", [np.int32, bool, np.float32, np.int64])
    @pytest.mark.parametrize("range_val", [0, 1])
    def test_range_min_max_equal(self, dtype, range_val):
        data_loader = DataLoader(
            input_metadata=meta(dtype), val_range=(range_val, range_val)
        )
        feed_dict = data_loader[0]
        assert np.all(feed_dict["X"] == range_val)
        assert np.all(feed_dict["Y"] == range_val)

    @pytest.mark.parametrize(
        "range",
        [
            (0, 1, np.int32),
            (5.0, 5.5, np.float32),
            (0, 1, bool),
            (float("inf"), float("inf"), np.float32),
            (float("-inf"), float("inf"), np.float32),
            (0, float("inf"), np.float32),
            (float("-inf"), 0, np.float32),
        ],
    )
    def test_val_ranges(self, range):
        min_val, max_val, dtype = range
        data_loader = DataLoader(
            input_metadata=meta(dtype), val_range=(min_val, max_val)
        )
        feed_dict = data_loader[0]
        assert np.all((feed_dict["X"] >= min_val) & (feed_dict["X"] <= max_val))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
    def test_val_range_dict(self, dtype):
        val_range = {"X": (2, 5), "Y": (-1, 2)}
        data_loader = DataLoader(input_metadata=meta(dtype), val_range=val_range)
        feed_dict = data_loader[0]
        assert np.all((feed_dict["X"] >= 2) & (feed_dict["X"] <= 5))
        assert np.all((feed_dict["Y"] >= -1) & (feed_dict["Y"] <= 2))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
    def test_val_range_dict_default(self, dtype):
        val_range = {"": (6, 8), "Y": (-3, 4)}
        data_loader = DataLoader(input_metadata=meta(dtype), val_range=val_range)
        feed_dict = data_loader[0]
        assert np.all((feed_dict["X"] >= 6) & (feed_dict["X"] <= 8))
        assert np.all((feed_dict["Y"] >= -3) & (feed_dict["Y"] <= 4))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
    def test_val_range_dict_fallback(self, dtype):
        val_range = {"Y": (-3, 4)}
        data_loader = DataLoader(input_metadata=meta(dtype), val_range=val_range)
        feed_dict = data_loader[0]
        assert np.all((feed_dict["X"] >= 0) & (feed_dict["X"] <= 1))
        assert np.all((feed_dict["Y"] >= -3) & (feed_dict["Y"] <= 4))

    def test_shape_tensor_detected(self):
        INPUT_DATA = (1, 2, 3)
        input_meta = TensorMetadata().add("X", dtype=np.int32, shape=(3,))
        # This contains the shape values
        overriden_meta = TensorMetadata().add("X", dtype=np.int32, shape=INPUT_DATA)
        data_loader = DataLoader(input_metadata=overriden_meta)
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert np.all(feed_dict["X"] == INPUT_DATA)  # values become INPUT_DATA

    def test_no_shape_tensor_false_positive_negative_dims(self):
        INPUT_DATA = (-100, 2, 4)
        # This should NOT be detected as a shape tensor
        input_meta = TensorMetadata().add("X", dtype=np.int32, shape=(3,))
        overriden_meta = TensorMetadata().add("X", dtype=np.int32, shape=INPUT_DATA)
        data_loader = DataLoader(input_metadata=overriden_meta)
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert feed_dict["X"].shape == (
            3,
        )  # Shape IS (3, ), because this is NOT a shape tensor
        assert np.any(
            feed_dict["X"] != INPUT_DATA
        )  # Contents are not INPUT_DATA, since it's not treated as a shape value

    def test_no_shape_tensor_false_positive_float(self):
        INPUT_DATA = (-100, -50, 0)
        # Float cannot be a shape tensor
        input_meta = TensorMetadata().add("X", dtype=np.float32, shape=(3,))
        overriden_meta = TensorMetadata().add("X", dtype=np.float32, shape=INPUT_DATA)
        data_loader = DataLoader(input_metadata=overriden_meta)
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert feed_dict["X"].shape == (3,)  # Values are NOT (3, )
        assert np.any(feed_dict["X"] != INPUT_DATA)  # Values are NOT (3, )

    def test_non_user_provided_inputs_never_shape_tensors(self):
        # If the user didn't provide metadata, then the value can never be a shape tensor.
        input_meta = TensorMetadata().add("X", dtype=np.int32, shape=(3,))
        data_loader = DataLoader()
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert feed_dict["X"].shape == (3,)  # Treat as a normal tensor

    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    @pytest.mark.parametrize("data_loader_backend_module", ["torch", "numpy"])
    def test_generate_scalar(self, dtype, data_loader_backend_module):
        data_loader = DataLoader(
            input_metadata=TensorMetadata().add("input", dtype=dtype, shape=[]),
            data_loader_backend_module=data_loader_backend_module,
        )

        scalar = data_loader[0]["input"]
        assert isinstance(
            scalar,
            np.ndarray if data_loader_backend_module == "numpy" else torch.Tensor,
        )
        assert scalar.shape == tuple()

    def test_error_on_unsupported_numpy_type(self):
        input_meta = TensorMetadata().add("X", dtype=DataType.BFLOAT16, shape=(3,))
        data_loader = DataLoader()
        data_loader.input_metadata = input_meta

        with pytest.raises(
            PolygraphyException,
            match="Please use a custom data loader to provide inputs.",
        ):
            data_loader[0]

    def test_bf16_supported_torch(self):
        input_meta = TensorMetadata().add("X", dtype=DataType.BFLOAT16, shape=(3,))
        data_loader = DataLoader(data_loader_backend_module="torch")
        data_loader.input_metadata = input_meta

        assert util.array.is_torch(data_loader[0]["X"])
    
    @pytest.mark.parametrize("name, should_match", [
        ("inp_*",      [True for _ in range(12)]),
        ("inp_?",      [False, False, False, *[True for _ in range(9)]]),
        ("inp_[abc]",  [*[False for _ in range(6)], True, True, True, False, False, False]),
        ("inp_[!abc]", [False, False, False, True, True, True, False, False, False, True, True, True]),
    ])
    def test_input_name_with_wildcards(self, name, should_match):
        match_case = [
            "inp_foo", "inp_bar", "inp_123", "inp_1", "inp_s", "inp_k",
            "inp_a", "inp_b", "inp_c", "inp_d", "inp_e", "inp_f",
        ]
        input_meta = TensorMetadata().add(name, dtype=np.float32, shape=(2, 2, 3))
        data_loader = DataLoader(input_metadata=input_meta)
        data_loader.input_metadata = TensorMetadata()
        for case in match_case:
            data_loader.input_metadata.add(case, dtype=np.float32, shape=(-1, 2, 3))

        res = [data_loader[0][name].shape == (2, 2, 3) for name in data_loader[0]]
        assert res == should_match 

build_torch = lambda a, **kwargs: util.array.to_torch(np.array(a, **kwargs))


@pytest.mark.parametrize("array_type", [np.array, build_torch])
class TestDataLoaderCache:
    def test_can_cast_dtype(self, array_type):
        # Ensure that the data loader can only be used once
        def load_data():
            yield {"X": array_type(np.ones((1, 1), dtype=np.float32))}

        cache = DataLoaderCache(load_data())

        fp32_meta = TensorMetadata().add("X", dtype=DataType.FLOAT32, shape=(1, 1))
        cache.set_input_metadata(fp32_meta)
        feed_dict = cache[0]
        assert util.array.dtype(feed_dict["X"]) == DataType.FLOAT32

        fp64_meta = TensorMetadata().add("X", dtype=DataType.FLOAT64, shape=(1, 1))
        cache.set_input_metadata(fp64_meta)
        feed_dict = cache[0]
        assert util.array.dtype(feed_dict["X"]) == DataType.FLOAT64

    # If one input isn't in the cache, we shouldn't give up looking
    # for other inputs
    def test_will_not_give_up_on_first_cache_miss(self, array_type):
        SHAPE = (32, 32)

        DATA = [OrderedDict()]
        DATA[0]["X"] = array_type(np.zeros(SHAPE, dtype=np.int64))
        DATA[0]["Y"] = array_type(np.zeros(SHAPE, dtype=np.int64))

        cache = DataLoaderCache(DATA)
        cache.set_input_metadata(
            TensorMetadata()
            .add("X", DataType.INT64, shape=SHAPE)
            .add("Y", DataType.INT64, SHAPE)
        )

        # Populate the cache with bad X but good Y.
        # The data loader cache should fail to coerce X to the right shape and then reload it from the data loader.
        cache.cache[0] = OrderedDict()
        cache.cache[0]["X"] = array_type(np.ones((64, 64), dtype=np.int64))
        cache.cache[0]["Y"] = array_type(np.ones(SHAPE, dtype=np.int64))

        feed_dict = cache[0]
        # Cache cannot reuse X, so it'll reload - we'll get all 0s from the data loader
        assert util.array.all(feed_dict["X"] == 0)
        # Cache can reuse Y, even though it's after X, so we'll get ones from the cache
        assert util.array.all(feed_dict["Y"] == 1)

    # The cache should ignore extra data generated by the data loader
    def test_ignores_extra_data(self, array_type):
        SHAPE = (32, 32)

        DATA = [OrderedDict()]
        DATA[0]["X"] = array_type(np.zeros(SHAPE, dtype=np.int64))
        DATA[0]["Y"] = array_type(np.zeros(SHAPE, dtype=np.int64))

        cache = DataLoaderCache(DATA)

        cache.set_input_metadata(TensorMetadata().add("X", DataType.INT64, shape=SHAPE))

        feed_dict = cache[0]
        assert list(feed_dict.keys()) == ["X"]
        assert util.array.all(feed_dict["X"] == 0)
