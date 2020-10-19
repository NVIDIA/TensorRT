#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from polygraphy.comparator import DataLoader
from polygraphy.comparator.data_loader import DataLoaderCache
from polygraphy.common import TensorMetadata
from polygraphy.util import misc
from polygraphy.logger import G_LOGGER

from tests.models.meta import ONNX_MODELS

from collections import OrderedDict
import numpy as np


class TestDataLoader(object):
    def test_can_override_shape(self):
        model = ONNX_MODELS["dynamic_identity"]

        shape = (1, 1, 4, 5)
        custom_input_metadata = TensorMetadata().add("X", dtype=None, shape=shape)
        data_loader = DataLoader(input_metadata=custom_input_metadata)
        # Simulate what the comparator does
        data_loader.input_metadata = model.input_metadata

        feed_dict = data_loader[0]
        assert tuple(feed_dict["X"].shape) == shape


    def test_range_min_max_equal(self):
        RANGE_VAL = 1
        data_loader = DataLoader(input_metadata=TensorMetadata().add("X", dtype=np.int32, shape=(1, 1)),
                                 int_range=(RANGE_VAL, RANGE_VAL))
        feed_dict = data_loader[0]
        assert np.all(feed_dict["X"] == RANGE_VAL)


    def test_shape_tensor_detected(self):
        INPUT_DATA = (1, 2, 3)
        input_meta = TensorMetadata().add("X", dtype=np.int32, shape=(3, ))
        # This contains the shape values
        overriden_meta = TensorMetadata().add("X", dtype=np.int32, shape=INPUT_DATA)
        data_loader = DataLoader(input_metadata=overriden_meta)
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert np.all(feed_dict["X"] == INPUT_DATA) # values become INPUT_DATA


    def test_no_shape_tensor_false_positive_negative_dims(self):
        INPUT_DATA = (-100, 2, 4)
        # This should NOT be detected as a shape tensor
        input_meta = TensorMetadata().add("X", dtype=np.int32, shape=(3, ))
        overriden_meta = TensorMetadata().add("X", dtype=np.int32, shape=INPUT_DATA)
        data_loader = DataLoader(input_metadata=overriden_meta)
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert feed_dict["X"].shape == (3, ) # Shape IS (3, ), because this is NOT a shape tensor
        assert np.any(feed_dict["X"] != INPUT_DATA) # Contents are not INPUT_DATA, since it's not treated as a shape value


    def test_no_shape_tensor_false_positive_float(self):
        INPUT_DATA = (-100, -50, 0)
        # Float cannot be a shape tensor
        input_meta = TensorMetadata().add("X", dtype=np.float32, shape=(3, ))
        overriden_meta = TensorMetadata().add("X", dtype=np.float32, shape=INPUT_DATA)
        data_loader = DataLoader(input_metadata=overriden_meta)
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert feed_dict["X"].shape == (3, ) # Values are NOT (3, )
        assert np.any(feed_dict["X"] != INPUT_DATA) # Values are NOT (3, )


    def test_non_user_provided_inputs_never_shape_tensors(self):
        # If the user didn't provide metadata, then the value can never be a shape tensor.
        input_meta = TensorMetadata().add("X", dtype=np.int32, shape=(3, ))
        data_loader = DataLoader()
        data_loader.input_metadata = input_meta

        feed_dict = data_loader[0]
        assert feed_dict["X"].shape == (3, ) # Treat as a normal tensor


class TestDataLoaderCache(object):
    def test_can_cast_dtype(self):
        # Ensure that the data loader can only be used once
        def load_data():
            yield {"X": np.ones((1, 1), dtype=np.float32)}
        cache = DataLoaderCache(load_data())

        fp32_meta = TensorMetadata().add("X", dtype=np.float32, shape=(1, 1))
        cache.set_input_metadata(fp32_meta)
        feed_dict = cache[0]
        assert feed_dict["X"].dtype == np.float32

        fp64_meta = TensorMetadata().add("X", dtype=np.float64, shape=(1, 1))
        cache.set_input_metadata(fp64_meta)
        feed_dict = cache[0]
        assert feed_dict["X"].dtype == np.float64


    # If one input isn't in the cache, we shouldn't give up looking
    # for other inputs
    def test_will_not_give_up_on_first_cache_miss(self):
        SHAPE = (32, 32)

        DATA = [OrderedDict()]
        DATA[0]["X"] = np.zeros(SHAPE, dtype=np.int64)
        DATA[0]["Y"] = np.zeros(SHAPE, dtype=np.int64)

        cache = DataLoaderCache(DATA)
        cache.set_input_metadata(TensorMetadata().add("X", np.int64, shape=SHAPE).add("Y", np.int64, SHAPE))

        # Populate the cache with bad X but good Y
        cache.cache[0] = OrderedDict()
        cache.cache[0]["X"] = np.ones((64, 64), dtype=np.int64)
        cache.cache[0]["Y"] = np.ones(SHAPE, dtype=np.int64)

        feed_dict = cache[0]
        # Cache cannot reuse X, so it'll reload - we'll get all 0s from the data loader
        assert np.all(feed_dict["X"] == 0)
        # Cache can reuse Y, even though it's after X, so we'll get ones from the cache
        assert np.all(feed_dict["Y"] == 1)
