#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


class TestCuda(object):
    def test_cuda(self):
        from polygraphy.common import cuda

        assert cuda.DeviceArray


class TestFunc(object):
    def test_func(self):
        from polygraphy.common import func

        assert hasattr(func, "extend")


class TestException(object):
    def test_exception(self):
        from polygraphy.common import exception

        assert hasattr(exception, "PolygraphyException")


class TestConstants(object):
    def test_constants(self):
        from polygraphy.common import constants

        assert constants.MARK_ALL

    def test_config(self):
        from polygraphy import constants

        assert (constants.INTERNAL_CORRECTNESS_CHECKS, constants.AUTOINSTALL_DEPS)


class TestUtilJson(object):
    def test_json(self):
        from polygraphy.util import Decoder, Encoder, from_json, load_json, save_json, to_json


class TestCompareFunc(object):
    def test_basic_compare_func(self):
        from polygraphy.comparator import CompareFunc

        CompareFunc.basic_compare_func(atol=1, rtol=1)
