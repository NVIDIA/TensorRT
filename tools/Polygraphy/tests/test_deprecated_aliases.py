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
import warnings

import numpy as np
import pytest

from polygraphy.comparator import (
    BaseCompareFunc,
    BasePostprocessFunc,
    CompareFunc,
    IndicesCompareFunc,
    IterationResult,
    PerceptualMetricsCompareFunc,
    PostprocessFunc,
    SimpleCompareFunc,
    TopKPostprocessFunc,
)


class TestConstants:
    def test_config(self):
        from polygraphy import constants

        assert (constants.INTERNAL_CORRECTNESS_CHECKS, constants.AUTOINSTALL_DEPS)


class TestCompareFuncStaticMethods:
    # The CompareFunc.* static methods are deprecated in favor of the functor classes (e.g.
    # SimpleCompareFunc), but must keep working - returning an instance of the corresponding
    # functor and emitting a DeprecationWarning - until they are removed.
    @pytest.mark.parametrize(
        "factory, functor_type",
        [
            (CompareFunc.simple, SimpleCompareFunc),
            (CompareFunc.indices, IndicesCompareFunc),
            (CompareFunc.perceptual_metrics, PerceptualMetricsCompareFunc),
        ],
    )
    def test_returns_functor_and_warns(self, factory, functor_type):
        with pytest.warns(DeprecationWarning, match=functor_type.__name__):
            compare_func = factory()
        assert isinstance(compare_func, functor_type)
        assert isinstance(compare_func, BaseCompareFunc)

    def test_arguments_are_forwarded(self):
        # The shim must pass constructor arguments through to the functor.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            compare_func = CompareFunc.indices(index_tolerance=2)
        assert compare_func.index_tolerance == 2

    def test_legacy_simple_still_compares(self):
        # A functional smoke test: the legacy entry point still performs a comparison.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            compare_func = CompareFunc.simple(atol=1.0)
        res0 = IterationResult(outputs={"out": np.array([0.0], dtype=np.float32)})
        res1 = IterationResult(outputs={"out": np.array([0.5], dtype=np.float32)})
        # |diff| (0.5) is within atol (1.0), so the outputs match.
        assert bool(compare_func(res0, res1)["out"])


class TestPostprocessFuncStaticMethods:
    # PostprocessFunc.top_k is deprecated in favor of the TopKPostprocessFunc functor, but must keep
    # working - returning a TopKPostprocessFunc and emitting a DeprecationWarning - until removed.
    def test_returns_functor_and_warns(self):
        with pytest.warns(DeprecationWarning, match="TopKPostprocessFunc"):
            postprocess_func = PostprocessFunc.top_k(k=3)
        assert isinstance(postprocess_func, TopKPostprocessFunc)
        assert isinstance(postprocess_func, BasePostprocessFunc)
