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

import os
import tempfile

import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from tests.helper import get_file_size, is_file_non_empty
from tests.models.meta import ONNX_MODELS


@pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported for TRT 7.2 and older")
def test_timing_cache(poly_run):
    with tempfile.TemporaryDirectory() as dir:
        # Test with files that haven't already been created instead of using NamedTemporaryFile().
        total_cache = os.path.join(dir, "total.cache")
        identity_cache = os.path.join(dir, "identity.cache")

        poly_run([ONNX_MODELS["const_foldable"].path, "--trt", "--timing-cache", total_cache])
        assert is_file_non_empty(total_cache)
        const_foldable_cache_size = get_file_size(total_cache)

        poly_run([ONNX_MODELS["identity"].path, "--trt", "--timing-cache", identity_cache])
        identity_cache_size = get_file_size(identity_cache)

        poly_run([ONNX_MODELS["identity"].path, "--trt", "--timing-cache", total_cache])
        total_cache_size = get_file_size(total_cache)

        # The total cache should be larger than either of the individual caches.
        assert total_cache_size >= const_foldable_cache_size and total_cache_size >= identity_cache_size
        # The total cache should also be smaller than or equal to the sum of the individual caches since
        # header information should not be duplicated.
        assert total_cache_size <= (const_foldable_cache_size + identity_cache_size)


def test_logger_severity():
    assert G_LOGGER.severity == G_LOGGER.module_severity.get()
    with G_LOGGER.verbosity():
        assert G_LOGGER.severity == G_LOGGER.CRITICAL


def test_debug_diff_tactics(poly_debug):
    status = poly_debug(["diff-tactics"])
    assert "debug diff-tactics is deprecated and will be removed" in status.stdout + status.stderr
