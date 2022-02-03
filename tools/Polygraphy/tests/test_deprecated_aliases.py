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
import pytest

from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import run_polygraphy_inspect


@pytest.fixture(scope="session", params=["none", "basic", "attrs", "full"])
def run_inspect_model_deprecated(request):
    yield lambda additional_opts: run_polygraphy_inspect(
        ["model"] + ["--mode={:}".format(request.param)] + additional_opts
    )


class TestInspectModelDeprecated(object):
    def test_model_onnx(self, run_inspect_model_deprecated):
        run_inspect_model_deprecated([ONNX_MODELS["identity"].path])

    def test_model_trt_sanity(self, run_inspect_model_deprecated):
        run_inspect_model_deprecated([ONNX_MODELS["identity"].path, "--display-as=trt"])

    def test_model_tf_sanity(self, run_inspect_model_deprecated):
        pytest.importorskip("tensorflow")

        run_inspect_model_deprecated([TF_MODELS["identity"].path, "--model-type=frozen"])


class TestConstants(object):
    def test_config(self):
        from polygraphy import constants

        assert (constants.INTERNAL_CORRECTNESS_CHECKS, constants.AUTOINSTALL_DEPS)


class TestCompareFunc(object):
    def test_basic_compare_func(self):
        from polygraphy.comparator import CompareFunc

        CompareFunc.basic_compare_func(atol=1, rtol=1)
