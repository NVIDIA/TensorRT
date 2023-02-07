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
import numpy as np
import pytest
from polygraphy import util
from polygraphy.backend.tf import SessionFromGraph, TfRunner
from polygraphy.exception import PolygraphyException
from tests.helper import is_file_non_empty
from tests.models.meta import TF_MODELS

pytest.importorskip("tensorflow")


class TestTfRunner:
    def test_can_name_runner(self):
        NAME = "runner"
        runner = TfRunner(None, name=NAME)
        assert runner.name == NAME

    def test_basic(self):
        model = TF_MODELS["identity"]
        with TfRunner(SessionFromGraph(model.loader)) as runner:
            assert runner.is_active
            model.check_runner(runner)
            assert runner.last_inference_time() is not None
        assert not runner.is_active

    @pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
    def test_save_timeline(self):
        model = TF_MODELS["identity"]
        with util.NamedTemporaryFile() as outpath:
            with TfRunner(SessionFromGraph(model.loader), allow_growth=True, save_timeline=outpath.name) as runner:
                model.check_runner(runner)
                assert is_file_non_empty(outpath.name)

    @pytest.mark.parametrize(
        "names, err",
        [
            (["fake-input", "Input:0"], "Extra inputs in"),
            (["fake-input"], "The following inputs were not found"),
            ([], "The following inputs were not found"),
        ],
    )
    def test_error_on_wrong_name_feed_dict(self, names, err):
        model = TF_MODELS["identity"]
        with TfRunner(SessionFromGraph(model.loader)) as runner:
            with pytest.raises(PolygraphyException, match=err):
                runner.infer({name: np.ones(shape=(1, 15, 25, 30), dtype=np.float32) for name in names})

    def test_error_on_wrong_dtype_feed_dict(self):
        model = TF_MODELS["identity"]
        with TfRunner(SessionFromGraph(model.loader)) as runner:
            with pytest.raises(PolygraphyException, match="unexpected dtype."):
                runner.infer({"Input:0": np.ones(shape=(1, 15, 25, 30), dtype=np.int32)})

    def test_error_on_wrong_shape_feed_dict(self):
        model = TF_MODELS["identity"]
        with TfRunner(SessionFromGraph(model.loader)) as runner:
            with pytest.raises(PolygraphyException, match="incompatible shape."):
                runner.infer({"Input:0": np.ones(shape=(1, 1, 25, 30), dtype=np.float32)})
