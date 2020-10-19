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
from polygraphy.backend.tf import TfRunner, SessionFromGraph

from tests.models.meta import TF_MODELS
from tests.common import check_file_non_empty

import tempfile
import pytest
import os


class TestTfRunner(object):
    def test_can_name_runner(self):
        NAME = "runner"
        runner = TfRunner(None, name=NAME)
        assert runner.name == NAME


    def test_basic(self):
        model = TF_MODELS["identity"]
        with TfRunner(SessionFromGraph(model.loader)) as runner:
            assert runner.is_active
            model.check_runner(runner)
        assert not runner.is_active


    @pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
    def test_save_timeline(self):
        model = TF_MODELS["identity"]
        with tempfile.NamedTemporaryFile() as outpath:
            with TfRunner(SessionFromGraph(model.loader), allow_growth=True, save_timeline=outpath.name) as runner:
                model.check_runner(runner)
                check_file_non_empty(outpath.name)
