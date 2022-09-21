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

import pytest
from polygraphy.tools.args import ModelArgs, TfLoadArgs
from tests.models.meta import TF_MODELS
from tests.tools.args.helper import ArgGroupTestHelper

tf = pytest.importorskip("tensorflow")


class TestTfLoaderArgs:
    def test_load_graph(self):
        arg_group = ArgGroupTestHelper(TfLoadArgs(), deps=[ModelArgs()])
        arg_group.parse_args([TF_MODELS["identity"].path, "--model-type=frozen"])
        graph, outputs = arg_group.load_graph()

        assert isinstance(graph, tf.Graph)
        assert outputs == ["Identity_2:0"]
