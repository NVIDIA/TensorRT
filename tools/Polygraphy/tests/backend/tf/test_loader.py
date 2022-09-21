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
from polygraphy import constants, util
from polygraphy.backend.tf import GraphFromFrozen, ModifyGraphOutputs, SaveGraph, graph_from_frozen
from polygraphy.logger import G_LOGGER
from tests.helper import is_file_non_empty
from tests.models.meta import TF_MODELS

tf = pytest.importorskip("tensorflow")


class TestLoggerCallbacks:
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.module_severity = sev


class TestFrozenGraphLoader:
    def test_load_graph(self):
        with tf.compat.v1.Graph().as_default() as graph:
            inp = tf.placeholder(shape=(1, 1, 1, 1), dtype=tf.float32)
            out = tf.identity(inp)

        graph, outputs = graph_from_frozen(graph)
        assert graph
        assert outputs

    def test_load_pb(self):
        tf_loader = GraphFromFrozen(TF_MODELS["identity"].path)
        tf_loader()


class TestModifyGraph:
    def test_layerwise(self):
        load_frozen = GraphFromFrozen(TF_MODELS["identity"].path)
        modify_tf = ModifyGraphOutputs(load_frozen, outputs=constants.MARK_ALL)

        graph, outputs = modify_tf()
        assert graph
        assert outputs


class TestSaveGraph:
    def test_save_pb(self):
        with util.NamedTemporaryFile() as outpath:
            tf_loader = SaveGraph(GraphFromFrozen(TF_MODELS["identity"].path), path=outpath.name)
            tf_loader()
            assert is_file_non_empty(outpath.name)

    def test_save_tensorboard(self):
        with tempfile.TemporaryDirectory() as outdir:
            tf_loader = SaveGraph(GraphFromFrozen(TF_MODELS["identity"].path), tensorboard_dir=outdir)
            tf_loader()
            assert os.path.exists(tf_loader.tensorboard_dir)
