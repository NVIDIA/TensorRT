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
from polygraphy.tools.args import ModelArgs, OnnxLoadArgs, OnnxSaveArgs
from polygraphy.tools.sparse import SparsityPruner
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool


class Prune(BaseSurgeonSubtool):
    """
    [EXPERIMENTAL] Prune the weights of a model to follow a 2:4 structured sparsity pattern without regard for accuracy.
    For every four weight values, two will be set to zero.

    **NOTE:** This tool is meant to help functionally test sparsity.
    It will almost certainly cause significant accuracy degradation and so should NOT be used outside of functional testing.
    """

    def __init__(self):
        super().__init__("prune")

    def show_start_end_logging_impl(self, args):
        return True

    def get_subscriptions_impl(self):
        return [
            ModelArgs(
                model_opt_required=True,
                input_shapes_opt_name=False,
                required_model_type="onnx",
            ),
            OnnxLoadArgs(
                allow_shape_inference=False,
                outputs_opt_prefix=False,
                allow_from_tf=False,
            ),
            OnnxSaveArgs(allow_shape_inference=False, output_opt_required=True),
        ]

    def run_impl_surgeon(self, args):
        model = super().load_model()

        pruner = SparsityPruner(model)
        new_model = pruner.prune()

        super().save_model(new_model)
