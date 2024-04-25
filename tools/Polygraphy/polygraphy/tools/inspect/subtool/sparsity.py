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
from polygraphy.tools.args import ModelArgs, OnnxLoadArgs
from polygraphy.tools.base import Tool
from polygraphy.tools.sparse import SparsityPruner


class Sparsity(Tool):
    """
    [EXPERIMENTAL] Display information about whether each weight tensor in an ONNX model follows a 2:4 structured sparsity pattern.
    """

    def __init__(self):
        super().__init__("sparsity")

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
        ]

    def run_impl(self, args):
        model = self.arg_groups[OnnxLoadArgs].load_onnx()
        pruner = SparsityPruner(model)
        pruner.check()
