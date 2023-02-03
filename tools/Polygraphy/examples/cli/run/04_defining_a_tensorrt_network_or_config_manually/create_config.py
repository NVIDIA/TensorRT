#!/usr/bin/env python3
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
"""
Creates a TensorRT builder configuration and enables FP16 tactics.
"""
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig


# If we define a function called `load_config`, polygraphy can use it to
# create the builder configuration.
#
# TIP: If our function isn't called `load_config`, we can explicitly specify
# the name with the script argument, separated by a colon. For example: `create_config.py:my_func`.
@func.extend(CreateConfig())
def load_config(config):
    # NOTE: func.extend() causes the signature of this function to be `(builder, network) -> config`
    # For details on how this works, see examples/api/03_interoperating_with_tensorrt

    config.set_flag(trt.BuilderFlag.FP16)

    # Notice that we don't need to return anything - `extend()` takes care of that for us!
