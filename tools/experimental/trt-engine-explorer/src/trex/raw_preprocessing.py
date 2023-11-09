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


from typing import Dict, List, Tuple
from .parser import *


def __disambiguate_layer_names(raw_layers: List) -> List:
    """If a layer name appears twice we need to disabmiguate it"""
    names_cnt = {}
    for raw_layer in raw_layers:
        name = raw_layer['Name']
        if name in names_cnt:
            names_cnt[name] += 1
            name += "_" + str(names_cnt[name])
            raw_layer['Name'] = name
        else:
            names_cnt[name] = 1
    return raw_layers


def __convert_deconv(raw_layers: List) -> List:
    for raw_layer in raw_layers:
        try:
            is_deconv = (
                raw_layer['ParameterType'] == "Convolution" and
                raw_layer['LayerType'] == "CaskDeconvolutionV2")
            if is_deconv:
                raw_layer['ParameterType'] = "Deconvolution"
        except KeyError:
            pass
    return raw_layers


def import_graph_file(graph_file: str):
    raw_layers, bindings = read_graph_file(graph_file)
    raw_layers = __convert_deconv(raw_layers)
    raw_layers = __disambiguate_layer_names(raw_layers)
    return raw_layers, bindings