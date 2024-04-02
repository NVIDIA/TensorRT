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

"""
JSON file parsing
"""


import re
import json
from typing import Dict, List, Tuple, BinaryIO


def read_json(json_file: str) -> BinaryIO:
    try:
        data = json.load(json_file)
    except:
        raise ValueError(f"Could not load JSON file {json_file}")
    return data


def read_graph_file(graph_file: str) -> List:
    err_msg = f"File {graph_file} does not conform to the expected JSON format."
    with open(graph_file) as json_file:
        graph = read_json(json_file)
        if not isinstance(graph, dict):
            raise ValueError(err_msg)
        layers = graph['Layers']
        try:
            bindings = graph['Bindings']
        except KeyError:
            # Older TRT didn't include bindings
            bindings = list()
        if not isinstance(layers, list):
            raise ValueError(err_msg)
        if not isinstance(layers[0], dict):
            details_msg = "\nMake sure to enable detailed ProfilingVerbosity when building the engine."
            details_msg += "\nSee https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#engine-inspector"
            raise ValueError(err_msg + details_msg)
        return layers, bindings


def read_profiling_file(profiling_file: str) -> List[Dict[str, any]]:
    perf = None
    with open(profiling_file) as json_file:
        perf = read_json(json_file)
        # Clean data (remove records with short size)
        perf = [rec for rec in perf if len(rec) in (4, 5)]
        return perf


def read_metadata_file(metadata_file: str, device: int=0):
    with open(metadata_file) as json_file:
        metadata = read_json(json_file)
        return metadata[device]


def read_timing_file(timing_json_file: str):
    with open(timing_json_file) as json_file:
        timing_recs = read_json(json_file)
        latencies_list = [rec['latencyMs'] for rec in timing_recs]
        return latencies_list


def read_perf_metadata_file(metadata_file: str, section: str):
    with open(metadata_file) as json_file:
        metadata = read_json(json_file)
        return metadata[section]


def get_device_properties(metadata_file: str) -> Dict:
    try:
        return read_perf_metadata_file(metadata_file, 'device_information')
    except (FileNotFoundError, TypeError):
        return {}


def get_performance_summary(metadata_file:str) -> Dict:
    try:
        return read_perf_metadata_file(metadata_file, 'performance_summary')
    except (FileNotFoundError, TypeError):
        return {}


def get_builder_config(metadata_file:str) -> Dict:
    try:
        d = read_perf_metadata_file(metadata_file, 'model_options')
        d.update(read_perf_metadata_file(metadata_file, 'build_options'))
        return d
    except (FileNotFoundError, TypeError):
        return {}


def import_graph_file(graph_file: str, profile_id: int=None):
    def filter_profiles(raw_layers: List[Dict], bindings: List[str], profile_id: int) -> List:
        """Use layers from one shape profile.

        A TRT engine may be built with one or more shape-profiles, and each of
        the profiles will have its own engine graph. The layers of the different
        profiles are distinguished from one another by the `[profile N]` suffix
        appended to layer and binding names.
        The first profile does not have this suffix.

        When `profile_id` is None or zero we use only the layers of the first profile.
        Otherwise we use only the layers from the requested profile.
        """
        def use_name(name: str) -> bool:
            name_belongs_to_some_profile = re.search(r"\[profile +[0-9]\]", name)
            if name_belongs_to_some_profile and not profile_id:
                # We detected that the engine was built with several shape-profiles
                # but we are not targeting any of them so move to the next layer.
                # The first profile always exists and never has a profile id in the names.
                return False
            if profile_id:
                # We detected that the engine was built with several shape-profiles
                # and we are targeting a specific profile.
                name_belongs_to_right_profile = re.search(rf"\[profile {profile_id}\]", name)
                if not name_belongs_to_right_profile:
                    return False
            return True

        filtered_raw_layers = []
        for raw_layer in raw_layers:
            name = raw_layer['Name']
            if use_name(name):
                filtered_raw_layers.append(raw_layer)
        filtered_bindings = [binding for binding in bindings if use_name(binding)]
        if not len(filtered_raw_layers) or not len(filtered_bindings):
            raise ValueError(
                f"Something went wrong went filtering layers from the provided "
                f"profile ({profile_id}).\nMost likely the profile data does not "
                "exist in the graph file so try without providing a profile id.")
        return filtered_raw_layers, filtered_bindings

    def disambiguate_layer_names(raw_layers: List) -> List:
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

    def convert_deconv(raw_layers: List) -> List:
        """Distinguish between convolution and convolution-transpose (deconvolution)"""
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

    def fix_metadata(raw_layers: List) -> List:
        """TensorRT 8.6 introduced the Metadata field, with a non-ASCII character
        that triggers an SVG rendering error. This function replaces this character.

        See: https://github.com/NVIDIA/TensorRT/issues/2779
        """
        TRT_METADATA_DELIM = '\x1E'
        for l in raw_layers:
            try:
                if TRT_METADATA_DELIM in l['Metadata']:
                    l['Metadata'] = l['Metadata'].replace(TRT_METADATA_DELIM, '+')
            except KeyError:
                pass
        return raw_layers

    def fix_unicode(raw_layers: List) -> List:
        """TensorRT 8.6 and 10.0 introduced non-ASCII characters to the graph JSON
        that trigger SVG rendering errors. This function replaces these characters.

        See: https://github.com/NVIDIA/TensorRT/issues/2779
        """
        UNICODE_UNIT_SEPARATOR = '\x1E'
        UNICODE_REC_SEPARATOR = '\x1F'
        TREX_SEPARATOR = '+'
        replace_unicode = lambda s: s.replace(UNICODE_UNIT_SEPARATOR, TREX_SEPARATOR
                                    ).replace(UNICODE_REC_SEPARATOR, TREX_SEPARATOR)
        for l in raw_layers:
            try:
                l['Name'] = replace_unicode(l['Name'])
                l['Metadata'] = replace_unicode(l['Metadata'])
            except KeyError:
                pass
        return raw_layers

    def __remove_signal_wait(raw_layers: List) -> List:
        for raw_layer in raw_layers:
            if raw_layer['LayerType'] in ("signal", "wait"):
                raw_layers.remove(raw_layer)
        return raw_layers


    raw_layers, bindings = read_graph_file(graph_file)
    raw_layers = fix_unicode(raw_layers)
    raw_layers = convert_deconv(raw_layers)
    raw_layers = __remove_signal_wait(raw_layers)
    raw_layers = disambiguate_layer_names(raw_layers)
    raw_layers, bindings = filter_profiles(raw_layers, bindings, profile_id)

    return raw_layers, bindings
