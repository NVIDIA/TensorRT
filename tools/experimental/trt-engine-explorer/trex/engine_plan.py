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
TensorRT Engine Exploration API - EnginePlan
"""


import warnings
from typing import List, Tuple
from copy import deepcopy
import pandas as pd
import ntpath
from .df_preprocessing import *
from .layer import Layer, fold_no_ops
from .parser import *


class EnginePlan:
    def __init__(self,
        graph_file: str,
        profiling_file: str=None,
        profiling_metadata_file: str=None,
        build_metadata_file: str=None,
        name: str=None,
        profile_id: int=None,
    ):
        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

        def create_layers(self, raw_layers):
            layers = [Layer(raw_layer) for raw_layer in raw_layers]
            self.layers = fold_no_ops(layers, self.bindings)
            self.all_layers = deepcopy(self.layers)
            self.constants = [layer for layer in self.layers if layer.type == 'Constant']
            self.layers = [layer for layer in self.layers if layer.type != 'Constant']
            return raw_layers

        def process_profiling_file(profiling_file, ignore_layers):
            if not profiling_file:
                return None
            raw_perf = read_profiling_file(profiling_file)
            raw_perf = [perf_rec for perf_rec in raw_perf if
                perf_rec['name'] not in ignore_layers]
            return raw_perf

        def merge_profiling_data(graph_df, raw_perf):
            def add_zero_perf(graph_df):
                df = graph_df
                df['latency.pct_time'] = [0] * len(df)
                df['latency.avg_time'] = [0] * len(df)
                df['latency.median_time'] = [0] * len(df)
                df['latency.time'] = [0] * len(df)
                return df

            if raw_perf is not None:
                perf_df = pd.DataFrame.from_dict(raw_perf)
                perf_df.rename(columns={
                    'percentage': 'latency.pct_time',
                    'averageMs': 'latency.avg_time',
                    'medianMs': 'latency.median_time',
                    'timeMs': 'latency.time',
                    }, inplace=True)
                if len(graph_df) == len(perf_df):
                    perf_df.drop(columns=['name'], inplace=True)
                    df = graph_df.join(perf_df)
                else:
                    warnings.warn(
                        "Partial profiling data: The number of layers in the engine "
                        f"graph ({len(graph_df)}) does not match the number of layers "
                        f"({len(perf_df)}) in the performance JSON.\n"
                        "This can happen if you're not using the first shape-profile.")
                    perf_df.rename(columns={'name': 'Name',}, inplace=True)
                    df = graph_df.merge(perf_df, on='Name', how='left').fillna(0)
            else:
                warnings.warn("Profiling data was not provided.")
                df = add_zero_perf(graph_df)
            return df

        def add_graph_summation_cols(df, layers):
            # Add new (summation) columns
            df['total_io_size_bytes'] = [l.total_io_size_bytes for l in layers]
            df['weights_size'] = [l.weights_size for l in layers]
            df['total_footprint_bytes'] = [l.total_footprint_bytes for l in layers]
            df['precision'] = [l.precision for l in layers]
            return df

        def construct_df(raw_layers):
            raw_layers = [raw_layer for raw_layer in raw_layers if
                raw_layer['LayerType'] not in ['Constant', 'NoOp']]
            graph_df = pd.DataFrame.from_dict(raw_layers)
            graph_df = fix_df(graph_df)
            return graph_df

        def compute_summary(self):
            self.total_act_size = sum(
                [l.total_io_size_bytes for l in self.layers])
            self.total_weights_size = sum(
                [l.weights_size for l in self.layers])
            assert self.total_weights_size == self.df['weights_size'].sum()
            self.total_runtime = sum(
                [avg_time for avg_time in self._df["latency.avg_time"]])

        self.name = name or path_leaf(graph_file)
        raw_layers, self.bindings = import_graph_file(graph_file, profile_id)
        raw_layers = create_layers(self, raw_layers)

        self._df = None
        ignore_layers = [raw_layer['Name'] for raw_layer in raw_layers if
            raw_layer['LayerType'] in ["Constant", "NoOp"]]
        self._raw_perf = process_profiling_file(
            profiling_file, ignore_layers=ignore_layers)
        graph_df = construct_df(raw_layers)
        graph_df = add_graph_summation_cols(graph_df, self.layers)
        self._df = merge_profiling_data(graph_df, self._raw_perf)
        compute_summary(self)
        self.device_properties = get_device_properties(profiling_metadata_file)
        self.performance_summary = get_performance_summary(profiling_metadata_file)
        self.builder_cfg = get_builder_config(build_metadata_file)
        assert self._df is not None, f"Failed parsing plan file {graph_file}"

    @property
    def df(self):
        return self._df

    def get_layers_by_type(self, layer_type):
        return filter_by_layer(self._df, layer_type)

    def find(self, layer_name: str):
        for l in self.layers:
            if layer_name == l.name: return l
        return None

    def get_bindings(self) -> Tuple[List[Activation], List[Activation]]:
        """Return a list of the inputs bindings and a list of the output bindings"""
        inputs, outputs = [], []
        processed_names = []
        for layer in self.layers:
            # BUG HERE: inputs and outputs are counted mutiple times.
            inputs += [inp for inp in layer.inputs if (inp.name in self.bindings and inp.name not in processed_names)]
            processed_names += [inp.name for inp in layer.inputs if inp.name not in processed_names]
            outputs += [outp for outp in layer.outputs if (outp.name in self.bindings and outp.name not in processed_names)]
            processed_names += [outp.name for outp in layer.outputs if outp.name not in processed_names]
        return list(set(inputs)), list(set(outputs))

    def summary(self):
        return print_summary(self)


def summary_dict(plan: EnginePlan):
    """Create a dictionary of important attributes of the engine plan."""
    MB_1 = 1024 * 1024
    bindings = plan.get_bindings()
    nl = "\n\t\t"
    d = {
        "Inputs": f"{nl.join([str(binding) for binding in bindings[0]])}",
        "Outputs": f"{nl.join([str(binding) for binding in bindings[1]])}",
        "Average time": f"{plan.total_runtime:.3f} ms",
        "Layers": f"{len(plan.df)}",
        "Weights": f"{plan.total_weights_size / MB_1 :.1f} MB",
        "Activations": f"{plan.total_act_size/ MB_1 :.1f} MB",
    }
    return d


def print_summary(plan: EnginePlan):
    def print_dict(d: Dict):
        for k,v in d.items():
            print(f"\t{k}: {v}")
    print("Model:")
    print_dict(summary_dict(plan))
    print("Device Properties:")
    print_dict(plan.device_properties)
    print("Builder Configuration:")
    print_dict(plan.builder_cfg)
    print("Performance Summary:")
    print_dict(plan.performance_summary)
