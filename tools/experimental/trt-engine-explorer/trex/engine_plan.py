#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

"""
TensorRT Engine Exploration API
"""


from collections import OrderedDict
from curses import meta
from typing import Dict, List, Tuple, BinaryIO
from copy import deepcopy
import numpy as np
import json
import pandas as pd
import ntpath
from .df_preprocessing import *
from .activations import *


class Layer:
    def __init__(self, raw_dict):
        self.raw_dict = raw_dict
        self.name = raw_dict['Name']
        try:
            self.type = raw_dict['ParameterType']
        except:
            self.type = raw_dict['LayerType']
        self.subtype = raw_dict['LayerType']
        self.inputs = [Activation(tensor) for tensor in raw_dict['Inputs']]
        self.outputs = [Activation(tensor) for tensor in raw_dict['Outputs']]
        self.outputs_size_bytes = np.sum([i.size_bytes for i in self.outputs])
        if self.inputs:
            self.precision = self.inputs[0].precision
            self.inputs_size_bytes = np.sum([i.size_bytes for i in self.inputs])
        else:
            self.inputs_size_bytes = 0
            self.precision = None

        self.total_io_size_bytes = self.inputs_size_bytes + self.outputs_size_bytes
        self._parse_weights()
        self.total_footprint_bytes = self.total_io_size_bytes + self.weights_size

    def _parse_constant(const):
        cnt = const['Count']
        data_type = const['Type']
        try:
            data_size = dict(
                {"Int8": 1, "Half": 2, "Float": 4, "Int32": 4})[data_type]
        except KeyError:
            # Backward compatbility.
            data_size = dict(
                {"Int8": 1, "FP16": 2, "FP32": 4, "Int32": 4})[data_type]
        return cnt, data_type, cnt * data_size

    def _parse_weights(self):
        try:
            self.weights_cnt, self.weights_type, self.weights_size = \
                Layer._parse_constant(self.raw_dict['Weights'])
        except KeyError:
            self.weights_cnt = 0
            self.weights_type = None
            self.weights_size = 0
        try:
            self.bias_cnt, self.bias_type, self.bias_size = \
                Layer._parse_constant(self.raw_dict['Bias'])
        except KeyError:
            self.bias_cnt = 0
            self.bias_type = None
            self.bias_size = 0

    def tooltip(self):
        tip = ""
        for key, value in sorted(self.raw_dict.items()):
            if key not in ['InputRegions', 'OutputRegions', 'Inputs', 'Outputs',
                    'ParameterType', 'LayerName']:
                tip += f"{key}:{value}\\n"
        return tip

    def __repr__(self):
        rep = f"Layer({self.name})"
        return rep

class EnginePlan:
    def __init__(self,
        graph_file: str,
        profiling_file: str=None,
        metadata_file: str=None
    ):
        def read_json(json_file: str) -> BinaryIO:
            try:
                data = json.load(json_file)
            except:
                raise ValueError(f"Could not load JSON file {json_file}")
            return data

        def read_metadata_file(metadata_file: str, device: int=0):
            with open(metadata_file) as json_file:
                metadata = read_json(json_file)
                return metadata[device]
            return None

        def read_profiling_file(profiling_file: str) -> List[Dict[str, any]]:
            perf = None
            with open(profiling_file) as json_file:
                perf = read_json(json_file)
                # Clean data (remove records with short size)
                perf = [rec for rec in perf if len(rec) == 4]
                return perf

        def read_graph_file(graph_file: str) -> List:
            with open(graph_file) as json_file:
                graph = read_json(json_file)
                layers = graph['Layers']
                try:
                    bindings = graph['Bindings']
                except KeyError:
                    # Older TRT didn't include bindings
                    bindings = list()
                return layers, bindings

        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

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

        def fold_no_ops(layers: List) -> List:
            """Remove layers of type No-Op"""

            def activation_consumers_dict(layers) -> Dict[Activation, List[Layer]]:
                """Return a dictionary of consumer-layers per activation tensor"""
                consumers = OrderedDict()
                for layer in layers:
                    for input in layer.inputs:
                        try:
                            consumers[input.name].append(layer.name)
                        except:
                            consumers[input.name] = [layer.name]
                return consumers

            def move_input(src: Layer, dst: Layer, i: int=0):
                dst.inputs[i] = src.inputs[i]

            def fold(no_op: Layer):
                try:
                    successors = activation_consumers[no_op.outputs[0].name]
                    for successor in successors:
                        move_input(src=no_op, dst=ret[successor])
                except KeyError:
                    pass

            ret = OrderedDict({layer.name: layer for layer in layers})
            activation_consumers = activation_consumers_dict(layers)

            for layer in layers:
                if layer.type == 'NoOp':
                    fold(layer)

            # Remove the No-Op layers from the final list.
            ret = [layer for layer in ret.values() if layer.type != 'NoOp']
            return ret

        def create_layers(self, raw_layers):
            layers = [Layer(raw_layer) for raw_layer in raw_layers]
            self.layers = fold_no_ops(layers)
            self.all_layers = deepcopy(self.layers)
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
            if raw_perf is not None:
                perf_df = pd.DataFrame.from_dict(raw_perf)
                perf_df.drop(columns=['name'], inplace=True)
                perf_df.rename(columns={
                    'percentage': 'latency.pct_time',
                    'averageMs': 'latency.avg_time',
                    'timeMs': 'latency.time',
                    }, inplace=True)
                df = graph_df.join(perf_df)
            else:
                print("Warning: profiling data was not provided.")
                df = graph_df
                df['latency.pct_time'] = [0] * len(df)
                df['latency.avg_time'] = [0] * len(df)
                df['latency.time'] = [0] * len(df)
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

        def get_device_properties(metadata_file) -> Tuple[Dict, int]:
            try:
                metadata = read_metadata_file(metadata_file)
                GB_1 = 1024 * 1024 * 1024
                mem_bus_bits = metadata['GLOBAL_MEMORY_BUS_WIDTH'] # bus width in bits
                mem_clk = metadata['MEMORY_CLOCK_RATE'] # effective DDR KHz
                DDR = 2 # bits per clock
                BYTE = 8 # bits
                mem_bandwidth = mem_bus_bits * mem_clk * DDR / BYTE

                return {
                    "Device Name": metadata['Name'],
                    "Compute Capability": metadata['ComputeCapability'],
                    "Cuda": metadata['CUDA_VERSION'],
                    "Total Memory (GB)": f"{metadata['TotalMemory'] / GB_1:.2f}",
                    "Memory bus width (bits)": mem_bus_bits,
                    "Memory clock (GHz)": f"{mem_clk / 1e6: .3f}",
                    "Peak memory bandwidth (GB/s)": f"{mem_bandwidth / 1e6: .1f}",
                    "Compute clock (GHz)": f"{metadata['CLOCK_RATE'] / 1e6: .2f}",
                    "SM count": metadata['MULTIPROCESSOR_COUNT'],
                }, mem_bandwidth
            except:
                return {}, 0

        self.name = path_leaf(graph_file)
        raw_layers, self.bindings = read_graph_file(graph_file)
        raw_layers = disambiguate_layer_names(raw_layers)
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
        self.metadata, self.mem_bandwidth = get_device_properties(metadata_file)
        assert self._df is not None, f"Failed parsing plan file {graph_file}"

    def summary_dict(self):
        """Create a dictionary of important attributes of the engine plan."""
        MB_1 = 1024 * 1024
        bindings = self.get_bindings()

        d = {
            "Inputs": f"{bindings[0]}",
            "Average time": f"{self.total_runtime:.3f} ms",
            "Layers": f"{len(self.df)}",
            "Weights": f"{self.total_weights_size / MB_1 :.1f} MB",
            "Activations": f"{self.total_act_size/ MB_1 :.1f} MB",
        }
        d.update(self.metadata)
        return d

    def print_summary(self):
        d = self.summary_dict()
        for k,v in d.items():
            print(f"{k}: {v}")

    def summary(self):
        return self.print_summary()

    def as_dataframe(self):
        return self._df

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
        for layer in self.layers:
            inputs += [inp for inp in layer.inputs if inp.name in self.bindings]
            outputs += [outp for outp in layer.outputs if outp.name in self.bindings]
        return inputs, outputs
