#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from polygraphy.common import constants
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Inline, Script


class DataLoaderArgs(BaseArgs):
    def __init__(self, read=True):
        self._read = read


    def add_to_parser(self, parser):
        data_loader_args = parser.add_argument_group("Data Loader", "Options for modifying data used for inference")
        data_loader_args.add_argument("--seed", metavar="SEED", help="Seed to use for random inputs",
                                        type=int, default=None)
        data_loader_args.add_argument("--int-min", help="Minimum integer value for random integer inputs", type=int, default=None)
        data_loader_args.add_argument("--int-max", help="Maximum integer value for random integer inputs", type=int, default=None)
        data_loader_args.add_argument("--float-min", help="Minimum float value for random float inputs", type=float, default=None)
        data_loader_args.add_argument("--float-max", help="Maximum float value for random float inputs", type=float, default=None)
        data_loader_args.add_argument("--iterations", metavar="NUM", help="Number of inference iterations for which to supply data", type=int, default=None)
        if self._read:
            data_loader_args.add_argument("--load-input-data", "--load-inputs", help="[EXPERIMENTAL] Path(s) to load inputs. Each file should be a pickled "
                                          "List[Dict[str, numpy.ndarray]]. Other data loader options are ignored when this option is used", default=[],
                                          dest="load_inputs", nargs="+")


    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker


    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"


    def parse(self, args):
        def omit_none_tuple(tup):
            if all([elem is None for elem in tup]):
                return None
            return tup

        self.seed = tools_util.get(args, "seed")
        self.int_range = omit_none_tuple(tup=(tools_util.get(args, "int_min"), tools_util.get(args, "int_max")))
        self.float_range = omit_none_tuple(tup=(tools_util.get(args, "float_min"), tools_util.get(args, "float_max")))
        self.iterations = tools_util.get(args, "iterations")
        self.load_inputs = tools_util.get(args, "load_inputs")


    def add_to_script(self, script):
        def _make_data_loader(script):
            data_loader_name = Inline("data_loader")

            input_metadata_str = Inline(repr(self.model_args.input_shapes)) if self.model_args.input_shapes else None
            if input_metadata_str:
                script.add_import(imports=["TensorMetadata"], frm="polygraphy.common")

            data_loader = Script.invoke_if_nondefault("DataLoader", seed=self.seed, iterations=self.iterations,
                                                    input_metadata=input_metadata_str, int_range=self.int_range, float_range=self.float_range)
            if data_loader is not None:
                script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
                script.append_prefix(Script.format_str("\n# Inference Inputs Loader\n{:} = {:}", data_loader_name, Inline(data_loader)))
            else:
                data_loader_name = None
            return data_loader_name

        if self.load_inputs:
            script.add_import(imports=["misc"], frm="polygraphy.util")

            data_loader_name = Inline("data_loader")
            script.append_prefix(Script.format_str("# Load inputs\n{data_loader} = []\nfor input_data_path in {load_inputs}:"
                                                "\n{tab}{data_loader}.extend(misc.pickle_load(input_data_path))",
                                                data_loader=data_loader_name, load_inputs=self.load_inputs, tab=Inline(constants.TAB)))
        else:
            data_loader_name = _make_data_loader(script)
        script.append_prefix("") # Newline
        return data_loader_name


    def get_data_loader(self):
        script = Script()
        data_loader_name = self.add_to_script(script)
        if data_loader_name is None: # All arguments are default
            from polygraphy.comparator import DataLoader
            return DataLoader()
        exec(str(script), globals(), locals())
        return locals()[data_loader_name]
