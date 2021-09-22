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

import numbers

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import Script, make_invocable, make_invocable_if_nondefault, safe


@mod.export()
class DataLoaderArgs(BaseArgs):
    def __init__(self):
        super().__init__()
        self.model_args = None

    def add_to_parser(self, parser):
        data_loader_args = parser.add_argument_group(
            "Data Loader", "Options for controlling how input data is loaded or generated"
        )
        data_loader_args.add_argument(
            "--seed", metavar="SEED", help="Seed to use for random inputs", type=int, default=None
        )
        data_loader_args.add_argument(
            "--val-range",
            help="Range of values to generate in the data loader. "
            "To specify per-input ranges, use the format: --val-range <out_name>:[min,max]. "
            "If no input name is provided, the range is used for any inputs not explicitly specified. "
            "For example: --val-range [0,1] inp0:[2,50] inp1:[3.0,4.6]",
            nargs="+",
            default=None,
        )
        data_loader_args.add_argument(
            "--int-min",
            help="[DEPRECATED: Use --val-range] Minimum integer value for random integer inputs",
            type=int,
            default=None,
        )
        data_loader_args.add_argument(
            "--int-max",
            help="[DEPRECATED: Use --val-range] Maximum integer value for random integer inputs",
            type=int,
            default=None,
        )
        data_loader_args.add_argument(
            "--float-min",
            help="[DEPRECATED: Use --val-range] Minimum float value for random float inputs",
            type=float,
            default=None,
        )
        data_loader_args.add_argument(
            "--float-max",
            help="[DEPRECATED: Use --val-range] Maximum float value for random float inputs",
            type=float,
            default=None,
        )
        data_loader_args.add_argument(
            "--iterations",
            "--iters",
            metavar="NUM",
            help="Number of inference iterations for which to supply data",
            type=int,
            default=None,
            dest="iterations",
        )
        data_loader_args.add_argument(
            "--load-inputs",
            "--load-input-data",
            help="[EXPERIMENTAL] Path(s) to load inputs. The file(s) should be a JSON-ified "
            "List[Dict[str, numpy.ndarray]], i.e. a list where each element is the feed_dict for a single iteration. "
            "Other data loader options are ignored when this option is used",
            default=[],
            dest="load_inputs",
            nargs="+",
        )
        data_loader_args.add_argument(
            "--data-loader-script",
            help="Path to a Python script that defines a function that loads input data. "
            "The function should take no arguments and return a generator or iterable that yields input data (Dict[str, np.ndarray]). "
            "When this option is specified, all other data loader arguments are ignored. ",
            default=None,
        )
        data_loader_args.add_argument(
            "--data-loader-func-name",
            help="When using a data-loader-script, this specifies the name of the function "
            "that loads data. Defaults to `load_data`. ",
            default="load_data",
        )

    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker

    def parse(self, args):
        def omit_none_tuple(tup):
            if all([elem is None for elem in tup]):
                return None
            return tup

        self.seed = args_util.get(args, "seed")

        self.int_range = omit_none_tuple(tup=(args_util.get(args, "int_min"), args_util.get(args, "int_max")))
        self.float_range = omit_none_tuple(tup=(args_util.get(args, "float_min"), args_util.get(args, "float_max")))
        if self.int_range or self.float_range:
            G_LOGGER.warning(
                "The --int-min/--int-max and --float-min/--float-max options are deprecated.\n"
                "Please use `--val-range` instead, which allows you to specify per-input data ranges."
            )

        self.val_range = args_util.parse_dict_with_default(args_util.get(args, "val_range"), cast_to=tuple)
        if self.val_range is not None:
            for name, vals in self.val_range.items():
                if len(vals) != 2:
                    G_LOGGER.critical(
                        "In --val-range, for input: {:}, expected to receive exactly 2 values, but received {:}.\n"
                        "Note: Option was parsed as: input: {:}, range: {:}".format(name, len(vals), name, vals)
                    )

                if any(not isinstance(elem, numbers.Number) for elem in vals):
                    G_LOGGER.critical(
                        "In --val-range, for input: {:}, one or more elements of the range could not be parsed as a number.\n"
                        "Note: Option was parsed as: input: {:}, range: {:}".format(name, name, vals)
                    )

        self.iterations = args_util.get(args, "iterations")

        self.load_inputs = args_util.get(args, "load_inputs")
        self.data_loader_script = args_util.get(args, "data_loader_script")
        self.data_loader_func_name = args_util.get(args, "data_loader_func_name")

    def _add_to_script(self, script, user_input_metadata_str=None):
        needs_invoke = False
        using_random_data = False

        if self.data_loader_script:
            script.add_import(imports=["mod"], frm="polygraphy")
            data_loader = make_invocable(
                "mod.import_from_script", self.data_loader_script, name=self.data_loader_func_name
            )
            needs_invoke = True
        elif self.load_inputs:
            script.add_import(imports=["load_json"], frm="polygraphy.json")
            data_loader = safe(
                "[]\nfor input_data_path in {load_inputs}:"
                "\n\t{data_loader}.extend(load_json(input_data_path, description='input data'))",
                load_inputs=self.load_inputs,
                data_loader=Script.DATA_LOADER_NAME,
            )
        else:
            using_random_data = True
            if user_input_metadata_str is None and self.model_args is not None and self.model_args.input_shapes:
                user_input_metadata_str = self.model_args.input_shapes

            if user_input_metadata_str:
                script.add_import(imports=["TensorMetadata"], frm="polygraphy.common")

            data_loader = make_invocable_if_nondefault(
                "DataLoader",
                seed=self.seed,
                iterations=self.iterations,
                input_metadata=user_input_metadata_str,
                int_range=self.int_range,
                float_range=self.float_range,
                val_range=self.val_range,
            )
            if data_loader:
                script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")

        if using_random_data != self.is_using_random_data():
            G_LOGGER.internal_error("is_using_random_data() reported a false positive!")

        return script.set_data_loader(data_loader), needs_invoke

    def add_data_loader(self, script, *args, **kwargs):
        """
        Adds a DataLoader to the script.

        Args:
            user_input_metadata_str (str(TensorMetadata)):
                    The name of a variable containing TensorMetadata.
                    This will control the shape and data type of the generated
                    data.

        Returns:
            str: The data loader, as a string. This may either be the variable name,
                or an invocation of the data loader function.
        """
        data_loader, needs_invoke = self._add_to_script(script, *args, **kwargs)
        if needs_invoke:
            data_loader = make_invocable(data_loader)
        return data_loader

    def get_data_loader(self, user_input_metadata=None):
        from polygraphy.comparator import DataLoader

        needs_invoke = False

        # run_script expects the callable to return just the variable name, but self.add_to_script
        # has 2 return values. We wrap it here to create a function with the right signature.
        def add_to_script_wrapper(script, *args, **kwargs):
            nonlocal needs_invoke
            name, needs_invoke = self._add_to_script(script, *args, **kwargs)
            return name

        data_loader = util.default(args_util.run_script(add_to_script_wrapper, user_input_metadata), DataLoader())
        if needs_invoke:
            data_loader = data_loader()
        return data_loader

    def is_using_random_data(self):
        """
        Whether this data loader will randomly generate data rather than use real data.

        Returns:
            bool
        """
        return not self.data_loader_script and not self.load_inputs
