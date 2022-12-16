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

import numbers

from polygraphy import constants, mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.script import Script, inline, make_invocable, make_invocable_if_nondefault, safe


@mod.export()
class DataLoaderArgs(BaseArgs):
    """
    Data Loader: loading or generating input data for inference.

    Depends on:

        - ModelArgs: if allow_custom_input_shapes == True
    """

    def __init__(self, allow_custom_input_shapes: bool = None):
        """
        Args:
            allow_custom_input_shapes (bool):
                    Whether to allow custom input shapes when randomly generating data.
                    Defaults to True.
        """
        super().__init__()
        self._allow_custom_input_shapes = util.default(allow_custom_input_shapes, True)

    def add_parser_args_impl(self):
        self.group.add_argument("--seed", metavar="SEED", help="Seed to use for random inputs", type=int, default=None)
        self.group.add_argument(
            "--val-range",
            help="Range of values to generate in the data loader. "
            "To specify per-input ranges, use the format: --val-range <input_name>:[min,max]. "
            "If no input name is provided, the range is used for any inputs not explicitly specified. "
            "For example: --val-range [0,1] inp0:[2,50] inp1:[3.0,4.6]",
            nargs="+",
            default=None,
        )
        self.group.add_argument(
            "--int-min",
            help="[DEPRECATED: Use --val-range] Minimum integer value for random integer inputs",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--int-max",
            help="[DEPRECATED: Use --val-range] Maximum integer value for random integer inputs",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--float-min",
            help="[DEPRECATED: Use --val-range] Minimum float value for random float inputs",
            type=float,
            default=None,
        )
        self.group.add_argument(
            "--float-max",
            help="[DEPRECATED: Use --val-range] Maximum float value for random float inputs",
            type=float,
            default=None,
        )
        self.group.add_argument(
            "--iterations",
            "--iters",
            metavar="NUM",
            help="Number of inference iterations for which the default data loader should supply data",
            type=int,
            default=None,
            dest="iterations",
        )

        custom_loader_group = self.group.add_mutually_exclusive_group()
        custom_loader_group.add_argument(
            "--load-inputs",
            "--load-input-data",
            help="Path(s) to load inputs. The file(s) should be a JSON-ified "
            "List[Dict[str, numpy.ndarray]], i.e. a list where each element is the feed_dict for a single iteration. "
            "When this option is used, all other data loader arguments are ignored. ",
            default=[],
            dest="load_inputs_paths",
            nargs="+",
        )
        custom_loader_group.add_argument(
            "--data-loader-script",
            help="Path to a Python script that defines a function that loads input data. "
            "The function should take no arguments and return a generator or iterable that yields input data (Dict[str, np.ndarray]). "
            "When this option is used, all other data loader arguments are ignored. "
            "By default, Polygraphy looks for a function called `load_data`. You can specify a custom function name "
            "by separating it with a colon. For example: `my_custom_script.py:my_func`",
            default=None,
        )

        self.group.add_argument(
            "--data-loader-func-name",
            help="[DEPRECATED - function name can be specified with --data-loader-script like so: `my_custom_script.py:my_func`] "
            "When using a data-loader-script, this specifies the name of the function "
            "that loads data. Defaults to `load_data`. ",
            default=None,
        )

    def parse_impl(self, args):
        """
        Attributes:
            seed (int): The seed to use for random data generation.
            val_range (Dict[str, Tuple[int]]): Per-input ranges of values to generate.
            iterations (int): The number of iterations for which to generate data.
            load_inputs_paths (List[str]): Path(s) from which to load inputs.
            data_loader_script (str): Path to a custom script to load inputs.
            data_loader_func_name (str): Name of the function in the custom data loader script that loads data.
        """

        def omit_none_tuple(tup):
            if all([elem is None for elem in tup]):
                return None
            return tup

        self.seed = args_util.get(args, "seed")

        self._int_range = omit_none_tuple(tup=(args_util.get(args, "int_min"), args_util.get(args, "int_max")))
        self._float_range = omit_none_tuple(tup=(args_util.get(args, "float_min"), args_util.get(args, "float_max")))
        if self._int_range or self._float_range:
            mod.warn_deprecated(
                "--int-min/--int-max and --float-min/--float-max",
                use_instead="--val-range, which allows you to specify per-input data ranges,",
                remove_in="0.50.0",
                always_show_warning=True,
            )

        self.val_range = args_util.parse_arglist_to_dict(
            args_util.get(args, "val_range"), cast_to=lambda x: tuple(args_util.cast(x))
        )
        if self.val_range is not None:
            for name, vals in self.val_range.items():
                if len(vals) != 2:
                    G_LOGGER.critical(
                        f"In --val-range, for input: {name}, expected to receive exactly 2 values, "
                        f"but received {len(vals)}.\nNote: Option was parsed as: input: {name}, range: {vals}"
                    )

                if any(not isinstance(elem, numbers.Number) for elem in vals):
                    G_LOGGER.critical(
                        f"In --val-range, for input: {name}, one or more elements of the range could not be parsed as a number.\n"
                        f"Note: Option was parsed as: input: {name}, range: {vals}"
                    )

        self.iterations = args_util.get(args, "iterations")

        self.load_inputs_paths = args_util.get(args, "load_inputs_paths")

        self.data_loader_script, self.data_loader_func_name = args_util.parse_script_and_func_name(
            args_util.get(args, "data_loader_script"), default_func_name="load_data"
        )
        func_name = args_util.get(args, "data_loader_func_name")
        if func_name is not None:
            mod.warn_deprecated("--data-loader-func-name", "--data-loader-script", "0.50.0", always_show_warning=True)
            self.data_loader_func_name = func_name

        if self.load_inputs_paths or self.data_loader_script:
            for arg in ["seed", "int_min", "int_max", "float_min", "float_max", "val_range", "iterations"]:
                val = args_util.get(args, arg)
                if val is not None:
                    G_LOGGER.warning(
                        f"Argument: '--{arg.replace('_', '-')}' will be ignored since a custom data loader was provided.\n"
                        "This argument is only valid when using the default data loader."
                    )

    def _add_to_script_helper(self, script, user_input_metadata_str=None):
        needs_invoke = False
        using_random_data = False

        if self.data_loader_script:
            script.add_import(imports=["mod"], frm="polygraphy")
            data_loader = make_invocable(
                "mod.import_from_script", self.data_loader_script, name=self.data_loader_func_name
            )
            needs_invoke = True
        elif self.load_inputs_paths:
            script.add_import(imports=["load_json"], frm="polygraphy.json")
            data_loader = safe(
                "[]\nfor input_data_path in {load_inputs_paths}:"
                "\n{tab}{data_loader}.extend(load_json(input_data_path, description='input data'))",
                load_inputs_paths=self.load_inputs_paths,
                data_loader=Script.DATA_LOADER_NAME,
                tab=inline(safe(constants.TAB)),
            )
        else:
            using_random_data = True
            if (
                user_input_metadata_str is None
                and self._allow_custom_input_shapes
                and self.arg_groups[ModelArgs].input_shapes
            ):
                user_input_metadata_str = self.arg_groups[ModelArgs].input_shapes

            if user_input_metadata_str:
                script.add_import(imports=["TensorMetadata"], frm="polygraphy.common")

            data_loader = make_invocable_if_nondefault(
                "DataLoader",
                seed=self.seed,
                iterations=self.iterations,
                input_metadata=user_input_metadata_str,
                int_range=self._int_range,
                float_range=self._float_range,
                val_range=self.val_range,
            )
            if data_loader:
                script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")

        if using_random_data != self.is_using_random_data():
            G_LOGGER.internal_error("is_using_random_data() reported a false positive!")

        return (script.set_data_loader(data_loader), needs_invoke)

    def add_to_script_impl(self, script, user_input_metadata_str=None):
        """
        Args:
            user_input_metadata_str (str(TensorMetadata)):
                    The name of a variable containing TensorMetadata.
                    This will control the shape and data type of the generated
                    data.

        Returns:
            str: The data loader, as a string. This may either be the variable name,
                or an invocation of the data loader function.
        """
        data_loader, needs_invoke = self._add_to_script_helper(script, user_input_metadata_str)
        if needs_invoke:
            data_loader = make_invocable(data_loader)
        return data_loader

    def get_data_loader(self, user_input_metadata=None):
        """
        Creates a data loader according to arguments provided on the command-line.

        Returns:
            Sequence[OrderedDict[str, numpy.ndarray]]
        """
        from polygraphy.comparator import DataLoader

        needs_invoke = False

        # run_script expects the callable to return just the variable name, but self.add_to_script
        # has 2 return values. We wrap it here to create a function with the right signature.
        def add_to_script_wrapper(script, *args, **kwargs):
            nonlocal needs_invoke
            name, needs_invoke = self._add_to_script_helper(script, *args, **kwargs)
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
        return not self.data_loader_script and not self.load_inputs_paths
