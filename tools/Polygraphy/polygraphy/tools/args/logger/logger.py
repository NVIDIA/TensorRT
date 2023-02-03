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
from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import inline_identifier, inline, safe


@mod.export()
class LoggerArgs(BaseArgs):
    """
    Logging: logging and debug output
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "-v",
            "--verbose",
            help="Increase logging verbosity. Specify multiple times for higher verbosity",
            action="count",
            default=0,
        )
        self.group.add_argument(
            "-q",
            "--quiet",
            help="Decrease logging verbosity. Specify multiple times for lower verbosity",
            action="count",
            default=0,
        )

        self.group.add_argument(
            "--verbosity",
            help="""
            The logging verbosity to use. Takes precedence over the `-v` and `-q` options,
            and unlike them, allows you to control per-path verbosity.
            Verbosity values should come from Polygraphy's logging verbosities defined in
            the `Logger` class and are case-insensitive.
            For example: `--verbosity INFO` or `--verbosity verbose`.
            To specify per-path verbosity, use the format:
            `<path>:<verbosity>`. For example:
            `--verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE`.
            Paths should be relative to the `polygraphy/` directory.
            For example, `polygraphy/backend` should be specified with just `backend`.
            The most closely matching path is used to determine verbosity.
            For example, with: `--verbosity warning backend:info backend/trt:verbose`,
            a file under `polygraphy/comparator` would use `WARNING` verbosity, one under
            `backend/onnx` would use `INFO`, and one under `backend/trt` would use `VERBOSE`.
            """,
            nargs="+",
            default=None,
        )

        self.group.add_argument("--silent", help="Disable all output", action="store_true", default=None)
        self.group.add_argument(
            "--log-format",
            help="Format for log messages: {{'timestamp': Include timestamp, 'line-info': Include file and line number, "
            "'no-colors': Disable colors}}",
            choices=["timestamp", "line-info", "no-colors"],
            nargs="+",
            default=[],
        )
        self.group.add_argument(
            "--log-file",
            help="Path to a file where Polygraphy logging output should be written. "
            "This may not include logging output from dependencies, like TensorRT or ONNX-Runtime. ",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            silent (bool): Whether to disable all logging output.
            log_format (List[str]): Formatting options for the logger.
            log_file (str): Path to a file where logging output should be written.
            verbosity (Dict[str, str]):
                    Per-path logging verbosities. A key of `""` represents the default verbosity.
        """
        self.silent = args_util.get(args, "silent")
        self.log_format = args_util.get(args, "log_format", default=[])
        self.log_file = args_util.get(args, "log_file")

        severity_level = args_util.get(args, "verbose") - args_util.get(args, "quiet")
        if severity_level >= 4:
            default_severity = "ULTRA_VERBOSE"
        elif severity_level < -4:
            default_severity = "CRITICAL"
        else:
            default_severity = {
                3: "SUPER_VERBOSE",
                2: "EXTRA_VERBOSE",
                1: "VERBOSE",
                0: None,
                -1: "START",
                -2: "FINISH",
                -3: "WARNING",
                -4: "ERROR",
            }[severity_level]

        self.verbosity = None
        verbosity = args_util.parse_arglist_to_dict(args_util.get(args, "verbosity"))
        if default_severity is not None:
            if verbosity is not None:
                verbosity[""] = default_severity
            else:
                verbosity = {"": default_severity}

        if verbosity is not None:
            self.verbosity = {}
            for path, sev in verbosity.items():
                self.verbosity[path] = inline(safe("G_LOGGER.{:}", inline_identifier(sev.upper())))

        # Enable logger settings immediately on parsing.
        self.get_logger()

    def add_to_script_impl(self, script):
        # Always required since it is used to print the exit message.
        script.append_preimport(safe("from polygraphy.logger import G_LOGGER"))

        logger_settings = []

        if self.silent:
            logger_settings.append("G_LOGGER.module_severity = G_LOGGER.CRITICAL")
        elif self.verbosity is not None:
            # Need to escape braces of the dictionary so it's not treated as a format-string by `safe()`.
            logger_settings.append(f"G_LOGGER.module_severity = {'{' + repr(self.verbosity) + '}'}")

        for fmt in self.log_format:
            if fmt == "no-colors":
                logger_settings.append("G_LOGGER.colors = False")
            elif fmt == "timestamp":
                logger_settings.append("G_LOGGER.timestamp = True")
            elif fmt == "line-info":
                logger_settings.append("G_LOGGER.line_info = True")

        if self.log_file:
            logger_settings.append(f"G_LOGGER.log_file = {repr(self.log_file)}")

        for setting in logger_settings:
            script.append_preimport(safe(setting))

        return safe("G_LOGGER")

    def get_logger(self):
        """
        Gets the global logger after applying command-line options.

        Returns:
            G_LOGGER
        """
        return args_util.run_script(self.add_to_script)
