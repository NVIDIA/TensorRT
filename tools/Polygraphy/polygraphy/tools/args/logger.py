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
from polygraphy import mod, util
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import safe


@mod.export()
class LoggerArgs(BaseArgs):
    def add_to_parser(self, parser):
        logging_args = parser.add_argument_group("Logging", "Options for logging and debug output")

        logging_args.add_argument(
            "-v",
            "--verbose",
            help="Increase logging verbosity. Specify multiple times for higher verbosity",
            action="count",
            default=0,
        )
        logging_args.add_argument(
            "-q",
            "--quiet",
            help="Decrease logging verbosity. Specify multiple times for lower verbosity",
            action="count",
            default=0,
        )

        logging_args.add_argument("--silent", help="Disable all output", action="store_true", default=None)
        logging_args.add_argument(
            "--log-format",
            help="Format for log messages: {{'timestamp': Include timestamp, 'line-info': Include file and line number, "
            "'no-colors': Disable colors}}",
            choices=["timestamp", "line-info", "no-colors"],
            nargs="+",
            default=[],
        )
        logging_args.add_argument(
            "--log-file",
            help="Path to a file where Polygraphy logging output should be written. "
            "This will not include logging output from dependencies, like TensorRT or ONNX-Runtime. ",
            default=None,
        )

    def parse(self, args):
        self.verbosity_count = args_util.get(args, "verbose") - args_util.get(args, "quiet")
        self.silent = args_util.get(args, "silent")
        self.log_format = args_util.get(args, "log_format", default=[])
        self.log_file = args_util.get(args, "log_file")

        # Enable logger settings immediately on parsing.
        self.get_logger()

    def add_to_script(self, script):
        # Always required since it is used to print the exit message.
        script.append_preimport(safe("from polygraphy.logger import G_LOGGER"))

        logger_settings = []
        if self.verbosity_count >= 4:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE")
        elif self.verbosity_count == 3:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.SUPER_VERBOSE")
        elif self.verbosity_count == 2:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.EXTRA_VERBOSE")
        elif self.verbosity_count == 1:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.VERBOSE")
        elif self.verbosity_count == -1:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.START")
        elif self.verbosity_count == -2:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.FINISH")
        elif self.verbosity_count == -3:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.WARNING")
        elif self.verbosity_count == -4:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.ERROR")
        elif self.verbosity_count <= -4:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.CRITICAL")

        if self.silent:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.CRITICAL")

        for fmt in self.log_format:
            if fmt == "no-colors":
                logger_settings.append("G_LOGGER.colors = False")
            elif fmt == "timestamp":
                logger_settings.append("G_LOGGER.timestamp = True")
            elif fmt == "line-info":
                logger_settings.append("G_LOGGER.line_info = True")

        if self.log_file:
            logger_settings.append("G_LOGGER.log_file = {:}".format(repr(self.log_file)))

        for setting in logger_settings:
            script.append_preimport(safe(setting))

        return safe("G_LOGGER")

    def get_logger(self):
        return args_util.run_script(self.add_to_script)
