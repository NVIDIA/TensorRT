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
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Script
from polygraphy.util import misc


class LoggerArgs(BaseArgs):
    def add_to_parser(self, parser):
        logging_args = parser.add_argument_group("Logging", "Options for logging and debug output")
        logging_args.add_argument("-v", "--verbose", help="Increase logging verbosity. Specify multiple times for higher verbosity", action="count", default=0)
        logging_args.add_argument("--silent", help="Disable all output", action="store_true", default=None)
        logging_args.add_argument("--log-format", help="Format for log messages: {{'timestamp': Include timestamp, 'line-info': Include file and line number, "
                                "'no-colors': Disable colors}}", choices=["timestamp", "line-info", "no-colors"], nargs="+", default=[])


    def parse(self, args):
        self.verbosity_count = tools_util.get(args, "verbose")
        self.silent = tools_util.get(args, "silent")
        self.log_format = misc.default_value(tools_util.get(args, "log_format"), [])

        # Enable logger settings immediately on parsing.
        self.get_logger()


    def add_to_script(self, script):
        # Always required since it is used to print the exit message.
        script.append_preimport("from polygraphy.logger import G_LOGGER")

        logger_settings = []
        if self.verbosity_count >= 4:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE")
        elif self.verbosity_count == 3:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.SUPER_VERBOSE")
        elif self.verbosity_count == 2:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.EXTRA_VERBOSE")
        elif self.verbosity_count == 1:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.VERBOSE")

        if self.silent:
            logger_settings.append("G_LOGGER.severity = G_LOGGER.CRITICAL")

        for fmt in self.log_format:
            if fmt == "no-colors":
                logger_settings.append("G_LOGGER.colors = False")
            elif fmt == "timestamp":
                logger_settings.append("G_LOGGER.timestamp = True")
            elif fmt == "line-info":
                logger_settings.append("G_LOGGER.line_info = True")

        for setting in logger_settings:
            script.append_preimport(setting)

        return "G_LOGGER"


    def get_logger(self):
        script = Script()
        logger_name = self.add_to_script(script)
        exec(str(script), globals(), locals())
        return locals()[logger_name]
