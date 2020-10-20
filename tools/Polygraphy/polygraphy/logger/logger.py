#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import inspect
import enum
import time
import sys
import os


try:
    ModuleNotFoundError
except:
    ModuleNotFoundError = ImportError


COLORED_MODULE_PRESENT = None
def has_colors():
    global COLORED_MODULE_PRESENT
    if COLORED_MODULE_PRESENT is None:
        try:
            import colored
            COLORED_MODULE_PRESENT = True
        except (ImportError, ModuleNotFoundError):
            COLORED_MODULE_PRESENT = False
            print("[W] 'colored' module is not installed, will not use colors when logging. "
                  "To enable colors, please install the 'colored' module: python3 -m pip install colored")
    return COLORED_MODULE_PRESENT


# Context manager to apply indentation to messages
class LoggerIndent(object):
    def __init__(self, logger, indent):
        self.logger = logger
        self.old_indent = self.logger.logging_indent
        self.indent = indent

    def __enter__(self):
        self.logger.logging_indent = self.indent
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.logging_indent = self.old_indent


# Context manager to temporarily set verbosity
class LoggerVerbosity(object):
    def __init__(self, logger, severity):
        self.logger = logger
        self.old_severity = self.logger.severity
        self.severity = severity

    def __enter__(self):
        self.logger.severity = self.severity
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.severity = self.old_severity


class LogMode(enum.IntEnum):
    EACH = 0 # Log the message each time
    ONCE = 1 # Log the message only once. The same message will not be logged again.


class Logger(object):
    ULTRA_VERBOSE = -20 # Cast it into the flames!
    SUPER_VERBOSE = -10
    EXTRA_VERBOSE = 0
    VERBOSE = 10
    INFO = 20
    SUCCESS = 21
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    SEVERITY_LETTER_MAPPING = {
        ULTRA_VERBOSE: "[U]",
        SUPER_VERBOSE: "[P]",
        EXTRA_VERBOSE: "[X]",
        VERBOSE: "[V]",
        INFO: "[I]",
        SUCCESS: "[S]",
        WARNING: "[W]",
        ERROR: "[E]",
        CRITICAL: "[C]",
    }

    SEVERITY_COLOR_MAPPING = {
        ULTRA_VERBOSE: "dark_green",
        SUPER_VERBOSE: "green",
        EXTRA_VERBOSE: "cyan",
        VERBOSE: "dark_gray",
        INFO: None,
        SUCCESS: "light_green",
        WARNING: "light_yellow",
        ERROR: "light_red",
        CRITICAL: "light_red",
    }

    def __init__(self, severity=INFO, colors=True, letter=True, timestamp=False, line_info=False, exit_on_errors=False):
        """
        Logger.

        Args:
            severity (Logger.Severity): Messages below this severity are ignored.
            colors (bool): Whether to use colored output.
            letter (bool): Whether to prepend each logging message with a letter indicating it's severity. Defaults to True.
            timestamp (bool): Whether to include a timestamp in the logging output. Defaults to False.
            line_info (bool): Whether to include file and line number information in the logging output. Defaults to False.
        """
        self._severity = severity
        self.logging_indent = 0
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,  os.pardir))
        self.once_logged = set()
        self.colors = colors
        self.letter = letter
        self.timestamp = timestamp
        self.line_info = line_info
        self.exit_on_errors = exit_on_errors
        self.logger_callbacks = []


    @property
    def severity(self):
        return self._severity


    @severity.setter
    def severity(self, value):
        self._severity = value
        for callback in self.logger_callbacks:
            callback(self._severity)


    def register_callback(self, callback):
        """
        Registers a callback with the logger, which will be invoked when the logging severity is modified.
        The callback is guaranteed to be called at least once in the register_callback function.

        Args:
            callback (Callable(Logger.Severity)): A callback that accepts the current logger severity.
        """
        callback(self._severity)
        self.logger_callbacks.append(callback)


    def indent(self, level=1):
        """
        Returns a context manager that indents all strings logged by the specified amount.
        """
        return LoggerIndent(self, level + self.logging_indent)


    def verbosity(self, severity=CRITICAL):
        """
        Returns a context manager that temporarily changes the severity of the logger for its duration.

        Args:
            severity (Logger.Severity):
                    The severity to set the logger to. Defaults to Logger.CRITICAL, which will suppress all messages.
        """
        return LoggerVerbosity(self, severity)


    # If once is True, the logger will only log this message a single time. Useful in loops.
    # message may be a callable which returns a message. This way, only if the message needs to be logged is it ever generated.
    def log(self, message, severity, mode=LogMode.EACH, stack_depth=2):
        def process_message(message, stack_depth):
            def get_prefix():
                def get_line_info():
                    adjusted_stack_depth = stack_depth
                    adjusted_stack_depth += 2
                    module = inspect.getmodule(sys._getframe(adjusted_stack_depth))
                    # Handle logging from the top-level of a module.
                    if not module:
                        adjusted_stack_depth -= 1
                        module = inspect.getmodule(sys._getframe(adjusted_stack_depth))
                    filename = module.__file__
                    filename = os.path.relpath(filename, self.root_dir)
                    # If the file is not located in polygraphy, use its basename instead.
                    if os.pardir in filename:
                        filename = os.path.basename(filename)
                    return "[{:}:{:}] ".format(filename, sys._getframe(adjusted_stack_depth).f_lineno)

                prefix = ""
                if self.letter:
                    prefix += Logger.SEVERITY_LETTER_MAPPING[severity] + " "
                if self.timestamp:
                    prefix += "({:}) ".format(time.strftime("%X"))
                if self.line_info:
                    prefix += get_line_info()
                return prefix


            def apply_indentation(prefix, message):
                from polygraphy.common import constants

                message_lines = str(message).splitlines()
                tab = constants.TAB * self.logging_indent
                newline_tab = "\n" + tab + " " * len(prefix)
                return tab + newline_tab.join([line for line in message_lines])


            def apply_color(message):
                if self.colors and has_colors():
                    import colored
                    color = Logger.SEVERITY_COLOR_MAPPING[severity]
                    return colored.stylize(message, [colored.fg(color)]) if color else message
                return message


            prefix = get_prefix()
            message = apply_indentation(prefix, message)
            return apply_color("{:}{:}".format(prefix, message))


        def should_log(message):
            should = severity >= self._severity
            if should and mode == LogMode.ONCE:
                message_hash = hash(message)
                should &= message_hash not in self.once_logged
                self.once_logged.add(message_hash)
            return should


        if not should_log(message):
            return

        if callable(message):
            message = message()
        message = str(message)
        print(process_message(message, stack_depth=stack_depth))


    def ultra_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.ULTRA_VERBOSE, mode=mode, stack_depth=3)


    def super_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.SUPER_VERBOSE, mode=mode, stack_depth=3)


    def extra_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.EXTRA_VERBOSE, mode=mode, stack_depth=3)


    def verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.VERBOSE, mode=mode, stack_depth=3)


    def info(self, message, mode=LogMode.EACH):
        self.log(message, Logger.INFO, mode=mode, stack_depth=3)


    def success(self, message, mode=LogMode.EACH):
        self.log(message, Logger.SUCCESS, mode=mode, stack_depth=3)


    def warning(self, message, mode=LogMode.EACH):
        self.log(message, Logger.WARNING, mode=mode, stack_depth=3)


    def error(self, message, mode=LogMode.EACH):
        self.log(message, Logger.ERROR, mode=mode, stack_depth=3)


    # Like error, but immediately exits.
    def critical(self, message):
        self.log(message, Logger.CRITICAL, stack_depth=3)
        if self.exit_on_errors:
            sys.exit(1)
        else:
            from polygraphy.common import PolygraphyException
            raise PolygraphyException("Error encountered - see logging output for details") from None # Erase exception chain


global G_LOGGER
G_LOGGER = Logger()
