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
import enum
import inspect
import os
import sys
import time
import traceback

COLORED_MODULE_PRESENT = None


def has_colors():
    global COLORED_MODULE_PRESENT
    if COLORED_MODULE_PRESENT is None:
        try:
            import colored

            COLORED_MODULE_PRESENT = True
        except:
            COLORED_MODULE_PRESENT = False
            print(
                "[W] 'colored' module is not installed, will not use colors when logging. "
                "To enable colors, please install the 'colored' module: python3 -m pip install colored"
            )
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
    """
    Specifies how messages should be logged.
    """

    EACH = 0
    """Log the message each time"""
    ONCE = 1
    """Log the message only once. The same message will not be logged again."""


class Logger(object):
    ULTRA_VERBOSE = -20  # Cast it into the flames!
    SUPER_VERBOSE = -10
    EXTRA_VERBOSE = 0
    VERBOSE = 10
    INFO = 20
    START = 22
    FINISH = 28
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    SEVERITY_LETTER_MAPPING = {
        ULTRA_VERBOSE: "[U]",
        SUPER_VERBOSE: "[S]",
        EXTRA_VERBOSE: "[X]",
        VERBOSE: "[V]",
        INFO: "[I]",
        START: "[I]",
        FINISH: "[I]",
        WARNING: "[W]",
        ERROR: "[E]",
        CRITICAL: "[!]",
    }

    SEVERITY_COLOR_MAPPING = {
        ULTRA_VERBOSE: "dark_gray",
        SUPER_VERBOSE: "medium_violet_red",
        EXTRA_VERBOSE: "medium_purple",
        VERBOSE: "light_magenta",
        INFO: None,
        START: "light_cyan",
        FINISH: "light_green",
        WARNING: "light_yellow",
        ERROR: "light_red",
        CRITICAL: "light_red",
    }

    def __init__(self, severity=INFO, colors=True, letter=True, timestamp=False, line_info=False):
        """
        Logger.

        Args:
            severity (Logger.Severity):
                    Messages below this severity are ignored.
            colors (bool):
                    Whether to use colored output.
                    Defaults to True.
            letter (bool):
                    Whether to prepend each logging message with a letter indicating it's severity.
                    Defaults to True.
            timestamp (bool):
                    Whether to include a timestamp in the logging output.
                    Defaults to False.
            line_info (bool):
                    Whether to include file and line number information in the logging output.
                    Defaults to False.
            log_file (str):
                    Path to a log file to write logging output from Polygraphy.
                    This will not include logging messages from libraries used by Polygraphy, like
                    TensorRT or ONNX-Runtime.
        """
        self._severity = severity
        self._log_path = None
        self._log_file = None
        self.logging_indent = 0
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.once_logged = set()
        self.colors = colors
        self.letter = letter
        self.timestamp = timestamp
        self.line_info = line_info
        self.logger_callbacks = []

    @property
    def log_file(self):
        return self._log_path

    @log_file.setter
    def log_file(self, value):
        self._log_path = value
        dir_path = os.path.dirname(self._log_path)
        if dir_path:
            dir_path = os.path.realpath(dir_path)
            os.makedirs(dir_path, exist_ok=True)
        self._log_file = open(self._log_path, "w")

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

    def log(self, message, severity, mode=LogMode.EACH, stack_depth=2, error_ok=False):
        """
        Logs a message to stdout.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            severity (Logger.Severity):
                    The severity with which to log this message. If the severity is less than
                    the logger's current severity, the message is suppressed. Provided callables
                    will not be called in that case.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
            stack_depth (int):
                    The stack depth to use to determine file and line information.
                    Defaults to 2.
            error_ok (bool):
                    Whether to suppress errors encountered while logging.
                    When this is True, in the event of an error, the message will not be
                    logged, but the logger will recover and resume execution.
                    When False, the logger will re-raise the exception.
        """
        from polygraphy import constants, config

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
            try:
                message = message()
            except Exception as err:
                if not error_ok or config.INTERNAL_CORRECTNESS_CHECKS:
                    raise
                message = "<Error while logging this message: {:}>".format(str(err))

        message = str(message)
        message = message.replace("\t", constants.TAB)

        # Use the warnings module in correctness checking mode so all warnings are
        # visible in the test result summary.
        if config.INTERNAL_CORRECTNESS_CHECKS and severity == Logger.WARNING:
            import warnings

            warnings.warn(message)

        message = process_message(message, stack_depth=stack_depth)

        if self._log_file is not None:
            self._log_file.write(message + "\n")
            self._log_file.flush()

        print(message, file=sys.stdout if severity < Logger.CRITICAL else sys.stderr)

    def backtrace(self, depth=0, limit=None, severity=ERROR):
        limit = limit if limit is not None else (3 - self.severity // 10) * 2  # Info provides 1 stack frame
        limit = max(limit, 0)
        self.log(" ".join(traceback.format_stack(f=sys._getframe(depth + 2), limit=limit)), severity=severity)

    def ultra_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.ULTRA_VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def super_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.SUPER_VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def extra_verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.EXTRA_VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def verbose(self, message, mode=LogMode.EACH):
        self.log(message, Logger.VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def info(self, message, mode=LogMode.EACH):
        self.log(message, Logger.INFO, mode=mode, stack_depth=3)

    def start(self, message, mode=LogMode.EACH):
        self.log(message, Logger.START, mode=mode, stack_depth=3)

    def finish(self, message, mode=LogMode.EACH):
        self.log(message, Logger.FINISH, mode=mode, stack_depth=3)

    def warning(self, message, mode=LogMode.EACH):
        self.log(message, Logger.WARNING, mode=mode, stack_depth=3)

    def error(self, message, mode=LogMode.EACH):
        self.log(message, Logger.ERROR, mode=mode, stack_depth=3)

    def critical(self, message):
        self.log(message, Logger.CRITICAL, stack_depth=3)
        from polygraphy.exception import PolygraphyException

        raise PolygraphyException(message) from None

    def internal_error(self, message):
        from polygraphy import config

        if config.INTERNAL_CORRECTNESS_CHECKS:
            self.log(message, Logger.CRITICAL, stack_depth=3)
            from polygraphy.exception import PolygraphyInternalException

            raise PolygraphyInternalException(message) from None

    def _str_from_module_info(self, module, name=None):
        ret = ""

        def try_append(func):
            nonlocal ret
            try:
                ret += func()
            except:
                pass

        try_append(lambda: name or "Loaded Module: {:<18}".format(module.__name__))
        try_append(lambda: " | Version: {:<8}".format(module.__version__))
        try_append(lambda: " | Path: {:}".format(list(map(os.path.realpath, module.__path__))))
        return ret

    def module_info(self, module, name=None, severity=VERBOSE):
        self.log(self._str_from_module_info(module, name), severity=severity, mode=LogMode.ONCE)

    def log_exception(self, func):
        """
        Decorator that causes exceptions in a function to be logged.
        This is useful in cases where the exception is caught by a caller, but should
        still be logged.
        """

        def wrapped(*args, **kwargs):
            from polygraphy.exception import PolygraphyException

            try:
                return func(*args, **kwargs)
            except PolygraphyException:
                # `PolygraphyException`s are always logged.
                raise
            except Exception as err:
                G_LOGGER.error(err)
                raise

        return wrapped


global G_LOGGER
G_LOGGER = Logger()

# For backwards compatibility
G_LOGGER.exit = G_LOGGER.critical
