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
import copy
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
class LoggerIndent:
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
class LoggerVerbosity:
    def __init__(self, logger, severity):
        self.logger = logger
        self.old_severity = copy.copy(self.logger.module_severity)
        self.module_severity = severity

    def __enter__(self):
        self.logger.module_severity = self.module_severity
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.module_severity = self.old_severity


class LogMode(enum.IntEnum):
    """
    Specifies how messages should be logged.
    """

    EACH = 0
    """Log the message each time"""
    ONCE = 1
    """Log the message only once. The same message will not be logged again."""


class SeverityTrie:
    """
    A trie that represents per-path logging verbosities.
    """

    def _split_path(self, path):
        # Leading or duplicate slashes can create empty elements in the path components. We ignore those.
        return list(filter(lambda x: x, path.split(os.path.sep)))

    def __init__(self, severity_dict):
        assert "" in severity_dict, "severity_dict must include default severity!"

        self.trie = {}
        for path, severity in severity_dict.items():
            cur_dict = self.trie
            for path_component in self._split_path(path):
                if path_component not in cur_dict:
                    cur_dict[path_component] = {}
                cur_dict = cur_dict[path_component]
            cur_dict[""] = severity

        # Skip path checking if we don't have any path entries.
        self.has_non_default_entries = len(self.trie) > 1

    def get(self, path=None):
        """
        Get the logging verbosity for the given path.

        Args:
            path (str): The path

        Returns:
            int: The logging verbosity.
        """
        default_severity = self.trie[""]
        if path is None or not self.has_non_default_entries:
            return default_severity

        cur_dict = self.trie

        def get_value(dct):
            return dct.get("", default_severity)

        for path_component in self._split_path(path):
            if path_component not in cur_dict:
                return get_value(cur_dict)
            cur_dict = cur_dict[path_component]
        return get_value(cur_dict)

    def __str__(self):
        return str(self.trie)


class Logger:
    """
    Global logging interface. Do **not** construct a logger manually.
    Instead, use ``G_LOGGER``, the global logger.
    """

    ULTRA_VERBOSE = -20  # Cast it into the flames!
    """Enable unreasonably verbose messages and above"""
    SUPER_VERBOSE = -10
    """Enable extremely verbose messages and above"""
    EXTRA_VERBOSE = 0
    """Enable extra verbose messages and above"""
    VERBOSE = 10
    """Enable verbose messages and above"""
    INFO = 20
    """Enable informative messages and above"""
    START = 22
    """Enable messages indicating when a task is started and above"""
    FINISH = 28
    """Enable messages indicating when a task is finished and above"""
    WARNING = 30
    """Enable only warning messages and above"""
    ERROR = 40
    """Enable only error messages and above"""
    CRITICAL = 50
    """Enable only critical/fatal error messages and above"""

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
        Args:
            severity (Union[int, Dict[str, int]]):
                    Severity below which messages will be ignored.
                    This can be specified on a per-submodule/file basis by providing a dictionary of paths to
                    logging severities. In this case, use the ``""`` to indicate the default severity.
                    Paths should be relative to the `polygraphy/` directory.
                    For example, `polygraphy/backend` can be specified with just `backend/`.
                    For example: ``{"": G_LOGGER.INFO, "backend/trt": G_LOGGER.VERBOSE}``
                    This is converted to a ``SeverityTrie`` on assignment.
                    Defaults to G_LOGGER.INFO.
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
        """
        self.logging_indent = 0
        self.once_logged = set()
        self.colors = colors
        self.letter = letter
        self.timestamp = timestamp
        self.line_info = line_info
        self.logger_callbacks = []
        self.log_file = None
        """
        Path to a log file to write logging output from Polygraphy.
        This will not include logging messages from libraries used by Polygraphy, like
        TensorRT or ONNX-Runtime.
        """
        self.module_severity = severity
        """
        Severity below which messages will be ignored.
        This can be specified on a per-submodule/file basis by providing a dictionary of paths to
        logging severities. In this case, use the ``""`` to indicate the default severity.
        Paths should be relative to the `polygraphy/` directory.
        For example, `polygraphy/backend` can be specified with just `backend/`.
        For example: ``{"": G_LOGGER.INFO, "backend/trt": G_LOGGER.VERBOSE}``
        This is converted to a ``SeverityTrie`` on assignment.
        Defaults to G_LOGGER.INFO.
        """

    @property
    def log_file(self):
        return self._log_path

    @log_file.setter
    def log_file(self, value):
        self._log_path = value
        self._log_file = None
        if self._log_path:
            dir_path = os.path.dirname(self._log_path)
            if dir_path:
                dir_path = os.path.realpath(dir_path)
                os.makedirs(dir_path, exist_ok=True)
            self._log_file = open(self._log_path, "w")

    @property
    def module_severity(self):
        return self._module_severity

    @module_severity.setter
    def module_severity(self, value):
        if isinstance(value, SeverityTrie):
            self._module_severity = value
        else:
            if not isinstance(value, dict):
                value = {"": value}
            if "" not in value:
                value[""] = Logger.INFO
            self._module_severity = SeverityTrie(value)

        self._run_callbacks()

    @property
    def severity(self):
        print(
            "Warning: Accessing the `severity` property of G_LOGGER is deprecated and will be removed in v0.45.0. Use `module_severity` instead"
        )
        return self._module_severity.get()

    @severity.setter
    def severity(self, value):
        print(
            "Warning: Accessing the `severity` property of G_LOGGER is deprecated and will be removed in v0.45.0. Use `module_severity` instead"
        )
        self.module_severity = value

    def module_path(self, path):
        """
        Converts a given path to a path relative to the Polygraphy root module.
        If the path is not part of the Polygraphy module, returns a path relative to the common prefix.

        Args:
            path (str): The path

        Returns:
            str: The path relative to the Polygraphy root module or common prefix.
        """
        import polygraphy

        module_root_dir = polygraphy.__path__[0]
        file_path = os.path.relpath(path, module_root_dir)
        if os.pardir in file_path:
            common_path_len = len(os.path.commonpath([module_root_dir, path]))
            file_path = path[common_path_len:].lstrip(os.path.sep)
        return file_path

    def _run_callbacks(self):
        for callback in self.logger_callbacks:
            callback(self._module_severity)

    def register_callback(self, callback):
        """
        Registers a callback with the logger, which will be invoked when the logging severity is modified.
        The callback is guaranteed to be called at least once in the register_callback function.

        Args:
            callback (Callable(SeverityTrie)):
                    A callback that accepts the current logger severity trie.
        """
        callback(self._module_severity)
        self.logger_callbacks.append(callback)

    def indent(self, level=1):
        """
        Returns a context manager that indents all strings logged by the specified amount.

        Args:
            level (int): The indentation level
        """
        return LoggerIndent(self, level + self.logging_indent)

    def verbosity(self, severity=CRITICAL):
        """
        Returns a context manager that temporarily changes the severity of the logger for its duration.

        Args:
            severity (Union[int, Dict[str, int]]):
                    Severity below which messages will be ignored.
                    This can be specified on a per-submodule/file basis by providing a dictionary of paths to
                    logging severities. In this case, use the ``""`` to indicate the default severity.
                    For example: ``{"": G_LOGGER.INFO, "backend/trt": G_LOGGER.VERBOSE}``
                    Defaults to Logger.CRITICAL, which will suppress all messages.
        """
        return LoggerVerbosity(self, severity)

    def log(self, message, severity, mode=LogMode.EACH, stack_depth=2, error_ok=False):
        """
        Logs a message to stdout.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            severity (int):
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
        from polygraphy import config, constants

        def get_rel_file_path_and_lineno():
            file_path = sys._getframe(stack_depth).f_code.co_filename
            line_no = sys._getframe(stack_depth).f_lineno
            # If we can't get a valid path, keep walking the stack until we can.
            new_stack_depth = stack_depth
            while not os.path.exists(file_path) and new_stack_depth > 0:
                new_stack_depth -= 1
                file_path = sys._getframe(new_stack_depth).f_code.co_filename
                line_no = sys._getframe(new_stack_depth).f_lineno

            return self.module_path(file_path), line_no

        def process_message(message, file_path, line_no):
            def get_prefix():
                prefix = ""
                if self.letter:
                    prefix += Logger.SEVERITY_LETTER_MAPPING[severity] + " "
                if self.timestamp:
                    prefix += f"({time.strftime('%X')}) "
                if self.line_info:
                    prefix += f"[{file_path}:{line_no}] "
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
            return apply_color(f"{prefix}{message}")

        file_path, line_no = None, None
        if self.line_info or self.module_severity.has_non_default_entries:
            file_path, line_no = get_rel_file_path_and_lineno()

        def should_log(message):
            if severity < self.module_severity.get(file_path):
                return False

            if mode == LogMode.ONCE:
                message_hash = hash(message)
                if message_hash in self.once_logged:
                    return False
                self.once_logged.add(message_hash)
            return True

        if not should_log(message):
            return

        if callable(message):
            try:
                message = message()
            except Exception as err:
                if not error_ok or config.INTERNAL_CORRECTNESS_CHECKS:
                    raise
                message = f"<Error while logging this message: {str(err)}>"

        message = str(message)

        # Use the warnings module in correctness checking mode so all warnings are
        # visible in the test result summary.
        if config.INTERNAL_CORRECTNESS_CHECKS and severity == Logger.WARNING:
            import warnings

            warnings.warn(message)

        message = process_message(message, file_path, line_no)

        if self._log_file is not None:
            self._log_file.write(message + "\n")
            self._log_file.flush()

        print(message, file=sys.stdout if severity < Logger.CRITICAL else sys.stderr)

    def backtrace(self, depth=0, limit=None, severity=ERROR):
        limit = (
            limit if limit is not None else (3 - self.module_severity.get() // 10) * 2
        )  # Info provides 1 stack frame
        limit = max(limit, 0)
        frame = sys._getframe(depth + 2)
        self.log(" ".join(traceback.format_stack(f=frame, limit=limit)), severity=severity)

    def ultra_verbose(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with ULTRA_VERBOSE severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.ULTRA_VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def super_verbose(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with SUPER_VERBOSE severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.SUPER_VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def extra_verbose(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with EXTRA_VERBOSE severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.EXTRA_VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def verbose(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with VERBOSE severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.VERBOSE, mode=mode, stack_depth=3, error_ok=True)

    def info(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with INFO severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.INFO, mode=mode, stack_depth=3)

    def start(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with START severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.START, mode=mode, stack_depth=3)

    def finish(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with FINISH severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.FINISH, mode=mode, stack_depth=3)

    def warning(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with WARNING severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.WARNING, mode=mode, stack_depth=3)

    def error(self, message, mode=LogMode.EACH):
        """
        Logs a message to stdout with ERROR severity.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.
        """
        self.log(message, Logger.ERROR, mode=mode, stack_depth=3)

    def critical(self, message):
        """
        Logs a message to stdout with CRITICAL severity and raises an exception.

        Args:
            message (Union[str, Callable() -> str]):
                    A string or callable which returns a string of the message to log.
            mode (LogMode):
                    Controls how the message is logged.
                    See LogMode for details.

        Raises:
            PolygraphyException
        """
        self.log(message, Logger.CRITICAL, stack_depth=3)
        from polygraphy.exception import PolygraphyException

        raise PolygraphyException(message) from None

    def internal_error(self, message):
        from polygraphy import config

        if not config.INTERNAL_CORRECTNESS_CHECKS:
            return

        self.log(message, Logger.CRITICAL, stack_depth=3)
        from polygraphy.exception import PolygraphyInternalException

        raise PolygraphyInternalException(message)

    def _str_from_module_info(self, module, name=None):
        ret = ""

        def try_append(func):
            nonlocal ret
            try:
                ret += func()
            except:
                pass

        try_append(lambda: name or f"Loaded Module: {module.__name__}")
        try_append(lambda: f" | Version: {module.__version__}")
        try_append(lambda: f" | Path: {list(map(os.path.realpath, module.__path__))}")
        return ret

    def module_info(self, module, name=None, severity=VERBOSE):
        message = self._str_from_module_info(module, name)
        self.log(message, severity=severity, stack_depth=3, mode=LogMode.ONCE)

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


G_LOGGER = Logger()
"""The global logger. Use this instead of constructing a logger"""

# For backwards compatibility
G_LOGGER.exit = G_LOGGER.critical
