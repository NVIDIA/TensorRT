#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import logging

import pytest

from polygraphy import util
from polygraphy.exception.exception import PolygraphyException
from polygraphy.logger.logger import Logger, SeverityTrie


# We don't use the global logger here because we would have to reset the state each time.
class TestLogger:
    def test_log_file(self):
        logger = Logger()
        with util.NamedTemporaryFile("w+") as log_file:
            logger.log_file = log_file.name
            assert logger.log_file == log_file.name
            logger.info("Hello")

            log_file.seek(0)
            assert log_file.read() == "[I] Hello\n"

    def test_line_info(self):
        logger = Logger(line_info=True)
        with util.NamedTemporaryFile("w+") as log_file:
            logger.log_file = log_file.name

            logger.info("Hello")
            log_file.seek(0)
            assert (
                f"[I] [tests/logger/test_logger.py:{inspect.currentframe().f_lineno - 3}] Hello\n"
                == log_file.read()
            )

    def test_severity_trie_with_no_default(self):
        logger = Logger(severity={"backend/trt": 10})
        assert logger.module_severity.get() == Logger.INFO
        assert logger.module_severity.get("backend/trt") == 10

    def test_callbacks_triggered(self):
        logger = Logger()

        num_times_called = 0

        def callback(module_severity):
            nonlocal num_times_called
            num_times_called += 1

        logger.register_callback(callback)
        assert num_times_called == 1

        # Callbacks should be triggered whenever severity is set
        logger.module_severity = {"", Logger.INFO}
        assert num_times_called == 2

        # Callbacks should be triggered both when we enter and exit the context manager.
        with logger.verbosity():
            assert num_times_called == 3
        assert num_times_called == 4

    @pytest.mark.serial
    def test_use_python_logging_system(self, tmp_python_log_file):
        # Clear log file
        with tmp_python_log_file.open("w") as fp:
            fp.write("")

        logger = Logger(severity=Logger.ULTRA_VERBOSE)
        logger.use_python_logging_system = True
        # add custom Polygraphy levels
        logging.addLevelName(2, "ULTRA_VERBOSE")
        logging.addLevelName(4, "SUPER_VERBOSE")
        logging.addLevelName(6, "EXTRA_VERBOSE")
        logging.addLevelName(22, "START")
        logging.addLevelName(28, "FINISH")

        # emit logs
        logger.ultra_verbose("ultra verbose")
        logger.super_verbose("super verbose")
        logger.extra_verbose("extra verbose")
        logger.verbose("verbose")
        logger.info("info")
        logger.start("start")
        logger.finish("finish")
        logger.warning("warning")
        logger.error("error")
        with pytest.raises(PolygraphyException):
            logger.critical("critical")

        # verify logs written in the log file
        with tmp_python_log_file.open() as fp:
            log_messages = fp.read()

        # Remove lines containing "pytest_shutil.workspace"
        log_messages = "\n".join(
            line
            for line in log_messages.splitlines()
            if "pytest_shutil.workspace" not in line
        ) + ("\n" if log_messages.endswith("\n") else "")

        assert (
            log_messages
            == """\
ULTRA_VERBOSE:Polygraphy:[U] ultra verbose
SUPER_VERBOSE:Polygraphy:[S] super verbose
EXTRA_VERBOSE:Polygraphy:[X] extra verbose
DEBUG:Polygraphy:[V] verbose
INFO:Polygraphy:[I] info
START:Polygraphy:[I] start
FINISH:Polygraphy:[I] finish
WARNING:Polygraphy:[W] warning
ERROR:Polygraphy:[E] error
CRITICAL:Polygraphy:[!] critical
"""
        )


class TestSeverityTrie:
    @pytest.mark.parametrize(
        "path,sev",
        [
            # Not in trie
            ("backend", 30),
            ("mod/importer.py", 30),
            # Exact paths in trie
            ("backend/onnx", 28),
            ("backend/trt", 20),
            # Submodule of path in trie
            ("backend/trt/loader.py", 50),
            ("backend/trt/runner.py", 20),
            ("backend/onnx/loader.py", 28),
            # Ensure paths with leading or duplicate slashes work
            ("/backend", 30),
            ("backend/////trt", 20),
        ],
    )
    def test_get(self, path, sev):
        # Duplicate slashes should be handled
        trie = SeverityTrie(
            {
                "": 30,
                "backend/trt": 20,
                "backend/trt/loader.py": 50,
                "backend///////onnx": 28,
            }
        )
        assert trie.get(path) == sev
