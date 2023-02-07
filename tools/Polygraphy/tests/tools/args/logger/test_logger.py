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

import os
import tempfile

import pytest
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import LoggerArgs
from tests.tools.args.helper import ArgGroupTestHelper

VERBOSITY_CASES = {
    "--silent": G_LOGGER.CRITICAL,
    "-qqqqq": G_LOGGER.CRITICAL,
    "-qqqq": G_LOGGER.ERROR,
    "-qqq": G_LOGGER.WARNING,
    "-qq": G_LOGGER.FINISH,
    "-q": G_LOGGER.START,
    "-v": G_LOGGER.VERBOSE,
    "-vv": G_LOGGER.EXTRA_VERBOSE,
    "-vvv": G_LOGGER.SUPER_VERBOSE,
    "-vvvv": G_LOGGER.ULTRA_VERBOSE,
}


class TestLoggerArgs:
    @pytest.mark.parametrize("case", VERBOSITY_CASES.items())
    def test_get_logger_verbosities(self, case):
        arg_group = ArgGroupTestHelper(LoggerArgs())
        flag, sev = case

        arg_group.parse_args([flag])
        logger = arg_group.get_logger()

        assert logger.module_severity.get() == sev

    @pytest.mark.parametrize(
        "option, expected_values",
        [
            (["--verbosity", "INFO"], {None: G_LOGGER.INFO, "backend/": G_LOGGER.INFO}),
            # Test case-sensitivity
            (["--verbosity", "info"], {None: G_LOGGER.INFO, "backend/": G_LOGGER.INFO}),
            (
                ["--verbosity", "INFO", "backend:VERBOSE"],
                {None: G_LOGGER.INFO, "backend": G_LOGGER.VERBOSE, os.path.join("backend", "trt"): G_LOGGER.VERBOSE},
            ),
            (
                ["--verbosity", "ULTRA_VERBOSE", "backend:VERBOSE"],
                {
                    None: G_LOGGER.ULTRA_VERBOSE,
                    "backend": G_LOGGER.VERBOSE,
                    os.path.join("backend", "trt"): G_LOGGER.VERBOSE,
                },
            ),
            (
                ["--verbosity", "backend/trt:VERBOSE"],
                {None: G_LOGGER.INFO, "backend/": G_LOGGER.INFO, os.path.join("backend", "trt"): G_LOGGER.VERBOSE},
            ),
        ],
    )
    def test_per_path_verbosities(self, option, expected_values):
        arg_group = ArgGroupTestHelper(LoggerArgs())
        arg_group.parse_args(option)

        logger = arg_group.get_logger()
        for path, sev in expected_values.items():
            assert logger.module_severity.get(path) == sev

    def test_logger_log_file(self):
        arg_group = ArgGroupTestHelper(LoggerArgs())

        with tempfile.TemporaryDirectory() as dirname:
            log_path = os.path.join(dirname, "fake_log_file.log")
            arg_group.parse_args(["--log-file", log_path])
            logger = arg_group.get_logger()
            assert logger.log_file == log_path
