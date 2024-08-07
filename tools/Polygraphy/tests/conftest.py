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

import copy
import ctypes.util
import glob
import logging
import os
import subprocess as sp
import sys

import pytest

from tests.helper import ROOT_DIR


@pytest.fixture()
def sandboxed_install_run(virtualenv, script_runner):
    """
    A special fixture that runs commands, but sandboxes any `pip install`s in a virtual environment.
    Packages from the test environment are still usable, but those in the virtual environment take precedence
    """

    VENV_PYTHONPATH = glob.glob(
        os.path.join(virtualenv.virtualenv, "lib", "python*", "site-packages")
    )[0]

    class StatusWrapper:
        def __init__(self, stdout=None, stderr=None, success=None) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.success = success

    def run_impl(command, cwd=None):
        env = copy.copy(os.environ)
        # Always prioritize our own copy of Polygraphy over anything in the venv.
        env["PYTHONPATH"] = ROOT_DIR + os.pathsep + VENV_PYTHONPATH

        print(f"Running command: {' '.join(command)}")

        status = StatusWrapper()
        if "pip" in command:
            virtualenv.run(command, cwd=cwd)
            status.success = True
        elif command[0] == "polygraphy":
            sr_status = script_runner.run(*command, cwd=cwd, env=env)
            status.stdout = sr_status.stdout
            status.success = sr_status.success
        else:
            sp_status = sp.run(
                command, cwd=cwd, env=env, stdout=sp.PIPE, stderr=sp.PIPE
            )

            def try_decode(inp):
                try:
                    return inp.decode()
                except UnicodeDecodeError:
                    return inp

            status.stdout = try_decode(sp_status.stdout)
            status.stderr = try_decode(sp_status.stderr)
            status.success = sp_status.returncode == 0

        return status

    return run_impl


@pytest.fixture()
def check_warnings_on_runner_impl_methods():
    """
    Fixture that ensures warnings are emitted when `_impl` methods of runners are called.
    """

    def check(runner):
        import contextlib
        import io

        import numpy as np

        from polygraphy.datatype import DataType

        outfile = io.StringIO()
        with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
            runner.activate()
            # Check that NumPy dtypes are still returned by default
            metadata = runner.get_input_metadata()
            for dtype, _ in metadata.values():
                assert isinstance(dtype, np.dtype)

            metadata = runner.get_input_metadata(use_numpy_dtypes=False)
            runner.infer(
                {
                    name: np.ones(shape, dtype=DataType.to_dtype(dtype, "numpy"))
                    for name, (dtype, shape) in metadata.items()
                }
            )
            runner.deactivate()

            outfile.seek(0)
            out = outfile.read()

            def check_warning(method, warning_expected):
                assert (
                    f"Calling '{type(runner).__name__}.{method}_impl()' directly is not recommended. Please use '{method}()' instead."
                    in out
                ) == warning_expected

            check_warning("get_input_metadata", warning_expected=False)
            check_warning("activate", warning_expected=False)
            check_warning("infer", warning_expected=False)
            check_warning("deactivate", warning_expected=False)

            runner.activate_impl()
            metadata = runner.get_input_metadata_impl()
            runner.infer_impl(
                {
                    name: np.ones(
                        shape,
                        dtype=DataType.to_dtype(DataType.from_dtype(dtype), "numpy"),
                    )
                    for name, (dtype, shape) in metadata.items()
                }
            )
            runner.deactivate_impl()

            outfile.seek(0)
            out = outfile.read()
            print(out)

            check_warning("get_input_metadata", warning_expected=True)
            check_warning("activate", warning_expected=True)
            check_warning("infer", warning_expected=True)
            check_warning("deactivate", warning_expected=True)

    return check


@pytest.fixture()
def check_warnings_on_loader_impl_methods():
    """
    Fixture that ensures warnings are emitted when loader `_impl` methods are called.
    """

    def check(loader):
        import contextlib
        import io

        outfile = io.StringIO()
        with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
            warning_msg = f"Calling '{type(loader).__name__}.call_impl()' directly is not recommended. Please use '__call__()' instead."
            loader.__call__()

            outfile.seek(0)
            out = outfile.read()

            assert warning_msg not in out

            loader.call_impl()

            outfile.seek(0)
            out = outfile.read()
            print(out)

            assert warning_msg in out

    return check


@pytest.fixture()
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Fixture has not been updated to work on Windows",
)
def nvinfer_lean_path():
    lean_library_name = ctypes.util.find_library("nvinfer_lean")
    for dirname in os.environ.get("LD_LIBRARY_PATH", "").split(os.path.pathsep) + [
        "/usr/lib/x86_64-linux-gnu"
    ]:
        path = os.path.join(dirname, lean_library_name)
        if os.path.exists(path):
            return path

    assert False, "Could not find nvinfer_lean!"


@pytest.fixture()
def tmp_python_log_file(tmp_path):
    # backup original logging configuration
    orig_handlers = logging.root.handlers[:]
    orig_level = logging.root.level
    logging.root.handlers = []
    tmp_log_file = tmp_path / "test.log"
    # setup logging to file
    logging.basicConfig(filename=tmp_log_file, level=0)
    try:
        yield tmp_log_file
    finally:
        # revert back original configuration
        logging.root.handlers = orig_handlers
        logging.root.level = orig_level
