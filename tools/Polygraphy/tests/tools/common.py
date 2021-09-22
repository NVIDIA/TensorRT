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
import subprocess as sp
import sys
import os

from polygraphy.logger import G_LOGGER

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
BIN_DIR = os.path.join(ROOT_DIR, "bin")
polygraphy = os.path.join(BIN_DIR, "polygraphy")


def check_subprocess(status):
    if status.returncode:
        G_LOGGER.critical(status.stdout + status.stderr)


def run_polygraphy(additional_opts=[], *args, **kwargs):
    cmd = [sys.executable, polygraphy] + additional_opts
    print("Running command: {:}".format(" ".join(cmd)))
    status = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, *args, **kwargs)
    status.stdout = status.stdout.decode()
    status.stderr = status.stderr.decode()
    print(status.stdout)
    print(status.stderr)
    check_subprocess(status)
    return status


def run_subtool(subtool, additional_opts, disable_verbose=False, *args, **kwargs):
    opts = [subtool]
    opts += additional_opts
    if not disable_verbose:
        opts += ["-vvvvv"]
    return run_polygraphy(opts, *args, **kwargs)


def run_polygraphy_run(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("run", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_convert(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("convert", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_inspect(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("inspect", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_precision(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("precision", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_surgeon(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("surgeon", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_template(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("template", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_debug(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("debug", additional_opts, disable_verbose, *args, **kwargs)


def run_polygraphy_data(additional_opts=[], disable_verbose=False, *args, **kwargs):
    return run_subtool("data", additional_opts, disable_verbose, *args, **kwargs)
