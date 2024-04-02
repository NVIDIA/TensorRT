#!/bin/sh
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

#
# TREx installation script.
#
# Usage:
#   $ source install.sh [--venv]
#


usage() {
    echo "Usage:"
    echo "  $ source $BASH_SOURCE [--venv]"
    OK=0
}

invalid_arg() {
    echo "Error: $ARG1 is not a valid argument." >&2
    usage
}

too_many_args() {
    echo "Error: Too many arguments"
    usage
}

install_venv() {
    VENV="env_trex"
    sudo apt install --yes virtualenv
    python3 -m virtualenv $VENV
    source ./$VENV/bin/activate
}

warn_if_trt_not_installed() {
    # Check if the tensorrt Python package is installed
    python3 -c "import tensorrt" &> /dev/null

    status=$?
    if [ $status -eq 1 ]; then
        echo
        echo "Warning: Python package tensorrt is not installed!" >&2
        echo "Package tensorrt is required by some of the notebooks and scripts." >&2
    fi
}

install_trex_core() { python3 -m pip install -e .; }

install_trex_full() { python3 -m pip install -e .[notebook]; }

install() {
    sudo apt install --yes graphviz
    INSTALL_TYPE=$1
    if [ $INSTALL_TYPE = "full" ]; then
        install_trex_full
    else
        install_trex_core
    fi
    warn_if_trt_not_installed
}

parse_args() {
    if [ $NARGS -gt 1 ]; then
        too_many_args
        return
    fi
    case $ARG1 in
    ("-h" | "--h" | "-help" | "--help")
        usage
        ;;
    ("--venv")
        install_venv
        ;;
    ("-c" | "--core")
        INSTALLATION_TYPE="core"
        ;;
    ("-f" | "--full")
        INSTALLATION_TYPE="full"
        ;;
    ("")
        ;;
    (*)
        invalid_arg
        ;;
    esac
}


OK=1
NARGS=$#
ARG1=$1
INSTALLATION_TYPE="full"
parse_args

if [ $OK -eq 1 ]; then
    install $INSTALLATION_TYPE
fi

if [ $OK -eq 0 ]; then
    echo "Installation aborted."
fi
