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

.ONESHELL: # Run all rules in the same shell to make it easier to run tests with environment variables
.PHONY: test_venv test leak_check clean build install docs

NPROC ?= 8

BUILD_DIR := build
TESTS_DIR := tests

EXTRA_PYTEST_OPTS ?=
PYTEST_OPTS := -v -x --durations=5 --failed-first --new-first --script-launch-mode=subprocess $(EXTRA_PYTEST_OPTS)

# Tests also check that docs can build
test: docs install
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH}
	export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1
	export CUDA_MODULE_LOADING=LAZY
    # Some tests need to be run serially - we annotate those with a `serial` marker.
	python3 -m pytest $(TESTS_DIR) -m "serial" $(PYTEST_OPTS) && \
		python3 -m pytest $(TESTS_DIR) -n $(NPROC) --dist=loadscope -m "not serial" $(PYTEST_OPTS)

leak_check:
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH}
	export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1
	export PYTHONMALLOC=malloc
	valgrind --leak-check=full python3 -m pytest $(TESTS_DIR) -v --durations=5 2>&1 | tee leak-check.log

clean:
	rm -rf dist/ $(BUILD_DIR)/ polygraphy.egg-info/

build:
	python3 setup.py bdist_wheel

install_deps: build
	- python3 -m pip install colored wheel

install: install_deps
	- python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

docs:
	mkdir -p $(BUILD_DIR)/docs
	python3 `which sphinx-build` docs $(BUILD_DIR)/docs/ -j $(NPROC) -W
