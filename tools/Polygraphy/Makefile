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

.ONESHELL: # Run all rules in the same shell to make it easier to run tests with environment variables
.PHONY: test_venv test test_multigpu leak_check clean build install docs

NPROC ?= 8
MPIRUN ?= mpirun
NCCL_LIB ?= /usr/lib/x86_64-linux-gnu/libnccl.so.2

BUILD_DIR := build
TESTS_DIR := tests

RUN_ALL_TESTS ?= 0
EXTRA_PYTEST_OPTS ?=
PYTEST_OPTS := $(EXTRA_PYTEST_OPTS) -v --durations=15 --failed-first --new-first --script-launch-mode=subprocess

EXTRA_PYTEST_MARKERS :=

ifeq ($(RUN_ALL_TESTS),0)
    EXTRA_PYTEST_MARKERS +=  and not slow
endif

# Tests also check that docs can build
test: docs install
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH}
	export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1
	export CUDA_MODULE_LOADING=LAZY
    # Some tests need to be run serially - we annotate those with a `serial` marker.
	python3 -m pytest $(TESTS_DIR) -m "serial and not multigpu $(EXTRA_PYTEST_MARKERS)" $(PYTEST_OPTS) && \
		python3 -m pytest $(TESTS_DIR) -n $(NPROC) --dist=loadscope -m "not serial and not multigpu $(EXTRA_PYTEST_MARKERS)" $(PYTEST_OPTS)

# Run tests that require multiple GPUs (multigpu marker). Intended for
# CI jobs with a multi-GPU runner; these tests are skipped automatically
# when fewer than 2 GPUs are detected.
test_multigpu: docs install
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH}
	export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1
	export CUDA_MODULE_LOADING=LAZY
	export OMPI_ALLOW_RUN_AS_ROOT=1
	export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
	export LD_PRELOAD=$(NCCL_LIB)
	$(MPIRUN) -n 2 python3 -m pytest -s $(TESTS_DIR) -m "multigpu" $(PYTEST_OPTS)

leak_check:
	export PYTHONPATH=$(CURDIR):$${PYTHONPATH}
	export POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS=1
	export PYTHONMALLOC=malloc
	valgrind --leak-check=full python3 -m pytest $(TESTS_DIR) -v --durations=5 2>&1 | tee leak-check.log

clean:
	rm -rf dist/ $(BUILD_DIR)/ polygraphy.egg-info/

build:
	@if ls dist/polygraphy-*.whl 1>/dev/null 2>&1; then \
		echo "Polygraphy wheel already exists in dist/, skipping build"; \
	else \
		rm -rf dist/ $(BUILD_DIR)/ polygraphy.egg-info/; \
		python3 -c "import build" 2>/dev/null || python3 -m pip install build; \
		python3 -m build --wheel; \
	fi

install_deps: build
	- python3 -m pip install colored wheel

install: install_deps
	- python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

docs: build
	mkdir -p $(BUILD_DIR)/docs
	python3 `which sphinx-build` docs $(BUILD_DIR)/docs/ -j $(NPROC) -W
