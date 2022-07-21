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

BUILD_DIR := build

clean:
	- rm -rf dist/ $(BUILD_DIR)/ polygraphy_trtexec.egg-info/ polygraphy_trtexec/__pycache__/ 

build: clean
	- python3 setup.py bdist_wheel

install_deps: build
	- python3 -m pip install colored wheel

install: install_deps
	- python3 -m pip install --force-reinstall --no-deps $(CURDIR)/dist/*.whl

test: install
	python3 -m pytest
