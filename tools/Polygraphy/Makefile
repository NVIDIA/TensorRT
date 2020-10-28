#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

.PHONY: test test_noparallel clean build install_venv install docs

# Tests also check that docs can build
test: docs
	python3 -m pytest tests/ -v -x -nauto --dist=loadscope

test_noparallel: docs
	python3 -m pytest tests/ -v -x

clean:
	rm -rf dist/ build/ polygraphy.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_common: build
	-python3 -m pip install colored

install_venv: install_common
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

install: install_common
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl --user

docs:
	mkdir -p build/docs
	sphinx-build docs build/docs/ -j auto -W
