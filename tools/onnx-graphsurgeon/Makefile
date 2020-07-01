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

test:
	python3 -m pytest tests/ -v -x --pdb

clean:
	rm -rf dist/ build/ onnx_graphsurgeon.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_venv: build
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

install: build
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl --user
