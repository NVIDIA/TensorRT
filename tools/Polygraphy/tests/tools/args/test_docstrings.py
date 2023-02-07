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
import inspect
import re

import pytest
import importlib

args_mod = importlib.import_module("polygraphy.tools.args")

from polygraphy.tools.args.base import BaseArgs

ARG_CLASSES = [cls for cls in args_mod.__dict__.values() if inspect.isclass(cls) and issubclass(cls, BaseArgs)]

USES_DEP_PAT = re.compile(r"self.arg_groups\[(.*?)\]")


class TestDocStrings:
    @pytest.mark.parametrize("arg_group_type", ARG_CLASSES)
    def test_docstrings_document_dependencies(self, arg_group_type):
        code = inspect.getsource(arg_group_type)
        deps = set(USES_DEP_PAT.findall(code))

        docstring = arg_group_type.__doc__
        doc_lines = [line.strip() for line in docstring.splitlines() if line.strip()]

        assert (
            ":" in doc_lines[0]
        ), f"Incorrect format for first line of docstring: {doc_lines[0]}. Expected 'Title: Description'"

        documented_deps = set()
        if len(doc_lines) > 1 and "Depends" in doc_lines[1]:
            assert (
                doc_lines[1] == "Depends on:"
            ), f"Incorrect format for second line of docstring: {doc_lines[1]}. Expected start of 'Depends on:' section."

            for line in doc_lines[2:]:
                if not line.startswith("-"):
                    continue
                documented_deps.add(line.lstrip("-").partition(":")[0].strip())

        assert documented_deps == deps, "Documented dependencies do not match actual dependencies"
