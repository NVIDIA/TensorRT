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
MEMBER_PAT = re.compile(r"self.(.*?)[ ,.\[]")


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

    # Checks that all members set by `parse` are documented.
    #
    # Note that this is a fairly dumb test in that it just looks for regexs matching `self.(.*)` to check
    # whether those members are documented in `parse_impl`. `parse_impl` typically only *sets* members
    # and doesn't use most members of the class, so this approach is generally ok.
    #
    # There are cases where we may not want to document some members, e.g. if they are deprecated.
    # In those cases, you can prefix the member with a `_` and it will be ignored by
    @pytest.mark.parametrize("arg_group_type", ARG_CLASSES)
    def test_parse_docstring_documents_populated_members(self, arg_group_type):
        code = inspect.getsource(arg_group_type.parse_impl)

        def should_include_member(member):
            if not member:
                return False

            EXCLUDE_PREFIXES = ["arg_groups", "_"]
            if any(member.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
                return False

            return True

        members = {member for member in MEMBER_PAT.findall(code) if should_include_member(member)}

        docstring = arg_group_type.parse_impl.__doc__
        if docstring is None:
            pytest.skip("parse_impl not required by this argument group")

        doc_lines = [line.strip() for line in docstring.splitlines() if line.strip()]

        attributes_doc_start = doc_lines.index("Attributes:")
        assert attributes_doc_start >= 0, "Expected parse_impl docstring to contain an `Attributes:` section."

        doc_lines = doc_lines[attributes_doc_start + 1 :]

        documented_members = set([line.strip().split()[0] for line in doc_lines])

        undocumented_members = members - documented_members
        assert not undocumented_members, "Some members are not documented!"
