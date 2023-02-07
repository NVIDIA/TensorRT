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

import pytest
from polygraphy.exception import PolygraphyInternalException
from polygraphy.tools.script import Script, inline, make_invocable, make_invocable_if_nondefault


def make_test_string():
    return Script.String("test")


class TestScript:
    @pytest.mark.parametrize(
        "func",
        [
            lambda _: inline(make_test_string()),
            lambda s: s.add_loader(make_test_string(), make_test_string()),
            lambda s: s.add_runner(make_test_string()),
            lambda s: s.append_preimport(make_test_string()),
            lambda s: s.append_suffix(make_test_string()),
            lambda s: s.set_data_loader(make_test_string()),
        ],
    )
    def test_add_funcs_fail_on_unsafe(self, func):
        script = Script()
        with pytest.raises(PolygraphyInternalException, match="was not checked for safety"):
            func(script)

    @pytest.mark.parametrize(
        "case, expected",
        [
            ("should_become_raw", "'should_become_raw'"),
            ("parens))", r"'parens))'"),
            ("'squotes'", "\"'squotes'\""),
            ('"dquotes"', "'\"dquotes\"'"),
            (r"braces{}{})", r"'braces{}{})'"),
            ("commas, ,", r"'commas, ,'"),
            ("escape_quote_with_backslash'", '"escape_quote_with_backslash\'"'),
            ("unterm_in_quotes_ok))", r"'unterm_in_quotes_ok))'"),
        ],
    )
    def test_non_inlined_strings_escaped(self, case, expected):
        out = make_invocable("Dummy", case, x=case)
        ex_out = f"Dummy({expected}, x={expected})"
        assert out.unwrap() == ex_out

    def test_invoke_none_args(self):
        assert make_invocable("Dummy", None).unwrap() == "Dummy(None)"
        assert make_invocable("Dummy", x=None).unwrap() == "Dummy()"

    def test_invoke_if_nondefault_none_args(self):
        assert make_invocable_if_nondefault("Dummy", None) is None
        assert make_invocable_if_nondefault("Dummy", x=None) is None

    def test_lazy_import(self):
        script = Script()
        script.add_import("numpy", imp_as="np")
        assert "np = mod.lazy_import('numpy')" in str(script)

    def test_import_from(self):
        script = Script()
        script.add_import("example", frm="mod")
        assert "from mod import example" in str(script)

    def test_import_duplicate_froms(self):
        script = Script()
        script.add_import("example", frm="mod")
        script.add_import("also", frm="mod")
        assert "from mod import also, example" in str(script)

    def test_import_duplicate_froms_with_as(self):
        script = Script()
        script.add_import("example", frm="mod")
        script.add_import("also", frm="mod", imp_as="temp")
        assert "from mod import also as temp, example" in str(script)
