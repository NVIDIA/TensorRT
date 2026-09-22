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
import ast
import glob
import os

import pytest

from tests.helper import ROOT_DIR

"""
Argument groups only parse strings and emit script *text*, so they must not use lazily
imported modules. Doing so turns a lazy dependency into an up-front requirement for the CLI
and, since `polygraphy run` always `exec`s the script it generates, can emit code that is not
even valid Python.

`tests/mod/test_dependencies.py` covers this dynamically, but only for `--help`, which never
reaches `parse_impl`. This test scans the source instead so the check is exhaustive and cheap.
"""

TOOLS_DIR = os.path.join(ROOT_DIR, "polygraphy", "tools")
ARGS_DIR = os.path.join(TOOLS_DIR, "args")

# `BaseArgs` methods that run while the CLI is parsing arguments or generating a script.
# These names are unique to the `BaseArgs` protocol, so matching on them also covers the
# argument groups that are defined alongside their subtool rather than under `tools/args`.
CLI_TIME_METHODS = {"add_parser_args_impl", "parse_impl", "add_to_script_impl"}

TOOL_MODULE_PATHS = sorted(
    glob.glob(os.path.join(TOOLS_DIR, "**", "*.py"), recursive=True)
)


def find_lazy_imports(tree):
    """
    Returns a mapping of module-level names bound to lazily imported modules to the name of
    the module they import.
    """
    lazy_imports = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue

        func = node.value.func
        func_name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )

        if func_name == "lazy_import_trt":
            module = "tensorrt"
        elif func_name == "lazy_import" and node.value.args:
            arg = node.value.args[0]
            if not isinstance(arg, ast.Constant):
                continue
            # Any version constraint, e.g. `onnx>=1.18`, is left in place since this is
            # only ever displayed.
            module = arg.value
        else:
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                lazy_imports[target.id] = module

    return lazy_imports


def find_lazy_module_uses(tree, lazy_imports, predicate):
    """
    Finds attribute accesses on lazily imported modules, retaining only those in functions
    for which `predicate(func_name)` returns True.
    """
    uses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not predicate(node.name):
            continue

        for subnode in ast.walk(node):
            if (
                isinstance(subnode, ast.Attribute)
                and isinstance(subnode.value, ast.Name)
                and subnode.value.id in lazy_imports
            ):
                uses.append((subnode, node.name))
    return uses


def make_message(path, uses, lazy_imports, explanation):
    violations = [
        f"{os.path.relpath(path, ROOT_DIR)}:{node.lineno}: `{node.value.id}.{node.attr}` "
        f"in `{func_name}` requires `{lazy_imports[node.value.id]}`"
        for node, func_name in uses
    ]
    return (
        f"{explanation}\n"
        "Emit the attribute as inline script text instead (see `make_trt_enum_val`). Violations:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "path",
    TOOL_MODULE_PATHS,
    ids=[os.path.relpath(path, TOOLS_DIR) for path in TOOL_MODULE_PATHS],
)
def test_arg_groups_do_not_use_lazy_modules_at_cli_time(path):
    # Lazy imports of Polygraphy submodules are included here: importing them is harmless,
    # but *calling into* them while parsing arguments can pull in their own dependencies.
    with open(path, "r") as f:
        tree = ast.parse(f.read())

    lazy_imports = find_lazy_imports(tree)
    if not lazy_imports:
        return

    uses = find_lazy_module_uses(
        tree, lazy_imports, lambda name: name in CLI_TIME_METHODS
    )
    assert not uses, make_message(
        path,
        uses,
        lazy_imports,
        "Argument groups must not use lazily imported modules while parsing arguments or "
        "generating scripts since that makes the module an up-front dependency of the CLI.",
    )


@pytest.mark.parametrize(
    "path",
    [path for path in TOOL_MODULE_PATHS if path.startswith(ARGS_DIR + os.sep)],
    ids=[
        os.path.relpath(path, ARGS_DIR)
        for path in TOOL_MODULE_PATHS
        if path.startswith(ARGS_DIR + os.sep)
    ],
)
def test_arg_modules_do_not_use_external_modules(path):
    # Everything under `tools/args` exists only to build scripts, so it should never touch an
    # external module at all. This is stricter than the check above and additionally catches
    # helpers that an argument group calls into.
    with open(path, "r") as f:
        tree = ast.parse(f.read())

    lazy_imports = {
        name: module
        for name, module in find_lazy_imports(tree).items()
        if not module.startswith("polygraphy")
    }
    if not lazy_imports:
        return

    uses = find_lazy_module_uses(tree, lazy_imports, lambda name: True)
    assert not uses, make_message(
        path,
        uses,
        lazy_imports,
        "Modules under `tools/args` must not use external modules anywhere since that makes "
        "the module an up-front dependency of the CLI.",
    )
