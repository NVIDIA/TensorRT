#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AST-based migrator for weakly-typed TensorRT Python builders.

Rewrites:

  - `builder.create_network(...)` -> uses NetworkDefinitionCreationFlag.STRONGLY_TYPED
  - `config.set_flag(trt.BuilderFlag.{FP16,BF16,INT8,FP8})` -> removed
  - `layer.precision = ...` and `layer.set_output_type(...)` -> removed

The rewrites are intentionally conservative: ambiguous patterns are skipped and
reported so the user can resolve them manually. Non-precision flags (REFIT,
SPARSE_WEIGHTS, TF32, etc.) are preserved — kTF32 is kept in TRT 11 and is
orthogonal to typing.

Usage:
    python3 migrate.py path/to/file.py          # dry-run, print unified diff
    python3 migrate.py path/to/file.py --write  # rewrite in place
    python3 migrate.py path/to/dir/  --write    # recurse over .py files
"""
from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path
from typing import Iterable

# Precision-hint BuilderFlag attribute names that must be removed. NOTE: TF32 is
# deliberately NOT here — kTF32 is kept in TRT 11 (orthogonal to typing), like REFIT.
PRECISION_FLAGS = frozenset({"FP16", "BF16", "INT8", "FP8"})

# NetworkDefinitionCreationFlag attribute names that map to STRONGLY_TYPED.
WEAK_NETWORK_FLAGS = frozenset({"EXPLICIT_BATCH"})


def _is_attr(node: ast.AST, *, attr_chain: tuple[str, ...]) -> bool:
    """Return True if `node` is an attribute access matching the trailing chain.

    Walks `node.attr -> node.value.attr -> ...` and compares to attr_chain
    in reverse. The leading element (e.g. module / object name) is not checked,
    so this matches `trt.BuilderFlag.FP16`, `tensorrt.BuilderFlag.FP16`, or any
    aliased import.
    """
    cur = node
    for expected in reversed(attr_chain):
        if not isinstance(cur, ast.Attribute):
            return False
        if cur.attr != expected:
            return False
        cur = cur.value
    return True


def _builder_flag_name(node: ast.AST) -> str | None:
    """If `node` is `<anything>.BuilderFlag.<NAME>`, return NAME."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if node.value.attr == "BuilderFlag":
            return node.attr
    return None


def _network_flag_name(node: ast.AST) -> str | None:
    """If `node` is `<anything>.NetworkDefinitionCreationFlag.<NAME>`, return NAME."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if node.value.attr == "NetworkDefinitionCreationFlag":
            return node.attr
    return None


def _strongly_typed_arg() -> ast.expr:
    """Build the AST for `1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)`.

    Uses bare `trt.` qualifier — matches the canonical import alias. Callers
    whose code uses `tensorrt.` directly will get a working but visually
    inconsistent line; that is acceptable as a one-line manual cleanup.
    """
    return ast.BinOp(
        left=ast.Constant(value=1),
        op=ast.LShift(),
        right=ast.Call(
            func=ast.Name(id="int", ctx=ast.Load()),
            args=[
                ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="trt", ctx=ast.Load()),
                        attr="NetworkDefinitionCreationFlag",
                        ctx=ast.Load(),
                    ),
                    attr="STRONGLY_TYPED",
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        ),
    )


class Migrator(ast.NodeTransformer):
    def __init__(self) -> None:
        self.removed_flag_calls = 0
        self.removed_precision_assigns = 0
        self.removed_set_output_type = 0
        self.rewrote_create_network = 0
        self.skipped: list[str] = []

    # ---- create_network rewrites ----------------------------------------

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "create_network":
            # Replace the single positional argument with the STRONGLY_TYPED flag.
            if len(node.args) == 0:
                # No positional arg: the flag may be passed as `flags=` keyword.
                # Blindly appending a positional arg would collide with it
                # (TypeError: got multiple values for argument 'flags').
                flags_kw = next((kw for kw in node.keywords if kw.arg == "flags"), None)
                if flags_kw is not None:
                    if self._is_weak_network_arg(flags_kw.value):
                        flags_kw.value = _strongly_typed_arg()
                        self.rewrote_create_network += 1
                    elif self._already_strongly_typed(flags_kw.value):
                        pass  # idempotent
                    else:
                        self.skipped.append(
                            f"create_network at line {node.lineno}: unrecognized flags= "
                            "argument shape, review manually"
                        )
                elif not node.keywords:
                    # Truly empty call create_network(): defaults to weak typing.
                    node.args = [_strongly_typed_arg()]
                    self.rewrote_create_network += 1
                else:
                    self.skipped.append(
                        f"create_network at line {node.lineno}: keyword args present "
                        "without flags=, review manually"
                    )
            elif len(node.args) == 1:
                arg = node.args[0]
                if self._is_weak_network_arg(arg):
                    node.args[0] = _strongly_typed_arg()
                    self.rewrote_create_network += 1
                elif self._already_strongly_typed(arg):
                    pass  # idempotent
                else:
                    self.skipped.append(
                        f"create_network at line {node.lineno}: unrecognized argument shape, "
                        "review manually"
                    )
        return node

    @staticmethod
    def _is_weak_network_arg(arg: ast.expr) -> bool:
        # Match `0`
        if isinstance(arg, ast.Constant) and arg.value == 0:
            return True
        # Match `1 << int(...EXPLICIT_BATCH)` (any weak-network flag name)
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.LShift):
            inner = arg.right
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "int":
                if inner.args:
                    name = _network_flag_name(inner.args[0])
                    if name in WEAK_NETWORK_FLAGS:
                        return True
        return False

    @staticmethod
    def _already_strongly_typed(arg: ast.expr) -> bool:
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.LShift):
            inner = arg.right
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "int":
                if inner.args and _network_flag_name(inner.args[0]) == "STRONGLY_TYPED":
                    return True
        return False

    # ---- set_flag / precision assignment removals -----------------------

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        # `config.set_flag(trt.BuilderFlag.FP16)` as an Expr statement.
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "set_flag":
                if call.args:
                    flag = _builder_flag_name(call.args[0])
                    if flag in PRECISION_FLAGS:
                        self.removed_flag_calls += 1
                        return None
            # `layer.set_output_type(0, trt.float16)` — drop unconditionally.
            if isinstance(call.func, ast.Attribute) and call.func.attr == "set_output_type":
                self.removed_set_output_type += 1
                return None
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        # `something.precision = trt.float16` (or any value) — drop.
        self.generic_visit(node)
        if len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Attribute) and tgt.attr == "precision":
                self.removed_precision_assigns += 1
                return None
        return node

    # ---- if-statement cleanup ------------------------------------------

    def visit_If(self, node: ast.If) -> ast.AST | None:
        """Drop `if builder.platform_has_fast_{fp16,int8,bf16}:` blocks that become empty."""
        self.generic_visit(node)
        test = node.test
        gating_attrs = {"platform_has_fast_fp16", "platform_has_fast_int8",
                        "platform_has_fast_bf16", "platform_has_fast_fp8"}
        if isinstance(test, ast.Attribute) and test.attr in gating_attrs:
            # If body becomes empty (because set_flag was the only statement) and
            # there is no else clause, the whole if is dead.
            if not node.body and not node.orelse:
                return None
            if not node.body:
                # Body emptied but orelse remains — replace with the orelse block.
                return node.orelse if len(node.orelse) > 1 else (
                    node.orelse[0] if node.orelse else None
                )
        return node


def _process_source(src: str) -> tuple[str, Migrator]:
    tree = ast.parse(src)
    m = Migrator()
    new_tree = m.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree), m


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p


def _process_file(path: Path, *, write: bool) -> bool:
    """Return True iff the file needs (or got) a migration.

    Keyed on the migrator's transform counts, not on text equality: `ast.unparse`
    reformats and drops comments on every round-trip, so a file that needs no
    migration would otherwise look "changed" and have its comments stripped.
    """
    original = path.read_text()
    try:
        new_src, m = _process_source(original)
    except SyntaxError as e:
        print(f"[skip] {path}: syntax error: {e}", file=sys.stderr)
        return False

    migrated = (m.rewrote_create_network + m.removed_flag_calls
                + m.removed_precision_assigns + m.removed_set_output_type) > 0
    if not migrated:
        for note in m.skipped:
            print(f"[note] {path}: {note}", file=sys.stderr)
        return False

    if write:
        path.write_text(new_src)
        print(
            f"[write] {path}: create_network={m.rewrote_create_network} "
            f"set_flag={m.removed_flag_calls} "
            f"precision_assign={m.removed_precision_assigns} "
            f"set_output_type={m.removed_set_output_type}"
        )
    else:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new_src.splitlines(keepends=True),
            fromfile=f"{path} (weakly typed)",
            tofile=f"{path} (strongly typed)",
        )
        sys.stdout.writelines(diff)
    for note in m.skipped:
        print(f"[note] {path}: {note}", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to process.")
    parser.add_argument("--write", action="store_true",
                        help="Rewrite files in place. Default is dry-run with unified diff.")
    args = parser.parse_args()

    any_changed = False
    for path in _iter_files(args.paths):
        if _process_file(path, write=args.write):
            any_changed = True

    # In dry-run, exit non-zero when changes are pending so callers/CI can gate
    # on it (like `black --check`). `--write` applies the changes and exits 0.
    return 1 if any_changed and not args.write else 0


if __name__ == "__main__":
    sys.exit(main())
