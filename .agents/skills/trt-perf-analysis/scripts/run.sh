#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

if [ $# -lt 1 ]; then
    printf '%s\n' "usage: run.sh <python-script> [args...]" >&2
    exit 2
fi

SCRIPT_PATH=$1
shift

try_python() {
    candidate=$1
    launcher_arg=$2
    shift 2

    if [ -z "$candidate" ]; then
        return 1
    fi
    if ! command -v "$candidate" >/dev/null 2>&1; then
        return 1
    fi

    if [ -n "$launcher_arg" ]; then
        "$candidate" "$launcher_arg" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1 || return 1
        exec "$candidate" "$launcher_arg" "$SCRIPT_PATH" "$@"
    fi

    "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1 || return 1
    exec "$candidate" "$SCRIPT_PATH" "$@"
}

if [ -n "${SKILL_PYTHON:-}" ]; then
    try_python "$SKILL_PYTHON" "" "$@"
fi
if [ -n "${PYTHON:-}" ]; then
    try_python "$PYTHON" "" "$@"
fi
if [ -n "${PYTHON3:-}" ]; then
    try_python "$PYTHON3" "" "$@"
fi

try_python python3 "" "$@"
try_python python "" "$@"
try_python py -3 "$@"
try_python python3.12 "" "$@"
try_python python3.11 "" "$@"
try_python python3.10 "" "$@"
try_python python3.9 "" "$@"
try_python python3.8 "" "$@"

printf '%s\n' "error: unable to find Python 3.8+ on PATH. Set SKILL_PYTHON to a Python executable." >&2
exit 127
