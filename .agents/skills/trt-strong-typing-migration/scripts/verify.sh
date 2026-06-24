#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify the `migrate.py` helper on an inline weakly-typed sample.
# Run from anywhere; resolves paths relative to this script.
#
# Usage:
#   bash scripts/verify.sh         # run the verification end-to-end
#   bash scripts/verify.sh --keep  # keep the temp workspace for inspection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATE="${SCRIPT_DIR}/migrate.py"

if [[ ! -f "${MIGRATE}" ]]; then
    echo "ERROR: migrate.py not found at ${MIGRATE}" >&2
    exit 1
fi

KEEP=0
if [[ "${1:-}" == "--keep" ]]; then
    KEEP=1
fi

WORK="$(mktemp -d -t trt-w2s-verify-XXXXXX)"

cleanup_workspace() {
    if [[ ${KEEP} -eq 1 ]]; then
        echo "Kept workspace at ${WORK}"
    else
        rm -rf "${WORK}"
    fi
}
trap cleanup_workspace EXIT

SAMPLE="${WORK}/sample_build.py"

# A representative weakly-typed Python builder. Contains every transform path
# migrate.py is expected to handle: EXPLICIT_BATCH network flag, set_flag for
# FP16/INT8 (removed) plus TF32/REFIT (preserved), per-layer precision overrides,
# and the platform_has_fast_* gating that becomes dead code after migration.
cat > "${SAMPLE}" <<'PY'
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)   # kept in TRT 11 (orthogonal to typing); must survive
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.REFIT)  # not a precision flag; must survive

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open("model.onnx", "rb") as f:
        parser.parse(f.read())

    last = network.get_layer(network.num_layers - 1)
    last.precision = trt.float16
    last.set_output_type(0, trt.float16)

    plan = builder.build_serialized_network(network, config)
    return plan
PY

echo "[verify] sample written to ${SAMPLE}"

echo "[verify] dry-run diff (exit 1 = changes pending, expected here):"
python3 "${MIGRATE}" "${SAMPLE}" || true

echo "[verify] writing rewrite in place:"
python3 "${MIGRATE}" "${SAMPLE}" --write

# Assertions on the rewritten file.
fail=0
assert_present() {
    local needle="$1"
    if ! grep -qF "${needle}" "${SAMPLE}"; then
        echo "ASSERT FAIL: expected to find: ${needle}" >&2
        fail=1
    fi
}
assert_absent() {
    local needle="$1"
    if grep -qF "${needle}" "${SAMPLE}"; then
        echo "ASSERT FAIL: expected to be removed: ${needle}" >&2
        fail=1
    fi
}

assert_present "NetworkDefinitionCreationFlag.STRONGLY_TYPED"
assert_absent  "NetworkDefinitionCreationFlag.EXPLICIT_BATCH"
assert_absent  "BuilderFlag.FP16"
assert_absent  "BuilderFlag.INT8"
assert_present "BuilderFlag.TF32"        # kTF32 is kept in TRT 11; must NOT be stripped
assert_present "BuilderFlag.REFIT"       # non-precision flag must remain
assert_absent  ".precision = trt.float16"
assert_absent  "set_output_type(0, trt.float16)"

# Syntactic validity.
python3 -c "import ast, sys; ast.parse(open('${SAMPLE}').read())" || {
    echo "ASSERT FAIL: rewritten file is not valid Python" >&2
    fail=1
}

if [[ ${fail} -ne 0 ]]; then
    echo "[verify] FAIL — see assertions above. Workspace: ${WORK}"
    KEEP=1
    exit 1
fi

echo "[verify] OK — migrate.py produced a well-formed strongly-typed rewrite"
