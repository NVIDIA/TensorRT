# trt-strong-typing-migration helper scripts

Scripts that automate the mechanical parts of the SKILL.md walkthrough.

## `migrate.py`

AST-based rewriter for Python TensorRT builder code. Detects the
weakly-typed builder pattern (`create_network` + `set_flag(BuilderFlag.FP16/...)`)
and rewrites it to the strongly-typed form.

```bash
# Show what would change (dry-run): prints a unified diff; exits 1 if changes are pending, 0 if nothing to do.
python3 migrate.py path/to/build.py

# Rewrite the file in place:
python3 migrate.py path/to/build.py --write

# Process every .py file in a directory tree:
python3 migrate.py path/to/project/ --write
```

The script uses `ast.NodeTransformer` so it understands call shapes, not just
text — it correctly handles `set_flag` accessed via aliased imports, multi-flag
construction in `create_network`, and per-layer `precision`/`set_output_type`
assignments. Unrelated logic and docstrings are left intact, but note: because
the rewrite round-trips through `ast.unparse`, **regular `#` comments and the
original formatting are not preserved**. Review the dry-run diff before
`--write`, and re-add any comments you need afterward.

### What it changes

| Before | After |
|--------|-------|
| `builder.create_network(0)` | `builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))` |
| `builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))` | `builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))` |
| `config.set_flag(trt.BuilderFlag.FP16 / BF16 / INT8 / FP8)` | (line removed) |
| `layer.precision = trt.float16` | (statement removed) |
| `layer.set_output_type(0, trt.float16)` | (statement removed) |

### What it does NOT change

- `set_flag` calls for non-precision-hint flags (`REFIT`, `SPARSE_WEIGHTS`,
  `DISABLE_TIMING_CACHE`, and `TF32`). These are orthogonal to typing — `kTF32`
  is kept in TRT 11.
- Calibration setup (`set_calibration_profile`, `IInt8Calibrator`). These must
  be removed manually because the correct replacement (Q/DQ in the ONNX) lives
  outside the build script.
- Conditional logic gating on `platform_has_fast_fp16` / `platform_has_fast_int8`
  — the body of those branches becomes a no-op after the flag is removed, but
  the conditional remains. Clean up manually if desired.

## `verify.sh`

End-to-end check that `migrate.py` is doing the right thing. Writes a
representative weakly-typed sample to a temp directory, runs `migrate.py
--write` on it, then asserts the rewritten file matches the strongly-typed
contract (flag present, precision flags absent, valid Python).

```bash
bash verify.sh           # run and clean up
bash verify.sh --keep    # keep the temp workspace for inspection
```

Exits 0 on success, 1 on any assertion failure. Run this before applying
`migrate.py` to a real codebase.
