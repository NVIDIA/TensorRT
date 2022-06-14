# TensorRT Engine Explorer Change Log

Dates are in YYYY-MM-DD format.

## v0.1.2 (2022-06-01)
- Graph rendering:
  - Add timing information to engine graph nodes.
  - Change layer coloring scheme.
  - Add printing of binding region names in the graph.
  - Fix bug: When a NoOp layer is folded, fold into the correct input of the next layer.
  - Fix bug: When a leaf NoOp layer outputs graph bindings, these outputs replace the output of the previous layer.
  - Add info to conv and pwgen graph nodes in detailed rendering mode. Convolutions display the activation and residual-add operations. Pointwise layers display a list of the fused operations.

- Notebooks:
  - Add a detailed comparison of two engines by computing the graph isomorphism (aligning the graphs).
  - Remove some non-useful diagrams in engine comparison (stacked layer latencies).
  - Add a notebook with trex API examples.

- Utilities:
  - Change log files names in `process_engine.py` to include the engine name.
  - Parse `trtexec` build and profiling log files to collect various metadata, instead of collecting device properties using pycuda API.
  - Add `--useSpinWait` to `process_engine.py`.

- Miscellaneous:
  - Improve error messages.
  - Fix bug: When computing convolution MACs account for group size.
  - Code refactoring.
  - Add QAT ResNet example directory (`trex/examples/pytorch/resnet/`). This is the example referenced by the TREx blog.
  - Add optional engine name when creating an EnginePlan instance. This is useful when comparing multiple plans.
  - Add `Resources.md` file, with links to reference performance materials.
  - Add convolution channel-alignment lint.
## v0.1.0 (2022-04-06)
- First release
