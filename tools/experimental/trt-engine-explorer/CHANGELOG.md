# TensorRT Engine Explorer Change Log

Dates are in YYYY-MM-DD format.

## v0.1.5 (2022-12-06)
- Updated requirements.txt for Ubuntu 20.04 and 22.04

## v0.1.4 (2022-09-01)
- Validated using Python 3.8.
- Added more tests.
- Added support for engine shapes profiles.
- Added `KNOWN_ISSUES.md`.
- Added `trex/excel_summary.py` which adds an API to generate an Excel summary spreadsheet with worksheets generated from several `trex.report_card` reports.
- Added `set_table_display_backend` to provide user control over the backend library displaying tables in notebooks.
- Graph rendering:
  - Add exportng of engine graph to ONNX format for viewing with Netron.
  - Refactor `graphing.py` and change the engine-graph rendering API: added fine control of rendering options.
  - Reformat layers: optionally render Origin attribute.
  - Constant layers: optionally display.
  - Convolution layers: optionally display activation details.
  - Elementwise layers: optionally display operations code.
- Utilities:
  - Add GPU clocks locking and power-limiter context manager: `utils/config_gpu.py`.
  - Add simple GPU context manager and activate when profiling.
  - Script `process_engine.py` will now lock the GPU and memory clocks to maximum frequency by default during engine profiling and release the clocks when done.
- Notebooks:
  - Added `report_card_reformat_overview()` - This is a pull-down of several views showing what are roles/functions of Reformat layers in the model.
  - Added `report_card_draw_plan_graph_extended()` - displays a set of widgets to control graph rendering options.
- Small bug fixes.

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
