# TensorRT Engine Explorer Change Log

Dates are in YYYY-MM-DD format.

## v0.1.8 (2024-March)
- Added `trex` command-line tool (see `bin/README.md`)
- Updated to support Python 3.10, new package installations and TensorRT 10.0.
- Made the scripts in the `utils/` directory executable for easier usage.
- Enabled JupyterLab
- Added notebooks/q_dq_placement.ipynb for experimenting with Q/DQ placement, data types and strong typing. Shows how to quickly iterate between ONNX definition, visualization and engine visualization.
- Installation:
  - Shortened the installation time (removed qgrid and its dependencies).
  - Separated the installation to core packages and notebook packages.
  - Removed non-core modules from inclusion in the default trex namespace. This is meant to simplify for users that don't require Jupyter etc.
  - Updated the installation instructions for [PyTorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization) to use source installation which is more reliable.
  - Updates script `install.sh` with options for full or core installation. Installation of a virtual environment is now an opt-in.
- Graph rendering:
  - Updated the graph rendering for TensorRT 10.0 `kgen` kernels.
  - Added an option to display engine layer metadata. TensorRT 10.0 adds ONNX layer information as a metadata field in the layer information.
  - Added a color for FP8 tensors when rendering SVG.
  - Added an option to prevent rendering of disconnected layer nodes.
  - Moved the colormap definitions to a separate file (trex/colors.py) to decouple graph rendering from non-core trex code.
  - Added support for TrainStation engine layers. A TrainStation is an internal TensorRT engine layer that manages data-dependent-shapes device memory allocation. TrainStation layers synchronize the stream they are invoked from.
- Deprecated functionality:
  - Removed display_df_qgrid. Data frames now display using the default panda's table renderer.
- Miscellaneous
  - Added copyright message to test files
  - Updated and fixed the build-log parsing for TensorRT 9.x/10.x.


## v0.1.7 (2023-August)
- Updated graph rendering for TensorRT 9.0 `kgen` kernels.
- Updated TensorRT data formats dictionary.

## v0.1.6 (2023-April)
- Graph rendering:
  - Add node highlighting option.
  - Fix bug https://github.com/NVIDIA/TensorRT/issues/2779

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
