# ONNX GraphSurgeon Change Log

Dates are in YYYY-MM-DD format.


## v0.2.8 (2020-10-08)
### Fixed
- Fixed a bug where graph inputs and outputs could be assigned `SynchronizedList` instances, and would therefore be modified if nodes in the graph were.


## v0.2.7 (2020-09-29)
### Changed
- Changed the default value of `remove_unused_node_outputs` in `cleanup()` to `False`, as a value of `True` can lead to unintuitive behavior,
    especially with looping constructs like `Scan` and `Loop`.


## v0.2.6 (2020-09-25)
### Fixed
- Fixed a bug where calling `graph.tensors()` would cause the inputs or outputs of some tensors to be modified.

### Changed
- `SynchronizedList.__add__()` no longer modifies the left operand.


## v0.2.5 (2020-09-21)
### Fixed
- Fixed a bug where nodes including subgraphs whose inputs/outputs had the same names as the node's inputs/outputs would not be imported correctly.


## v0.2.4 (2020-09-14)
### Fixed
- `fold_constants()` will no longer fail if there is nothing to fold in the graph
- `cleanup()` will now properly remove the producer nodes of graph inputs.
- Fixed a bug where graph input/output tensors not attached to nodes would not be correctly exported.


## v0.2.3 (2020-06-17)
### Added
- `Graph.register()` now accepts an `opsets` argument so that functions can be registered for specific opsets.

### Removed
- `has_metadata` has been removed from `Tensor`, since the function is no longer used.


## v0.2.2 (2020-06-17)
### Fixed
- ONNX GraphSurgeon now enforces the constraint that graph inputs/outputs must include type information.
- Fixed a bug where `opset` was not being considering when running inference for constant folding.


## v0.2.1 (2020-06-10)
### Added
- Added `layer()` function to `Graph` to make it easier to generate models from scratch
- Added `i()` and `o()` convenience functions to `Tensor`, which are similar to the functions for `Node`, but return `Tensor`s instead of `Node`s


## v0.2.0 (2020-04-15)
### Added
- Added an `examples` directory
- Added `has_metadata()` to `Tensor` classes to determine if dtype/shape are known.
- Added a `check_duplicates` parameter to `Graph.tensors()` to make it easy to check for duplicate tensors in the graph.

### Changed
- Various improvements to the logger
- Updated `OnnxImporter` so that it can correctly import shapes and types from an ONNX graph after shape inference.
- Made `Tensor` an abstract class - all tensors in a graph are now either `Variable` or `Constant`
- Renames `generate_tensor_map()` to `tensors()` in `Graph`
- Removed `Tensor` suffix from Tensor classes.


## v0.1.3 (2020-02-26)
### Fixed
- The `import_onnx` and `export_onnx` functions will now preserve opset information and `dim_param` values in shapes.


## v0.1.2 (2020-02-19)
### Added
- Added `i()` and `o()` convenience functions to `Node` for retrieving input/output nodes.
- Added `fold_constants()` to `Graph` to allow for folding constants in the graph.
- Added `__deepcopy__()` to `Graph`.
- Added `to_constant()` and `to_variable()` functions to `Variable` and `Constant` respectively to transmute them in-place.


## v0.1.1 (2020-02-11)
### Changed
- Removed some type annotations to allow compatibility with Python 3.5.


## v0.1.0 (2020-02-11)
### Added
- Added `Node`, `Tensor` and `Graph` classes.
- Added `BaseImporter` and `OnnxImporter` classes.
- Added support for importing initializers in the `OnnxImporter`
- Added `Variable` and `Constant`
- Consolidates inputs/outputs of Nodes/Tensors. Now, inputs/outputs should generally only be added to `Node`s.
- Added `OnnxExporter` to export `Graph` to `onnx.GraphProto`
- Added `OnnxExporter` and `OnnxImporter` to public imports
- Added `toposort` function to `Graph`, which will topologically sort it.
- Added `cleanup` function to `Graph`, which will remove unused nodes and tensors.
- Added high-level API for importing/exporting `Graph`s from/to ONNX models.
- `Graph`s are now generated with a default name of `onnx_graphsurgeon`
