# ONNX GraphSurgeon Change Log

Dates are in YYYY-MM-DD format.

## v0.3.26 (2022-12-09)
### Fixed
- Fixed a bug where node domain was not preserved.


## v0.3.25 (2022-10-14)
### Added
- Added a `should_exclude_node` parameter to `fold_constants` to allow for excluding nodes
    from constant folding.

### Fixed
- Fixed a bug where `fold_constants` would fold quantization nodes, which are intended to be executed
    at runtime even though they are computable beforehand.


## v0.3.24 (2022-08-31)
### Fixed
- Fixed a bug where `fold_constants` would not work at all when `onnxruntime` was not installed.
    Now, `fold_constants` can still partially fold the graph even when `onnxruntime` is not available.


## v0.3.23 (2022-08-24)
### Fixed
- Fixed a bug in `fold_constants` where shape tensor cast elision would not work correctly
    if one input of a binary op was produced by a constant node and had a data type that
    differed from that of the other input prior to the cast.
    For example, a pattern like this would have previously failed, but now works as expected:
    ```
    inp (int32)            Constant
        |                     |
    Cast (to=float32)   constant_out (float32)
                   \       /
                      Sub
                       |
                 Cast (to=int32)
    ```
- Fixed a bug where shape-tensor cast elision would invalidate the graph if the original
    casted inputs were being used as graph outputs or by other nodes.


## v0.3.22 (2022-08-22)
### Changed
- Updated `fold_constants` to issue clearer warnings and avoid evaluating tensors which exceed
    the size threshold.


## v0.3.21 (2022-08-19)
### Added
- Added a `size_threshold` option in `fold_constants` which allows for disabling constant folding
    for nodes which would generate tensors larger than the given size.


## v0.3.20 (2022-07-12)
### Fixed
- Fixed a bug where shape tensor cast elision would sometimes fail when the Cast input had a type of int64.
- Fixed a bug where opset information would not be propagated down to nested graphs.


## v0.3.19 (2022-04-13)
### Added
- Added support for flattening conditional subgraphs into the parent graph in `fold_constants()`.


## v0.3.18 (2022-03-31)
### Fixed
- Fixed a bug where `{node/tensor}.{inputs/outputs} += <value>` would cause the inputs/outputs of the node/tensor
    to be cleared.


## v0.3.17 (2022-03-18)
### Added
- Added `producer_name` and `producer_version` to `Graph` class so that they are preserved during model import/export.


## v0.3.16 (2022-02-23)
### Fixed
- Fixed a bug where `Graph.fold_constants()` was not providing a value for the `providers` parameter in `onnxruntime.InferenceSession`.


## v0.3.15 (2022-01-18)
### Fixed
- Fixed a bug where `Graph.toposort()` would not consider implicit inputs of nodes with subgraphs.
    For example, a graph including an `If` node whose subgraphs used tensors from the outer graph
    may previously have been sorted such that it occurred before the nodes producing those tensors.


## v0.3.14 (2021-10-14)
### Fixed
- Fixed a bug where `numpy.dtype` would not be exported correctly when specified as a node attribute.


## v0.3.13 (2021-09-21)
### Added
- `Graph.tensors()` will now display a warning when duplicate tensors are detected in the graph, even if `check_duplicates=False`.
    As before, when `check_duplicates=True`, it will throw an exception in such cases.


## v0.3.12 (2021-08-24)
### Added
- Added support for `Cast` elision in `fold_constants()`.


## v0.3.11 (2021-07-14)
### Changed
- Updated `fold_constants()` so that it no longer fails if a shape folding pass fails when `error_ok` is `True`.

### Fixed
- Fixed a bug where `fold_constants()` would fail if a model contained a `Slice` node without a `starts` or `ends` input.


## v0.3.10 (2021-05-20)
### Added
-  Added support for folding `Shape -> Slice` patterns even when the entire shape may not be known.


## v0.3.9 (2021-04-20)
### Changed
- `fold_constants()` will no longer store values for foldable tensors whose outputs are all foldable.
    For example, while folding a constant subgraph like `A (constant) -> B -> C`, previously, `B` values
    would be computed in addition to `C`. With these changes, only `C` values are computed and stored.
    This can reduce memory usage significantly.


## v0.3.8 (2021-04-15)
### Fixed
- Fixed a bug where `copy()` would not work with subgraphs that included tensors with the same
    names as outer graph tensors unless a `tensor_map` was provided.


## v0.3.7 (2021-03-31)
### Added
- `fold_constants()` can now fold `Shape -> Gather` patterns even when the entire shape may not be known.
- Added an `error_ok` parameter in `fold_constants()` which can be set to `False` to re-raise errors encountered
    during inference.

### Fixed
- Fixed a bug where `copy()` would not correctly copy tensors in nested graphs.
- Fixed a bug where `fold_constants()` would attempt to fold nodes including graph attributes even if nodes within
    the nested graph could not be folded.


## v0.3.6 (2021-03-27)
### Fixed
- `fold_constants()` no longer loads constant values into numpy arrays. This can save a significant amount of memory.
- `cleanup()` will no longer remove unused graph inputs by default - this was causing invalid ONNX models to be generated
    in cases with `Loop` nodes. Set `remove_unused_graph_inputs` to `True` to revert to the old behavior.
- `cleanup()` will no longer reorder node inputs in cases where they are also graph outputs.


## v0.3.5 (2021-03-24)
### Added
- Added support for models with externally stored data. See the README for details on how to import and export such models.

### Fixed
- Operator domains are now preserved when exporting graphs to ONNX.

## v0.3.4 (2021-03-10)
### Fixed
- `fold_constants` will no longer attempt to run inference if there are no constants to compute.


## v0.3.3 (2021-03-04)
### Fixed
- Fixed a bug in `fold_constants` where it would fail if ONNX-Runtime could not run a node with constant inputs.
    In such cases, the graph is now partitioned to exclude the node before running another pass of constant folding.
- Fixed a bug where graph output tensors would still point to consumer nodes that had been removed from the graph.
- Constant folding is now significantly faster in models with large weights.


## v0.3.2 (2021-02-13)
### Added
- Added support for folding `Shape` nodes in `fold_constants`. This requires that shape inference has been run
    on the graph, and that the input to the `Shape` node has a static shape.
    This behavior can be disabled by setting `fold_shapes=False`.

### Changed
- `cleanup`, `toposort`, and `fold_constants` are now recursively applied to subgraphs by default.
    This behavior can be disabled by setting `recurse_subgraphs=False`.


## v0.3.1 (2021-02-12)
### Fixed
- Fixed a bug where `do_type_check` would not propagate to subgraphs.
- Fixed a bug where `cleanup()` would incorrectly remove outer-level nodes if they were used only by inner-nodes of subgraphs.

### Removed
- Removed `__deepcopy__` from `Graph` as it wasn't deep-copying weights or attributes.
    The method is now called `copy` and makes a shallow copy of everything except `Node`s and `Tensor` instances.


## v0.3.0 (2021-02-12)
### Fixed
- Fixed a bug where shapes including empty strings for `dim_param` would be treated as empty tensors.
    They are now correctly imported as tensors with dynamic shapes.
- Fixed a bug where variable tensors with unknown shapes would be imported as scalars.


## v0.2.9 (2021-02-01)
### Changed
- The `values` property of `Constant` tensors is now lazily loaded. This can greatly improve model loading times.


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
