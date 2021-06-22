# Command-Line Argument Helpers

This directory includes helper modules for managing groups of command-line arguments.

Included are various predefined argument groups that can be reused across command-line tools.

## Argument Groups

An argument group essentially bundles together a set of arguments and helper functions that use
those arguments. The interface is defined in [BaseArgs](./base.py). Additionally, argument groups
may specify dependencies on other argument groups via the `register()` mechanism - for example,
an ONNX loader argument group may depend on model arguments.

Tools can then subscribe to these argument groups via the `subscribe_args()` function defined in
[tool.py](../base/tool.py).

For details, see [the example](../../../examples/dev/01_writing_cli_tools).
