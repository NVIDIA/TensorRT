# Command-Line Argument Groups

## Introduction

The Polygraphy command-line toolkit includes several tools, many of which require the same or similar functionality as other tools.
To facilitate efficient code reuse, commonly used functionality is bundled into reusable components, which we'll refer
to as `Argument Group`s.

## Argument Groups

An `Argument Group` combines a set of command-line options, parsing logic, and functionality related to those arguments.
The interface is defined in [BaseArgs](./base.py). The most commonly used methods are:

- `add_parser_args(self, parser)`: Adds options to an `argparse` argument parser.

- `parse(self, args)`: Parses command-line arguments from the `args` object created by `argparse` and
    populates attributes in the argument group.
    For simple arguments, this amounts to assigning an attribute of the argument group to a
    corresponding attribute in the `args` object.
    For more complex arguments, such as input shapes, this may involve more complex parsing.

- `add_to_script(self, script)`: Adds code to a Python script that will provide functionality
    related to the argument group. For example, `OnnxLoadArgs`, which is responsible for
    loading ONNX models, may populate the script with a `OnnxFromPath` loader.

    *NOTE: You may be wondering why we add code to a script instead of just providing a*
    *method that will perform the relevant function, e.g. `load_onnx` in this case.*
    *The reason is that adding to a script allows tools like `polygraphy run` to compose*
    *together complex behavior and generate Python scripts which can be edited and used for more advanced needs.*

Many of the argument groups *also* provide immediately evaluated helper methods, such as `load_onnx()` in the case of
`OnnxLoadArgs`, which can be used by tools that do not need to generate scripts.
These helpers typically use the `run_script` method from `polygraphy.tools.args.util` to reuse the logic from their
`add_to_script` method.

## Usage

Tools can subscribe to argument groups by implementing the `get_subscriptions()` interface defined in [tool.py](../base/tool.py).
This will add all the command-line options provided by the argument group to the tool, and these will be parsed
automatically before the tool's `run` method is called.
The tool can then access the argument groups via `self.arg_groups`.

For details, see [the example](../../../examples/dev/01_writing_cli_tools).
