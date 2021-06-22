# Polygraphy Tools

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Examples](#examples)
- [Adding New Tools](#adding-new-tools)


## Introduction

This directory includes command-line tools from the Polygraphy debugging toolkit.

For more information about a specific tool, see the README in the corresponding directory here.

Note that some of the tools included are still experimental. Any tool labeled `[EXPERIMENTAL]`
may be subject to backwards-incompatible changes, or even complete removal at any point in time.


## Usage

All the tools provided by Polygraphy can be invoked using the polygraphy binary: [`bin/polygraphy`](../../bin/polygraphy)

For usage information on a specific tool, you can see the help output with: `polygraphy <subtool> -h`


## Examples

For examples, see the corresponding subdirectory in [examples/cli](../../examples/cli)


## Adding New Tools

You can add a new tool by adding a new file in this directory, creating a
class that extends the [`Tool` base class](./base/tool.py), and adding
the new tool to the [registry](./registry.py).

For details on developing tools, see [this example](../../examples/dev/01_writing_cli_tools/).
