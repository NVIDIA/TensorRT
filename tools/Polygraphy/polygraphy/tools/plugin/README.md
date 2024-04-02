# Plugin

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `plugin` tool helps with plugin substitution in an onnx model.
The plugins need to advertise the graph pattern that they can substitute.
This is done by placing a file called pattern.py inside the plugin's directory.
See for example [toyPlugin](../../../examples/cli/plugin/01_match_and_replace_plugin/plugins/toyPlugin)

## Subtools

- `match` finds potential opportunities for substituting subgraphs with a plugin.
    This creates an intermediate file (config.yaml) which the user should further edit to pick the desired plugin substitutions.

- `list` finds potential opportunities for substituting subgraphs with a plugin, without generating an intermediate file.
    This command is to list the potential substitutions, a kind of a dry run of the match tool.

- `replace` replaces subgraphs with plugins, based on the intermediate file (config.yaml).

## Usage

See `polygraphy plugin -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/plugin)
