# Check

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `check` tool can be used to check and validate for various use-cases.


## Subtools

- [EXPERIMENTAL] `lint` can be used to validate ONNX models and catch exceptions/warnings over independent nodes
    in the graph in a JSON format.

- `accuracy` re-checks accuracy results saved by `polygraphy run --save-accuracy-results` against
    different thresholds, without re-running inference.

## Usage

See `polygraphy check -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/check)
