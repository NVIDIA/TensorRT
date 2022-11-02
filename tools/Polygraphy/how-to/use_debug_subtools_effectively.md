# Using `debug` Subtools Effectively


## Table Of Contents

- [Introduction](#introduction)
- [Tips And Tricks](#tips-and-tricks)
    - [Using Interactive Mode To Get Started](#using-interactive-mode-to-get-started)
    - [What Counts As A "Failure"?](#what-counts-as-a-failure)


## Introduction

The `debug` subtools work on the same general principles:

1. Iteratively perform some task that generates some output
2. Evaluate the generated output to determine if it should be considered `good` or `bad`
3. Sort any tracked artifacts into `good` and `bad` directories based on (2)
4. Make changes if required and then repeat the process

This guide provides some general information as well as tips and tricks on
how to use `debug` subtools effectively.


## Tips And Tricks

### Using Interactive Mode To Get Started

Automatic mode (i.e. when a `--check` command is provided) in the `debug` subtools
provides a totally hands-off approach. However, if you're just starting out, it can also be
harder to use and reason about.
The `debug` tools in Polygraphy also provide an interactive mode which will prompt and
guide you through the process.

To use interactive mode, you can simply omit the `--check` option.
For example, with `debug reduce`:
```bash
polygraphy debug reduce <model.onnx> -o <reduced.onnx>
```


### What Counts As A "Failure"?

In automatic mode (i.e. when a `--check` command is provided), `debug` subtools will assume that
a non-zero exit status indicates a failure. There are a few cases where this assumption may not
be desirable:

- You may want to ignore certain types of failures so that you can get a minimal model that
    fails in a very specific way.

- Your `--check` command may not exit with a non-zero status at all!
    Instead, you may want to trigger a failure based on the output of such a command instead of
    its exit status.

For these cases, you can use the `--fail-regex` option, which will trigger a failure only when
some part of the output from the `--check` command (on either `stdout` or `stderr`) matches
one of the specified regular expressions.
