# Polygraphy: A Deep Learning Inference Prototyping and Debugging Toolkit


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Advanced](#advanced)
    - [Using The Python API](#using-the-python-api)
    - [Building API Docs](#building-api-docs)
    - [Enabling Internal Correctness Checks](#enabling-internal-correctness-checks)
- [Contributing](#contributing)


## Introduction

Polygraphy is a toolkit designed to assist in running and debugging deep learning models
in various frameworks. It includes a [Python API](./polygraphy) and
[a command-line interface (CLI)](./polygraphy/tools) built using this API.

Among other things, Polygraphy lets you:

- Run inference among multiple backends, like TensorRT and ONNX-Runtime, and compare results
    (example: [API](examples/api/01_comparing_frameworks/), [CLI](examples/cli/run/01_comparing_frameworks/))
- Convert models to various formats, e.g. TensorRT engines with post-training quantization
    (example: [API](examples/api/04_int8_calibration_in_tensorrt/), [CLI](examples/cli/convert/01_int8_calibration_in_tensorrt/))
- View information about various types of models
    (example: [CLI](examples/cli/inspect/))
- Modify ONNX models on the command-line:
    - Extract subgraphs (example: [CLI](examples/cli/surgeon/01_isolating_subgraphs/))
    - Simplify and sanitize (example: [CLI](examples/cli/surgeon/02_folding_constants/))
- Isolate faulty tactics in TensorRT
    (example: [CLI](examples/cli/debug/01_debugging_flaky_trt_tactics/))


## Installation


### Installing Prebuilt Wheels

```bash
python3 -m pip install colored polygraphy --index-url https://pypi.ngc.nvidia.com
```

**NOTE:** When using this method, the command-line toolkit will be installed into `${HOME}/.local/bin` by default.
Make sure to add this directory to your `PATH` environment variable.


### Building From Source

#### Using Make Targets

```bash
make install
```

#### Building Manually

1. Build a wheel:

```bash
python3 setup.py bdist_wheel
```

2. Install the wheel manually from **outside** the repository:

```bash
python3 -m pip install Polygraphy/dist/polygraphy-*-py2.py3-none-any.whl
```

**NOTE:** It is strongly recommended to install the `colored` module for colored output
from Polygraphy, as this can greatly improve readability:
```bash
python3 -m pip install colored
```


### Installing Dependencies

Polygraphy has no hard-dependencies on other Python packages. However, much of the functionality included
does require other Python packages.

#### Automatically Installing Dependencies

It's non-trivial to determine all the packages that will be required ahead of time,
since it depends on exactly what functionality is being used.

To make this easier, Polygraphy can optionally automatically install or upgrade dependencies at runtime, as they are needed.
To enable this behavior, set the `POLYGRAPHY_AUTOINSTALL_DEPS` environment variable to `1` or
`polygraphy.config.AUTOINSTALL_DEPS = True` using the Python API.

NOTE: By default, dependencies will be installed using the current interpreter, and may overwrite existing
packages. The default installation command, which is `python3 -m pip install`, can be overriden by setting
the `POLYGRAPHY_INSTALL_CMD` environment variable, or setting `polygraphy.config.INSTALL_CMD = "..."` using the Python API.

#### Installing Manually

Each `backend` directory includes a `requirements.txt` file that specifies the minimum set of packages
it depends on. This does not necessarily include all packages required for all the functionality provided
by the backend, but does serve as a good starting point.

You can install the requirements for whichever backends you're interested in with:
```bash
python3 -m pip install -r polygraphy/backend/<name>/requirements.txt
```

If additional packages are required, warnings or errors will be logged.
You can install the additional packages manually with:
```bash
python3 -m pip install <package_name>
```

## Usage

Polygraphy includes a command-line interface, [`polygraphy`](./bin/polygraphy), which provides various tools.
For usage information, run `polygraphy --help`

For details on the various tools included in the Polygraphy toolkit, see the
[tools directory](./polygraphy/tools).


## Examples

For examples of both the CLI and Python API, see the [examples directory](./examples).


## Advanced

### Using The Python API

For a high-level overview of the Polygraphy Python API, see the [API directory](./polygraphy).
For more details, see the [Polygraphy API documentation](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html).


### Building API Docs

To build the API documentation, first install required packages:

```bash
python3 -m pip install -r docs/requirements.txt
```

and then use the `make` target to build docs:

```bash
make docs
```

The HTML documentation will be generated under `build/docs`
To view the docs, open `build/docs/index.html` in a browser or HTML viewer.

### Enabling Internal Correctness Checks

Polygraphy includes various runtime checks for internal correctness, which are
disabled by default. These checks can be enabled by setting the `POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS`
environment variable to `1` or `polygraphy.config.INTERNAL_CORRECTNESS_CHECKS = True` in the Python API.
A failure in this type of check indicates a bug in Polygraphy.

When the checks are enabled, Polygraphy will ensure, for example, that loaders do not
modify their state when they are called, and that runners will reset their state correctly in
`deactivate()`.

Note that `POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS` only relates to checks that validate Polygraphy's internal
APIs. User input validation and public API checks are always enabled and cannot be disabled.


## Contributing

For information on how you can contribute to this project, see [CONTRIBUTING.md](./CONTRIBUTING.md)
