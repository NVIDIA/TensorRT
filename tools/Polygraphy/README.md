# Polygraphy: A Deep Learning Inference Prototyping and Debugging Toolkit


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Command-line Toolkit](#command-line-toolkit)
- [Python API](#python-api)
- [Examples](#examples)
- [How-To Guides](#how-to-guides)
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

**IMPORTANT**: **Polygraphy supports only Python 3.6 and later.**
    **Before following the instructions below, please ensure you are using a supported version of Python.**


### Installing Prebuilt Wheels

```bash
python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

**NOTE:** *On Linux, the command-line toolkit is usually installed to `${HOME}/.local/bin` by default.*
    *Make sure to add this directory to your `PATH` environment variable.*


### Building From Source

#### Using Make Targets (Linux)

```bash
make install
```

#### Using Powershell Script (Windows)

Make sure you are allowed to execute scripts on your system then run:
```ps
.\install.ps1
```

#### Building Manually

1. Install prerequisites:

```
python -m pip install wheel
```

2. Build a wheel:

```
python setup.py bdist_wheel
```

3. Install the wheel manually from **outside** the repository:

    On Linux, run:

    ```
    python -m pip install Polygraphy/dist/polygraphy-*-py2.py3-none-any.whl
    ```

    On Windows, using Powershell, run:

    ```ps
    $wheel_path = gci -Name Polygraphy\dist
    python -m pip install Polygraphy\dist\$wheel_path
    ```


    **NOTE:** *It is strongly recommended to install the `colored` module for colored output*
    *from Polygraphy, as this can greatly improve readability:*
    ```
    python -m pip install colored
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

**NOTE**: *By default, dependencies will be installed using the current interpreter, and may overwrite existing*
    *packages. The default installation command, which is `python -m pip install`, can be overriden by setting*
    *the `POLYGRAPHY_INSTALL_CMD` environment variable, or setting `polygraphy.config.INSTALL_CMD` using the Python API.*

If you'd like Polygraphy to prompt you before automatically installing or
upgrading pacakges, set the `POLYGRAPHY_ASK_BEFORE_INSTALL` environment variable to `1`
or `polygraphy.config.ASK_BEFORE_INSTALL = True` using the Python API.

#### Installing Manually

Each `backend` directory includes a `requirements.txt` file that specifies the minimum set of packages
it depends on. This does not necessarily include all packages required for all the functionality provided
by the backend, but does serve as a good starting point.

You can install the requirements for whichever backends you're interested in with:
```bash
python -m pip install -r polygraphy/backend/<name>/requirements.txt
```

If additional packages are required, warnings or errors will be logged.
You can install the additional packages manually with:
```bash
python -m pip install <package_name>
```


## Command-line Toolkit

For details on the various tools included in the Polygraphy toolkit,
see the [CLI User Guide](./polygraphy/tools).


### Python API

For more information on the Polygraphy Python API, including a high-level overview and the
Python API reference documentation, see the [API directory](./polygraphy).


## Examples

For examples of both the CLI and Python API, see the [examples directory](./examples).


## How-To Guides

For how-to guides, see the [how-to guides directory](./how-to).


## Contributing

For information on how you can contribute to this project, see [CONTRIBUTING.md](./CONTRIBUTING.md)
