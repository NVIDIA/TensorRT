# Polygraphy: A Deep Learning Inference Prototyping and Debugging Toolkit


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Advanced](#advanced)
    - [Using The API](#using-the-api)
    - [Building API Docs](#building-api-docs)
- [Contributing](#contributing)


## Introduction

Polygraphy is a toolkit designed to assist in running and debugging deep learning models
in various frameworks. It includes a [Python API](./polygraphy), and
[several command-line tools](./polygraphy/tools) built using this API.


## Installation

**NOTE:** It is strongly recommended to install the `colored` module for colored output
from Polygraphy, as this can greatly improve readability.

Each `backend` directory includes a `requirements.txt` file that specifies which packages
it depends on. You can install the requirements for whichever backends you're interested in
using:

```bash
python3 -m pip install -r polygraphy/backend/<name>/requirements.txt
```

### Install From Package Index

```bash
python3 -m pip install nvidia-pyindex
python3 -m pip install polygraphy
```

### Building From Source

#### Using Make Targets

```bash
make install
```
Or, if installing inside a virtual environment:
```bash
make install_venv
```

#### Building Manually

1. Build a wheel:

```bash
python3 setup.py bdist_wheel
```

2. Install the wheel manually from **outside** the repository:

```bash
python3 -m pip install polygraphy/dist/polygraphy-X.Y.Z-py2.py3-none-any.whl --user
```
where `X.Y.Z` is the version number.


## Usage

Polygraphy includes a command-line interface, [`polygraphy`](./bin/polygraphy), which provides various tools.
For usage information, run `polygraphy --help`

For details on the various tools included in the Polygraphy toolkit, see the
[tools directory](./polygraphy/tools).


## Examples

For examples of both the CLI and Python API, see the [examples directory](./examples).


## Advanced

### Using The API

For details on the Polygraphy Python API, see the [API directory](./polygraphy).
To view documentation about a specific class or function, you can view the
docstring with Python's `help()` function.


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


## Contributing

For information on how you can contribute to this project, see [CONTRIBUTING.md](./CONTRIBUTING.md)
