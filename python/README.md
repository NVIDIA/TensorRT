# TensorRT Python Bindings

## Installation

### Download pybind11

Create a directory for external sources and download pybind11 into it.
```bash
export EXT_PATH=~/external

mkdir -p $EXT_PATH && cd $EXT_PATH
git clone https://github.com/pybind/pybind11.git
```

### Download Python headers

#### Add Main Headers

1. Get the source code from the official [python sources](https://www.python.org/downloads/source/)
2. Copy the contents of the `Include/` directory into `$EXT_PATH/pythonX.Y/include/` directory.

#### Add PyConfig.h

1. Download the deb package for the desired platform from [here](https://packages.debian.org/search?searchon=contents&keywords=pyconfig.h&mode=path&suite=unstable&arch=any).
    Typical plaforms include `x86_64` (`amd64`), `aarch64` (`arm64`), and `ppc64le` (`ppc64el`)
2. Unpack the debian with `ar x <libpython...>.deb`
3. Unpack the contained `data.tar.xz` with `tar -xvf`
4. Copy the `./usr/include/<platform>/` directory into the `$$EXT_PATH/pythonX.Y/include/` directory here.
    It should only contain a single file - `pyconfig.h`


### Build Python bindings

Use `build.sh` to generate the installable wheels for intended python version and target architecture.

Example: for python 3.8 `x86_64` wheel,
```bash
cd $TRT_OSSPATH/python
PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=8 TARGET_ARCHITECTURE=x86_64 ./build.sh
```

### Install the python wheel

```bash
python3 -m pip install build/dist/tensorrt-*.whl
```
