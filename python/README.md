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

Example: Python 3.9
```bash
wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz
tar -xvf Python-3.9.16.tgz
mkdir -p $EXT_PATH/python3.9
cp -r Python-3.9.16/Include/ $EXT_PATH/python3.9/include
```

#### Add PyConfig.h

1. Download the deb package for the desired platform from [here](https://packages.debian.org/search?searchon=contents&keywords=pyconfig.h&mode=path&suite=unstable&arch=any).
    Typical plaforms include `x86_64` (`amd64`), `aarch64` (`arm64`), and `ppc64le` (`ppc64el`).
    For older versions of Python, you may need to select a different suite.
2. Unpack the debian with `ar x <libpython...>.deb`
3. Unpack the contained `data.tar.xz` with `tar -xvf`
4. Find `pyconfig.h` in the `./usr/include/<platform>/pythonX.Y/` directory and copy it into `$EXT_PATH/pythonX.Y/include/`.


### Build Python bindings

Use `build.sh` to generate the installable wheels for intended python version and target architecture.

Example: for Python 3.9 `x86_64` wheel,
```bash
cd $TRT_OSSPATH/python
TENSORRT_MODULE=tensorrt PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=9 TARGET_ARCHITECTURE=x86_64 ./build.sh
```

### Install the python wheel

```bash
python3 -m pip install ./build/bindings_wheel/dist/tensorrt-*.whl
```
