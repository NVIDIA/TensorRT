# TensorRT Python Bindings

## Installation

### Download pybind11

Create a directory for external sources and download pybind11 into it.
```bash
export EXT_PATH=~/external

mkdir -p $EXT_PATH && cd $EXT_PATH
git clone https://github.com/pybind/pybind11.git
```

The default build file assumes that pybind11 is located in $HOME, but you can modify this with `-DPYBIND11_DIR=/some/other/dir`.

### Download Python headers

#### Add Main Headers

1. Get the source code from the official [python sources](https://www.python.org/downloads/source/)
2. Copy the contents of the `Include/` directory into `$EXT_PATH/pythonX.Y/include/` directory.

#### Add PyConfig.h

1. Download the deb package for the desired platform from [here](https://packages.debian.org/search?searchon=contents&keywords=pyconfig.h&mode=path&suite=unstable&arch=any)
    As of this writing, we typically want `x86_64` (`amd64`), `aarch64` (`arm64`), and `ppc64le` (`ppc64el`)
2. Unpack the debian with `ar x <libpython...>.deb`
3. Unpack the contained `data.tar.xz` with `tar -xvf`
4. Copy the `./usr/include/<platform>/` directory into the `$$EXT_PATH/pythonX.Y/include/` directory here. It should only contain a single file - `pyconfig.h`


### Build Python bindings

Use `build.sh` to generate the installable wheels for intended python version and target architecture.

Example: for python 3.8  `x86_64` wheel,
```bash
cd $TRT_OSSPATH/python
PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=8 TARGET_ARCHITECTURE=x86_64 ./build.sh
```

### Install the python wheel

```bash
pip3 install build/dist/tensorrt-*.whl
```

## Project Structure
- **docstrings/** contains headers for defining docstrings.
    - **infer/** contains NvInfer.h docstrings.
        - **pyCoreDoc.h** covers material in pyCore.cpp.
        - **pyFoundationalTypesDoc.h** covers material in  pyFoundationalTypes.cpp.
        - **pyInt8Doc.h** covers material in  pyInt8.cpp.
        - **pyGraphDoc.h** covers material in  pyGraph.cpp.
        - **pyPluginDoc.h** covers material in pyPlugin.cpp.
    - **parsers/** contains parser docstrings.
        - **pyCaffeDoc.h** covers material in pyCaffe.cpp.
        - **pyOnnxDoc.h** covers material in  pyOnnx.cpp.
        - **pyUffDoc.h** covers material in  pyUff.cpp.
    - **pyTensorRTDoc.h** covers material in  pyTensorRT.cpp.
- **include/** contains source headers.
    - **ForwardDeclarations.h** contains forward declarations for binding functions defined in the source files.  
    - **utils.h** contains useful utilities for bindings.
- **skeleton/** contains skeleton files to ease the process of adding new files to the project.
    - **pySOME_NAME.cpp** is a skeleton source file for binding code.
    - **pySOME_NAME.rst** is a skeleton source file for Sphinx documentation.
    - **pySOME_NAMEDoc.h** is a skeleton header file for docstrings.
    - **test_SOME_NAME.py** is a skeleton Python testing script.
- **src/** contains all binding code.
    - **infer/** contains binding code for `NvInfer.h` .
        - **pyCore.cpp** binds the Builder, Engine, Logger, Runtime and other core components of the TensorRT API.
        - **pyFoundationalTypes.cpp** binds Dims and all its subclasses, DataType, Weights, etc.
        - **pyInt8.cpp** binds classes related to Int8 Calibration.
        - **pyGraph.cpp** binds all Layer classes (i.e. subclasses of `ILayer`), ITensor, and INetworkDefinition.
        - **pyPlugin.cpp** binds all classes related to TensorRT plugins.
    - **parsers/** contains binding code for parsers.
        - **pyCaffe.cpp** contains binding code for the Caffe parser.
        - **pyOnnx.cpp** contains binding code for the open source ONNX parser.
        - **pyUff.cpp** contains binding code for the UFF parser.
    - **pyTensorRT.cpp** contains binding code for the top-level module. This is where everything comes together under a single module.
- **CMakeLists.txt** defines the building process.

### Adding Bindings to Existing Files
The process for adding new bindings typically involves three steps:

1. Adding the binding code to the appropriate `.cpp` file. There are a few differences between the C++ and Python APIs:
    - The Python bindings use `snake_case` instead of `camelCase`.
    - Getters/setters are bound to properties where possible (not possible if the getter takes arguments or the setter has a return value, for example).
    - The Python bindings conform to `numpy` naming conventions. So for example, `dims` -> `shape`, `size_in_bytes` -> `nbytes`, `number_of_elements` -> `size`, `operation` -> `op` etc.  
2. Adding docstrings in the corresponding docstring header file (in `docstrings/`)
    - If you are adding a new class, you will also need to update the corresponding `.rst` in `documentation/python/` in the top-level.
    - New functions in already-documented classes should be picked up automatically, but you should run `make python_docs` and check to make sure.
    - When documenting properties, add ```:ivar property_name: :class:`PropertyType` description``` to the `descr` docstring instead of creating a new string.
    - It is possible (and highly recommended) to link to other classes with ```:class:`ClassName` ``` or to functions with ```:func:`FuncName` ```. Note that these links **MUST** be followed by a space, or they will not work.
3. Adding binding tests to the corresponding test file. These tests typically only check for the presence of fields, but if it is possible to do more extensive testing, you should include that too.

### Adding a New Source File to the Project
Before you add new source files, make sure that none of the existing source files are appropriate locations for the bindings you want to add.  

**IMPORTANT** Update the `Project Structure` section above as you add new files!

Throughout this process, you should replace any occurrences of `<SOME_NAME>` with a name describing what functionality your new files cover.

### Adding the Binding Files
1. Make a copy of `skeleton/py<SOME_NAME>.cpp` in some appropriate subdirectory of `src/` and rename it.
2. Make a copy of `skeleton/py<SOME_NAME>Doc.h` in the corresponding location in `docstrings/` (which mirrors the directory structure of `src`).
2. Make a copy of `skeleton/py<SOME_NAME>.rst` in the corresponding location in `docs/` (which also mirrors the directory structure of `src`).
3. In your newly created `.cpp` source file:
    1. Replace `#include "<SUBDIRECTORY_PATH>/py<SOME_NAME>Doc.h"` with the path to the file you added in `docstrings/`
    2. Replace `void bind<SOME_NAME>(py::module& m)` with some meaningful name.
4. In your newly created `.h` docstring file:
    1. Replace all instances of `<SOME_NAME>` as above.
    2. Add strings for each class and function. Each string is a `constexpr const char*`.
        Every class should have a `descr` string that describes what it is, as well as any attributes/members it has.
        Every function should have a docstring with the same name as its Python binding.
        In the bindings, you can use these strings instead of writing docstrings directly in the code.  
5. In your newly created `.rst` docs file:
    1. Replace all instances of `<SOME_NAME>` as above.
    2. Add sections for each class/enum you added.
6. Update `include/ForwardDeclarations.h` with a forward declaration of your new function.
7. Finally, call your binding function in `src/pyTensorRT.cpp`.

