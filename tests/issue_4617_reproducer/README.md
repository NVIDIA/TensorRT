# Issue #4617: Crash if TensorRT 10.1x is used from a dynamic library

## Description

On Linux, if TensorRT is used by a dynamic library that is loaded using `dlopen()`, there is a crash when the program finishes. This is likely caused by TensorRT doing `dlclose()` on `libnvinfer_builder_resource.so.10.x` from a static C++ object destructor.

## Root Cause

The problem occurs due to the order of destructor execution in glibc:
1. When a dynamically loaded library is unloaded with `dlclose()`, its destructors are called
2. If TensorRT has static C++ objects that call `dlclose()` on `libnvinfer_builder_resource.so.10.x` in their destructors
3. This can cause a crash because the library unloading order is not guaranteed

## Recommended Fix

The problem would likely go away if TensorRT would unload `libnvinfer_builder_resource` from:
- `nvinfer1::IBuilder` destructor, OR
- A normal function marked with `__attribute__((destructor))` rather than a C++ static object destructor

This is because `__attribute__((destructor))` functions are called at a different phase than C++ static destructors, providing better control over cleanup order.

## Environment

- **TensorRT Version**: 10.13, 10.14
- **NVIDIA GPU**: RTX 5090
- **NVIDIA Driver Version**: 580.95.05
- **CUDA Version**: 13.0
- **Operating System**: Ubuntu 24.04.3 LTS

## Files

- `test_library.cpp` - A simple TensorRT library that can be dynamically loaded
- `test_main.cpp` - Main program that loads the library with dlopen()
- `Makefile` - Build script
- `run_test.sh` - Script to run the reproducer

## Building

```bash
make
```

## Running

```bash
./run_test.sh
```

or

```bash
LD_LIBRARY_PATH=. ./test
```

## Expected Behavior

The program should exit cleanly without any crashes.

## Actual Behavior

The program crashes during cleanup when the dynamically loaded library is unloaded.
