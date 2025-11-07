#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to run the TensorRT Issue #4617 reproducer

set -e

echo "=== TensorRT Issue #4617 Reproducer ==="
echo ""

# Check if TensorRT is installed
if [ -z "$TRT_LIBPATH" ]; then
    echo "TRT_LIBPATH not set, checking default locations..."
    
    # Try common TensorRT installation paths
    if [ -d "/usr/lib/x86_64-linux-gnu" ] && [ -f "/usr/lib/x86_64-linux-gnu/libnvinfer.so" ]; then
        export TRT_LIB_DIR="/usr/lib/x86_64-linux-gnu"
        export TRT_INCLUDE_DIR="/usr/include/x86_64-linux-gnu"
        echo "Found TensorRT in system paths"
    else
        echo "ERROR: TensorRT not found in default locations"
        echo "Please set TRT_LIBPATH environment variable to your TensorRT installation"
        echo "Example: export TRT_LIBPATH=/path/to/TensorRT-10.13.3.9"
        exit 1
    fi
else
    export TRT_LIB_DIR="$TRT_LIBPATH/lib"
    export TRT_INCLUDE_DIR="$TRT_LIBPATH/include"
    echo "Using TensorRT from: $TRT_LIBPATH"
fi

echo ""
echo "Building test..."
make clean
make

echo ""
echo "Running test..."
echo "NOTE: If the bug is present, the program will crash during library unload"
echo ""

# Run the test with proper library path
LD_LIBRARY_PATH="$TRT_LIB_DIR:." ./test

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== TEST PASSED ==="
    echo "The program completed successfully without crashes."
    echo "This indicates the issue is either fixed or not present in your TensorRT version."
else
    echo "=== TEST FAILED ==="
    echo "Exit code: $EXIT_CODE"
    if [ $EXIT_CODE -eq 139 ]; then
        echo "Segmentation fault detected - this is likely the bug described in issue #4617"
    fi
fi

exit $EXIT_CODE
