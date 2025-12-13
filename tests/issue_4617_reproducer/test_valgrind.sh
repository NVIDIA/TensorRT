#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to run the reproducer under Valgrind to detect memory issues

set -e

echo "=== TensorRT Issue #4617 Valgrind Test ==="
echo ""

# Check if valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "ERROR: valgrind is not installed"
    echo "Install with: sudo apt-get install valgrind"
    exit 1
fi

# Build the test
echo "Building test..."
make clean
make

echo ""
echo "Running test under Valgrind..."
echo "This may take several minutes..."
echo ""

# Run with valgrind
LD_LIBRARY_PATH="${TRT_LIB_DIR:-.}:." valgrind \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file=valgrind_output.txt \
    ./test

EXIT_CODE=$?

echo ""
echo "Valgrind output saved to: valgrind_output.txt"
echo ""

# Check for errors
if grep -q "ERROR SUMMARY: 0 errors" valgrind_output.txt; then
    echo "=== NO MEMORY ERRORS DETECTED ==="
else
    echo "=== MEMORY ERRORS DETECTED ==="
    echo "See valgrind_output.txt for details"
    EXIT_CODE=1
fi

# Check for leaks
if grep -q "definitely lost: 0 bytes" valgrind_output.txt; then
    echo "=== NO MEMORY LEAKS DETECTED ==="
else
    echo "=== MEMORY LEAKS DETECTED ==="
    echo "See valgrind_output.txt for details"
    EXIT_CODE=1
fi

exit $EXIT_CODE
