# Makefile for TensorRT Issue #4617 Reproducer
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compiler settings
CXX := g++
CXXFLAGS := -std=c++14 -Wall -Wextra -fPIC

# TensorRT paths - adjust these based on your installation
TRT_INCLUDE_DIR ?= /usr/include/x86_64-linux-gnu
TRT_LIB_DIR ?= /usr/lib/x86_64-linux-gnu

# If TRT_LIBPATH is set (from environment), use it
ifdef TRT_LIBPATH
    TRT_INCLUDE_DIR := $(TRT_LIBPATH)/include
    TRT_LIB_DIR := $(TRT_LIBPATH)/lib
endif

# Include and library flags
INCLUDES := -I$(TRT_INCLUDE_DIR) -I../../../include
LDFLAGS := -L$(TRT_LIB_DIR) -Wl,-rpath,$(TRT_LIB_DIR)
LIBS := -lnvinfer -ldl

# Targets
.PHONY: all clean test test-run test-stress

all: test test_stress libtest_tensorrt.so

# Build the test library
libtest_tensorrt.so: test_library.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -o $@ $< $(LDFLAGS) $(LIBS)

# Build the main test program
test: test_main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -ldl

# Build the stress test program
test_stress: test_dlopen_stress.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -ldl -pthread

# Run the basic test
test-run: all
	@echo "Running basic test with LD_LIBRARY_PATH=$(TRT_LIB_DIR):."
	LD_LIBRARY_PATH=$(TRT_LIB_DIR):. ./test

# Run the stress test
test-stress: all
	@echo "Running stress test with LD_LIBRARY_PATH=$(TRT_LIB_DIR):."
	LD_LIBRARY_PATH=$(TRT_LIB_DIR):. ./test_stress

# Clean build artifacts
clean:
	rm -f test test_stress libtest_tensorrt.so *.o valgrind_output.txt

# Help target
help:
	@echo "TensorRT Issue #4617 Reproducer Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all targets (default)"
	@echo "  test         - Build the basic test executable"
	@echo "  test_stress  - Build the stress test executable"
	@echo "  test-run     - Build and run the basic test"
	@echo "  test-stress  - Build and run the stress test"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment Variables:"
	@echo "  TRT_LIBPATH      - Path to TensorRT installation (optional)"
	@echo "  TRT_INCLUDE_DIR  - Path to TensorRT headers (default: /usr/include/x86_64-linux-gnu)"
	@echo "  TRT_LIB_DIR      - Path to TensorRT libraries (default: /usr/lib/x86_64-linux-gnu)"
	@echo ""
	@echo "Example:"
	@echo "  make TRT_LIBPATH=/path/to/TensorRT-10.13.3.9"
	@echo "  make test-run"
	@echo "  make test-stress"
