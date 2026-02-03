# Quick Start Guide - Issue #4617 Test Suite

## TL;DR

This test suite reproduces and helps verify the fix for GitHub issue #4617: TensorRT crashes when used from a dynamically loaded library.

## Quick Test

```bash
cd tests/issue_4617_reproducer
./run_test.sh
```

If you see "TEST PASSED", the issue is either fixed or not present in your TensorRT version.
If the program crashes (exit code 139), the issue is present.

## What's Included

| File | Purpose |
|------|---------|
| `test_main.cpp` | Basic reproducer - loads/unloads TensorRT library |
| `test_library.cpp` | TensorRT wrapper library |
| `test_dlopen_stress.cpp` | Stress test with multiple threads |
| `run_test.sh` | Automated test runner |
| `test_valgrind.sh` | Memory leak detection |
| `Makefile` | Build system |
| `README.md` | Detailed documentation |
| `ISSUE_ANALYSIS.md` | Technical analysis and proposed fixes |
| `SOLUTION_SUMMARY.md` | Executive summary |

## Prerequisites

- TensorRT 10.13 or later installed
- GCC/G++ compiler
- Make
- (Optional) Valgrind for memory testing

## Building

```bash
# If TensorRT is in a custom location
export TRT_LIBPATH=/path/to/TensorRT-10.13.3.9

# Build all tests
make

# Or build and run
make test-run
```

## Running Tests

### Basic Test
```bash
./run_test.sh
```

### Stress Test (100 iterations, 4 threads)
```bash
make test-stress
```

### Memory Leak Detection
```bash
./test_valgrind.sh
```

## Expected Results

### If Bug is Present
- Program crashes during library unload
- Exit code: 139 (SIGSEGV)
- Error message about segmentation fault

### If Bug is Fixed
- Program exits cleanly
- Exit code: 0
- Message: "TEST PASSED"

## Understanding the Issue

The crash occurs because TensorRT calls `dlclose()` on `libnvinfer_builder_resource.so` from a static C++ destructor. When your library is unloaded with `dlclose()`, this causes a crash due to destructor ordering issues in glibc.

## Recommended Fix

Move the `dlclose()` call from a static destructor to the `IBuilder` destructor or use `__attribute__((destructor))`. See `ISSUE_ANALYSIS.md` for detailed solutions.

## Workaround for Users

Until the fix is released:

```cpp
// Option 1: Use RTLD_NODELETE when loading your library
void* handle = dlopen("./my_lib.so", RTLD_LAZY | RTLD_NODELETE);

// Option 2: Don't call dlclose() - let OS clean up at process exit
// (Just don't call dlclose on the handle)
```

## Troubleshooting

### "Failed to load library"
- Check that TensorRT is installed
- Set `TRT_LIBPATH` environment variable
- Verify `LD_LIBRARY_PATH` includes TensorRT lib directory

### "Failed to create TensorRT builder"
- Ensure CUDA is installed and working
- Check NVIDIA driver version
- Verify GPU is accessible

### Build errors
- Install build-essential: `sudo apt-get install build-essential`
- Check GCC version: `gcc --version` (need 7.0+)

## More Information

- **Detailed Analysis**: See `ISSUE_ANALYSIS.md`
- **Solution Summary**: See `SOLUTION_SUMMARY.md`
- **Full Documentation**: See `README.md`
- **GitHub Issue**: #4617

## Support

For questions or issues, please comment on GitHub issue #4617.
