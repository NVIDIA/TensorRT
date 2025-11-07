# GitHub Issue #4617: Complete Solution Package

## Executive Summary

This document provides a complete solution package for GitHub issue #4617: "Crash if TensorRT 10.1x is used from a dynamic library". The issue causes TensorRT to crash when used from a dynamically loaded library (via `dlopen()`).

**Status**: Solution provided with comprehensive test suite and documentation

**Location**: `/vercel/sandbox/tests/issue_4617_reproducer/`

## Problem Description

When TensorRT is used by a dynamic library loaded with `dlopen()`, the program crashes during cleanup when `dlclose()` is called. This is caused by TensorRT calling `dlclose()` on `libnvinfer_builder_resource.so.10.x` from a static C++ object destructor.

### Affected Versions
- TensorRT 10.13
- TensorRT 10.14

### Environment
- **Platform**: Linux (Ubuntu 24.04.3 LTS)
- **GPU**: RTX 5090
- **Driver**: 580.95.05
- **CUDA**: 13.0

## Root Cause

The crash occurs due to destructor ordering issues in glibc. When a shared library is unloaded:
1. `dlclose()` is called on the user's library
2. Static C++ destructors are executed
3. If TensorRT's static destructor calls `dlclose()` on another library
4. This causes undefined behavior and crashes

## Solution Package Contents

### Test Suite (`tests/issue_4617_reproducer/`)

| File | Description |
|------|-------------|
| `test_main.cpp` | Basic reproducer that loads/unloads TensorRT library |
| `test_library.cpp` | TensorRT wrapper library for dynamic loading |
| `test_dlopen_stress.cpp` | Multi-threaded stress test (100 iterations, 4 threads) |
| `Makefile` | Build system with multiple targets |
| `run_test.sh` | Automated test runner |
| `test_valgrind.sh` | Memory leak detection script |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | Comprehensive documentation |
| `QUICK_START.md` | Quick start guide |
| `ISSUE_ANALYSIS.md` | Technical analysis with proposed fixes |
| `SOLUTION_SUMMARY.md` | Executive summary |
| `.gitignore` | Git ignore file for build artifacts |

## Proposed Fixes

### Primary Solution: Reference-Counted Resource Management

Move resource management from static destructor to `IBuilder` destructor:

```cpp
class BuilderResourceManager {
    static void* handle;
    static std::atomic<int> refCount;
    static std::mutex mutex;
    
public:
    static void acquire() {
        std::lock_guard<std::mutex> lock(mutex);
        if (refCount++ == 0) {
            handle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY);
        }
    }
    
    static void release() {
        std::lock_guard<std::mutex> lock(mutex);
        if (--refCount == 0 && handle) {
            dlclose(handle);
            handle = nullptr;
        }
    }
};

class IBuilder {
public:
    IBuilder() {
        BuilderResourceManager::acquire();
    }
    
    virtual ~IBuilder() noexcept {
        BuilderResourceManager::release();
    }
};
```

**Benefits**:
- Explicit lifetime management
- Thread-safe
- No static destructor issues
- Works correctly with dlopen/dlclose

### Alternative Solution: Use RTLD_NODELETE

Simpler workaround - use `RTLD_NODELETE` flag when loading the builder resource:

```cpp
handle = dlopen("libnvinfer_builder_resource.so.10.x", RTLD_LAZY | RTLD_NODELETE);
```

**Benefits**:
- Simple one-line fix
- Prevents library unload issues

**Drawbacks**:
- Library stays in memory until process exit

## Testing Instructions

### Quick Test
```bash
cd tests/issue_4617_reproducer
./run_test.sh
```

### Build and Run Manually
```bash
cd tests/issue_4617_reproducer
make
LD_LIBRARY_PATH=. ./test
```

### Stress Test
```bash
make test-stress
```

### Memory Leak Detection
```bash
./test_valgrind.sh
```

## Expected Test Results

### Before Fix (Bug Present)
- **Exit Code**: 139 (SIGSEGV)
- **Behavior**: Program crashes during `dlclose()`
- **Valgrind**: Memory errors detected

### After Fix (Bug Fixed)
- **Exit Code**: 0
- **Behavior**: Clean exit with success message
- **Valgrind**: No memory errors
- **Stress Test**: All iterations pass

## Implementation Checklist for NVIDIA

- [ ] Locate static object managing `libnvinfer_builder_resource.so`
- [ ] Remove static destructor calling `dlclose()`
- [ ] Implement `BuilderResourceManager` with reference counting
- [ ] Add resource acquisition to `IBuilder` constructor
- [ ] Add resource release to `IBuilder` destructor
- [ ] Ensure thread safety (mutex/atomic)
- [ ] Run reproducer test suite
- [ ] Test with AddressSanitizer
- [ ] Test with Valgrind
- [ ] Test on multiple Linux distributions
- [ ] Update release notes
- [ ] Update documentation

## User Workarounds

Until the fix is released, users can:

### Option 1: Use RTLD_NODELETE
```cpp
void* handle = dlopen("./my_tensorrt_library.so", RTLD_LAZY | RTLD_NODELETE);
// Use library...
dlclose(handle); // Won't actually unload
```

### Option 2: Avoid dlclose()
```cpp
void* handle = dlopen("./my_tensorrt_library.so", RTLD_LAZY);
// Use library...
// Don't call dlclose() - let OS clean up at process exit
```

### Option 3: Static Linking
Link TensorRT statically if possible to avoid dynamic loading issues.

## Technical References

- **dlopen man page**: https://man7.org/linux/man-pages/man3/dlopen.3.html
- **GCC destructor attribute**: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
- **C++ Static Initialization Order**: https://en.cppreference.com/w/cpp/language/siof
- **GitHub Issue**: #4617

## Files Created

```
/vercel/sandbox/tests/issue_4617_reproducer/
├── .gitignore
├── ISSUE_ANALYSIS.md          # Technical analysis
├── Makefile                    # Build system
├── QUICK_START.md             # Quick start guide
├── README.md                   # Main documentation
├── SOLUTION_SUMMARY.md        # Executive summary
├── run_test.sh                # Test runner
├── test_dlopen_stress.cpp     # Stress test
├── test_library.cpp           # TensorRT wrapper
├── test_main.cpp              # Main reproducer
└── test_valgrind.sh           # Memory test
```

## Next Steps

1. **For NVIDIA TensorRT Team**:
   - Review the proposed solutions
   - Implement the fix in TensorRT core library
   - Run the test suite to verify the fix
   - Include in next TensorRT release

2. **For Users Experiencing the Issue**:
   - Use the test suite to verify the issue
   - Apply one of the workarounds
   - Monitor for TensorRT updates

3. **For Contributors**:
   - Test on different Linux distributions
   - Test with different TensorRT versions
   - Report results on GitHub issue #4617

## Contact and Support

- **GitHub Issue**: #4617
- **Test Suite Location**: `/vercel/sandbox/tests/issue_4617_reproducer/`
- **Documentation**: See files in test suite directory

## License

All test code and documentation:
- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
- SPDX-License-Identifier: Apache-2.0

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-07  
**Status**: Complete - Ready for Review
