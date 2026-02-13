# Issue #4617 Test Suite - Document Index

## Overview

This directory contains a complete test suite and solution package for GitHub issue #4617: "Crash if TensorRT 10.1x is used from a dynamic library".

## Quick Navigation

### 🚀 Getting Started
- **[QUICK_START.md](QUICK_START.md)** - Start here! Quick guide to run tests
- **[README.md](README.md)** - Main documentation with build instructions

### 📋 Documentation
- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Executive summary of the issue and solutions
- **[ISSUE_ANALYSIS.md](ISSUE_ANALYSIS.md)** - Deep technical analysis with proposed fixes
- **[INDEX.md](INDEX.md)** - This file - navigation guide

### 🧪 Test Code
- **[test_main.cpp](test_main.cpp)** - Basic reproducer (loads/unloads library)
- **[test_library.cpp](test_library.cpp)** - TensorRT wrapper library
- **[test_dlopen_stress.cpp](test_dlopen_stress.cpp)** - Multi-threaded stress test

### 🔧 Build & Run
- **[Makefile](Makefile)** - Build system
- **[run_test.sh](run_test.sh)** - Automated test runner
- **[test_valgrind.sh](test_valgrind.sh)** - Memory leak detection
- **[.gitignore](.gitignore)** - Git ignore patterns

## Document Purpose Guide

### For Users Experiencing the Issue
1. Start with **QUICK_START.md** to run the test
2. Read **SOLUTION_SUMMARY.md** for workarounds
3. Check **README.md** for detailed instructions

### For Developers Fixing the Issue
1. Read **ISSUE_ANALYSIS.md** for technical details
2. Review the proposed solutions
3. Run the test suite to verify the fix
4. Check **SOLUTION_SUMMARY.md** for implementation checklist

### For QA/Testing
1. Use **run_test.sh** for basic testing
2. Run **test_valgrind.sh** for memory testing
3. Use `make test-stress` for stress testing
4. Refer to **README.md** for expected results

### For Documentation/Release Notes
1. **SOLUTION_SUMMARY.md** has the executive summary
2. **ISSUE_ANALYSIS.md** has technical details
3. **README.md** has user-facing information

## Test Suite Components

### Basic Test (`test_main.cpp` + `test_library.cpp`)
- **Purpose**: Reproduce the crash
- **What it does**: Loads TensorRT library, creates builder, unloads library
- **Run with**: `./run_test.sh` or `make test-run`
- **Expected**: Crash if bug present, clean exit if fixed

### Stress Test (`test_dlopen_stress.cpp`)
- **Purpose**: Verify fix under load
- **What it does**: 100 iterations × 4 threads of load/unload
- **Run with**: `make test-stress`
- **Expected**: All iterations pass without crashes

### Memory Test (`test_valgrind.sh`)
- **Purpose**: Detect memory leaks and errors
- **What it does**: Runs basic test under Valgrind
- **Run with**: `./test_valgrind.sh`
- **Expected**: No memory errors or leaks

## Build Targets

```bash
make              # Build all tests
make test         # Build basic test only
make test_stress  # Build stress test only
make test-run     # Build and run basic test
make test-stress  # Build and run stress test
make clean        # Remove build artifacts
make help         # Show help message
```

## Environment Variables

- `TRT_LIBPATH` - Path to TensorRT installation (optional)
- `TRT_INCLUDE_DIR` - Path to TensorRT headers
- `TRT_LIB_DIR` - Path to TensorRT libraries

## File Sizes and Complexity

| File | Lines | Purpose | Complexity |
|------|-------|---------|------------|
| test_main.cpp | ~100 | Basic reproducer | Simple |
| test_library.cpp | ~80 | TensorRT wrapper | Simple |
| test_dlopen_stress.cpp | ~120 | Stress test | Medium |
| ISSUE_ANALYSIS.md | ~400 | Technical docs | Detailed |
| SOLUTION_SUMMARY.md | ~250 | Executive summary | Medium |
| README.md | ~80 | User guide | Simple |
| QUICK_START.md | ~150 | Quick guide | Simple |

## Testing Workflow

```
┌─────────────────┐
│  Quick Start    │
│  QUICK_START.md │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Run Basic Test │
│  ./run_test.sh  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│ PASS  │ │  FAIL    │
└───┬───┘ └────┬─────┘
    │          │
    │          ▼
    │    ┌──────────────┐
    │    │ Bug Present  │
    │    │ See Analysis │
    │    └──────────────┘
    │
    ▼
┌─────────────────┐
│  Run Stress     │
│  make test-     │
│  stress         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Run Valgrind   │
│  ./test_        │
│  valgrind.sh    │
└─────────────────┘
```

## Issue Status

- **Reported**: GitHub Issue #4617
- **Affected Versions**: TensorRT 10.13, 10.14
- **Platform**: Linux
- **Status**: Solution provided, awaiting implementation
- **Test Suite**: Complete
- **Documentation**: Complete

## Related Files in Repository

- `/vercel/sandbox/ISSUE_4617_SOLUTION.md` - Top-level solution document
- `/vercel/sandbox/include/NvInfer.h` - TensorRT API header
- `/vercel/sandbox/samples/` - Other TensorRT samples

## Support and Contact

- **GitHub Issue**: #4617
- **Test Suite Location**: `/vercel/sandbox/tests/issue_4617_reproducer/`
- **For Questions**: Comment on GitHub issue #4617

## Version History

- **v1.0** (2025-11-07): Initial release
  - Complete test suite
  - Comprehensive documentation
  - Multiple test scenarios
  - Proposed solutions

---

**Last Updated**: 2025-11-07  
**Maintainer**: TensorRT OSS Community  
**License**: Apache-2.0
