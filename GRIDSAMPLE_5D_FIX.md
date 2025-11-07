# Fix for GitHub Issue #4618: GridSample 5D Input Validation

## Problem Description

When attempting to convert an ONNX model with 5D GridSample operation to TensorRT, users encountered a cryptic error:

```
addGridsample: Error Code 3: API Usage Error
```

### Root Cause

- **ONNX Specification**: Supports both 4D (NCHW) and 5D (NCDHW) GridSample operations
- **TensorRT API**: Only supports 4D GridSample operations
- **ONNX Parser**: Did not validate input dimensions before calling TensorRT's `addGridSample()` API

This resulted in the error being caught deep in TensorRT's internal validation, producing an unhelpful error message.

## Solution

Added explicit validation in the ONNX parser to check that GridSample inputs are 4D before attempting to create the TensorRT layer.

### Code Changes

**File**: `parsers/onnx/onnxOpImporters.cpp`

**Location**: `DEFINE_BUILTIN_OP_IMPORTER(GridSample)` function (around line 5470)

**Change**: Added validation check after rank equality validation:

```cpp
// TensorRT only supports 4D GridSample (NCHW format for 2D spatial data)
// ONNX spec supports both 4D and 5D (NCDHW for 3D volumetric data), but TensorRT does not support 5D
ONNXTRT_CHECK_NODE((inputRank == 4),
    "TensorRT only supports 4D GridSample operations (NCHW format). Input tensor has rank "
        << inputRank << ". For 5D volumetric GridSample (NCDHW), consider using a custom plugin or "
        << "reshaping the input to 4D if applicable.",
    node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
```

## Benefits

1. **Clear Error Message**: Users now get a descriptive error explaining the limitation
2. **Early Detection**: Error is caught during ONNX parsing, not deep in TensorRT internals
3. **Helpful Guidance**: Error message suggests workarounds (custom plugin or reshaping)
4. **Proper Error Code**: Uses `ErrorCode::kUNSUPPORTED_NODE` which is semantically correct

## Error Message Comparison

### Before (Cryptic)
```
addGridsample: Error Code 3: API Usage Error
```

### After (Clear and Helpful)
```
TensorRT only supports 4D GridSample operations (NCHW format). Input tensor has rank 5. 
For 5D volumetric GridSample (NCDHW), consider using a custom plugin or reshaping the 
input to 4D if applicable.
```

## Testing

### Test Models Created

Two ONNX test models have been created:

1. **5D GridSample Model** (`/tmp/gridsample_5d.onnx`)
   - Input shape: [1, 1, 512, 32, 32] (5D - NCDHW)
   - Grid shape: [1, 512, 32, 32, 3] (5D)
   - Expected: Should fail with clear error message

2. **4D GridSample Model** (`/tmp/gridsample_4d.onnx`)
   - Input shape: [1, 1, 32, 32] (4D - NCHW)
   - Grid shape: [1, 32, 32, 2] (4D)
   - Expected: Should parse successfully

### How to Test

```bash
# Generate test models
python3 test_gridsample_5d.py

# Build TensorRT with the fix
cd /path/to/TensorRT
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)

# Test with trtexec or Python API
trtexec --onnx=/tmp/gridsample_5d.onnx  # Should fail with clear message
trtexec --onnx=/tmp/gridsample_4d.onnx  # Should succeed
```

## Workarounds for Users

If you need 5D GridSample functionality, consider these options:

### Option 1: Custom TensorRT Plugin
Implement a custom TensorRT plugin that supports 5D GridSample operations.

### Option 2: Reshape to 4D (if applicable)
If your use case allows, reshape the 5D tensor to 4D by combining dimensions:
```python
# Example: Combine batch and depth dimensions
# From: [N, C, D, H, W] -> To: [N*D, C, H, W]
```

### Option 3: Use PyTorch/ONNX Runtime
For inference, use PyTorch or ONNX Runtime which support 5D GridSample natively.

## Related Documentation

- **TensorRT API**: `include/NvInfer.h` - `IGridSampleLayer` documentation
- **ONNX Parser**: `parsers/onnx/docs/operators.md` - GridSample operator limitations
- **ONNX Spec**: Supports both 4D and 5D GridSample (opset 16+)

## Impact

- **Breaking Change**: No - this only adds validation, doesn't change existing behavior
- **Backward Compatible**: Yes - 4D GridSample operations continue to work as before
- **User Experience**: Significantly improved - clear error messages instead of cryptic API errors

## Version Information

- **TensorRT Version**: 10.13.3.9+
- **ONNX Parser Version**: 10.13.0+
- **Fix Date**: November 2025
