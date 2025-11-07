# Fix for GitHub Issue #4618: GridSample 5D Input Validation

## Quick Summary

**Issue**: ONNX to TensorRT conversion fails for 5D GridSample with cryptic error  
**Fix**: Added clear validation and error message in ONNX parser  
**Status**: ✅ Complete

---

## Problem Statement

### User's Issue
When converting an ONNX model with 5D GridSample operation to TensorRT, users encountered:

```python
data = torch.ones((1, 1, 512, 32, 32), dtype=torch.float32)  # 5D input
grid = torch.ones((1, 512, 32, 32, 3), dtype=torch.float32).cuda()
res = torch.nn.functional.grid_sample(img, grid)
```

**Error Message (Before Fix)**:
```
addGridsample: Error Code 3: API Usage Error
```

### Root Cause Analysis

1. **ONNX Specification**: Supports both 4D (NCHW) and 5D (NCDHW) GridSample
2. **TensorRT API**: Only supports 4D GridSample (documented in `NvInfer.h`)
3. **ONNX Parser**: Missing validation - passed 5D tensors directly to TensorRT API
4. **Result**: Cryptic error from deep within TensorRT's internal validation

---

## Solution Implemented

### Code Change

**File**: `parsers/onnx/onnxOpImporters.cpp`  
**Function**: `DEFINE_BUILTIN_OP_IMPORTER(GridSample)`  
**Location**: Line ~5475

**Added Validation**:
```cpp
// TensorRT only supports 4D GridSample (NCHW format for 2D spatial data)
// ONNX spec supports both 4D and 5D (NCDHW for 3D volumetric data), but TensorRT does not support 5D
ONNXTRT_CHECK_NODE((inputRank == 4),
    "TensorRT only supports 4D GridSample operations (NCHW format). Input tensor has rank "
        << inputRank << ". For 5D volumetric GridSample (NCDHW), consider using a custom plugin or "
        << "reshaping the input to 4D if applicable.",
    node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
```

### What This Does

1. ✅ Validates input tensor rank before calling TensorRT API
2. ✅ Provides clear, actionable error message
3. ✅ Suggests workarounds (custom plugin, reshaping)
4. ✅ Uses proper error code (`ErrorCode::kUNSUPPORTED_NODE`)
5. ✅ Catches error early in parsing phase

---

## Error Message Comparison

### ❌ Before (Cryptic)
```
addGridsample: Error Code 3: API Usage Error
```
- No explanation of what went wrong
- No guidance on how to fix
- Error occurs deep in TensorRT internals

### ✅ After (Clear & Helpful)
```
TensorRT only supports 4D GridSample operations (NCHW format). 
Input tensor has rank 5. For 5D volumetric GridSample (NCDHW), 
consider using a custom plugin or reshaping the input to 4D if applicable.
```
- Clear explanation of the limitation
- Identifies the specific problem (rank 5)
- Provides actionable workarounds
- Error caught early during ONNX parsing

---

## Testing

### Test Files Created

1. **`test_gridsample_5d.py`** - Python script to generate test models
2. **`/tmp/gridsample_5d.onnx`** - 5D GridSample model (should fail with clear error)
3. **`/tmp/gridsample_4d.onnx`** - 4D GridSample model (should succeed)

### Running Tests

```bash
# Generate test models
python3 test_gridsample_5d.py

# Test with TensorRT (after building with fix)
trtexec --onnx=/tmp/gridsample_5d.onnx  # Should show clear error message
trtexec --onnx=/tmp/gridsample_4d.onnx  # Should succeed
```

### Expected Results

| Test Case | Input Shape | Expected Result |
|-----------|-------------|-----------------|
| 5D Model | [1,1,512,32,32] | ❌ Clear error message |
| 4D Model | [1,1,32,32] | ✅ Successful conversion |

---

## Workarounds for Users

### Option 1: Custom TensorRT Plugin ⭐ Recommended for Production

Implement a custom TensorRT plugin that supports 5D GridSample:

```cpp
// Implement IPluginV2DynamicExt for 5D GridSample
class GridSample5DPlugin : public IPluginV2DynamicExt {
    // ... implementation
};
```

### Option 2: Reshape to 4D

If your use case allows, reshape 5D tensors by combining dimensions:

```python
# Example: Combine batch and depth dimensions
# From: [N, C, D, H, W] -> To: [N*D, C, H, W]

import torch

def reshape_5d_to_4d(input_5d, grid_5d):
    N, C, D, H, W = input_5d.shape
    # Reshape input: [N, C, D, H, W] -> [N*D, C, H, W]
    input_4d = input_5d.permute(0, 2, 1, 3, 4).reshape(N*D, C, H, W)
    
    # Reshape grid: [N, D, H, W, 3] -> [N*D, H, W, 2]
    # Note: Need to drop the depth coordinate
    grid_4d = grid_5d.reshape(N*D, H, W, 3)[..., :2]
    
    return input_4d, grid_4d
```

### Option 3: Use Alternative Runtime

For inference, use PyTorch or ONNX Runtime which support 5D GridSample natively:

```python
import onnxruntime as ort

session = ort.InferenceSession("model_with_5d_gridsample.onnx")
outputs = session.run(None, {"input": input_data, "grid": grid_data})
```

---

## Technical Details

### Validation Logic

```
Input Validation Flow:
1. Check input is not scalar (rank > 0) ✓ Already existed
2. Check grid is not scalar (rank > 0) ✓ Already existed  
3. Check input and grid have same rank ✓ Already existed
4. Check input rank is 4 ✅ NEW - Added by this fix
5. Call TensorRT addGridSample API
```

### Error Code Used

- **`ErrorCode::kUNSUPPORTED_NODE`**: Semantically correct for unsupported operation
- Consistent with other dimension validation errors in the codebase

### Backward Compatibility

- ✅ **No breaking changes**: Existing 4D GridSample operations work as before
- ✅ **Additive change**: Only adds validation, doesn't modify existing logic
- ✅ **Safe**: Prevents invalid operations from reaching TensorRT API

---

## Documentation References

1. **TensorRT API Documentation**
   - File: `include/NvInfer.h`
   - Class: `IGridSampleLayer`
   - Quote: "The input and grid tensors must be shape tensors of rank 4."

2. **ONNX Parser Documentation**
   - File: `parsers/onnx/docs/operators.md`
   - GridSample entry: "Input must be 4D input."

3. **ONNX Specification**
   - Supports both 4D and 5D GridSample (opset 16+)
   - Test file: `parsers/onnx/third_party/onnx/onnx/backend/test/case/node/gridsample.py`

---

## Files Modified/Created

### Modified
- ✏️ `parsers/onnx/onnxOpImporters.cpp` - Added validation check

### Created
- 📄 `test_gridsample_5d.py` - Test script
- 📄 `GRIDSAMPLE_5D_FIX.md` - Detailed documentation
- 📄 `FIX_SUMMARY.md` - Summary document
- 📄 `ISSUE_4618_FIX_README.md` - This file

---

## Version Information

- **TensorRT Version**: 10.13.3.9+
- **ONNX Parser Version**: 10.13.0+
- **Fix Date**: November 2025
- **Issue Number**: #4618

---

## Building with the Fix

```bash
# Clone and update submodules
git clone -b main https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive

# Build
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)
```

---

## Verification Checklist

- ✅ Code change implemented correctly
- ✅ Validation logic follows existing patterns
- ✅ Error message is clear and helpful
- ✅ Test models created (4D and 5D)
- ✅ Documentation written
- ✅ Backward compatibility maintained
- ✅ No breaking changes introduced

---

## Contact & Support

For questions or issues related to this fix:
- GitHub Issue: #4618
- TensorRT Forums: https://devtalk.nvidia.com/default/board/304/tensorrt/

---

## License

This fix is part of TensorRT Open Source Software and is licensed under Apache 2.0.
