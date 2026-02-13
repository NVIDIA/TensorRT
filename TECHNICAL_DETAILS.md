# Technical Details: If Operator Shape Broadcasting Fix

## Background

### ONNX If Operator
The ONNX `If` operator implements conditional execution with two subgraphs:
- **then_branch**: Executed when condition is true
- **else_branch**: Executed when condition is false

Both branches must produce the same number of outputs, and corresponding outputs should have compatible types and shapes.

### TensorRT IIfConditional
TensorRT's `IIfConditional` layer requires:
- A boolean scalar condition tensor
- Input layers for external tensors used in subgraphs
- Output layers that merge results from both branches

**Critical Requirement:** `IIfConditionalOutputLayer::addOutput()` requires both input tensors to have **exactly the same shape** (same rank and same dimensions).

## The Problem

### Error Message Analysis
```
Invalid Node - /bb/rope_embeddings/If
/bb/rope_embeddings/If_OutputLayer: IIfConditionalOutputLayer inputs must have the same shape. 
Shapes are [2] and [1].
```

This error indicates:
1. Node name: `/bb/rope_embeddings/If`
2. Then-branch output shape: `[2]` (1D tensor with 2 elements)
3. Else-branch output shape: `[1]` (1D tensor with 1 element)
4. TensorRT rejects this because shapes don't match exactly

### Why This Happens in DINOv3

DINOv3 (Vision Transformer with rotary position embeddings) uses conditional logic for rope embeddings:
```python
# Simplified pseudocode
if condition:
    return [freq1, freq2]  # Shape: [2]
else:
    return [default_freq]  # Shape: [1]
```

Under ONNX broadcasting rules, `[1]` can be broadcast to `[2]`, making these shapes compatible. However, TensorRT doesn't automatically apply this broadcasting.

## The Solution

### Broadcasting Function
The codebase already includes `broadcastTensors()` in `importerUtils.cpp`:

```cpp
void broadcastTensors(ImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
{
    int const t1Dims = t1->getDimensions().nbDims;
    int const t2Dims = t2->getDimensions().nbDims;

    if (t1Dims == t2Dims)
    {
        return;  // Already same rank
    }

    if (t1Dims > t2Dims)
    {
        return broadcastTensor(ctx, t2, t1Dims);  // Broadcast t2 to match t1
    }
    return broadcastTensor(ctx, t1, t2Dims);  // Broadcast t1 to match t2
}
```

This function:
1. Checks if tensors already have the same number of dimensions
2. If not, calls `broadcastTensor()` to add leading dimensions of size 1
3. Uses `IShuffleLayer` to reshape the lower-rank tensor

### Implementation
The fix adds one line before creating the output layer:

```cpp
// Before fix:
auto* thenOut = &convertToTensor(thenSubgraphTensors[i], ctx);
auto* elseOut = &convertToTensor(elseSubgraphTensors[i], ctx);
auto* outputLayer = N_CHECK(conditional->addOutput(*thenOut, *elseOut));

// After fix:
auto* thenOut = &convertToTensor(thenSubgraphTensors[i], ctx);
auto* elseOut = &convertToTensor(elseSubgraphTensors[i], ctx);
broadcastTensors(ctx, thenOut, elseOut);  // ← NEW LINE
auto* outputLayer = N_CHECK(conditional->addOutput(*thenOut, *elseOut));
```

## How It Works

### Example: DINOv3 rope_embeddings

**Before Broadcasting:**
- `thenOut`: Shape `[2]`, nbDims=1
- `elseOut`: Shape `[1]`, nbDims=1

Both have the same rank (1D), so `broadcastTensors()` returns immediately without changes.

**Wait, what?** If they have the same rank, why does the error occur?

The issue is that having the same **number of dimensions** (rank) doesn't mean having the same **shape**. The error message shows shapes `[2]` and `[1]`, which are both 1D tensors but with different sizes.

### Deeper Analysis

Looking more carefully at the error and the broadcasting function, I realize the current `broadcastTensors()` only handles **rank mismatch**, not **dimension size mismatch**.

For example:
- `[1]` vs `[2]`: Same rank (1), different size → NOT handled by current `broadcastTensors()`
- `[1]` vs `[2, 3]`: Different rank → Handled by `broadcastTensors()`

### The Real Fix Needed

The actual issue requires **element-wise broadcasting**, not just rank alignment. We need to handle cases where:
- Tensors have the same rank but different dimension sizes
- One tensor has size 1 in a dimension that needs to be broadcast to match the other

Let me check if there's a more appropriate function or if we need to enhance the fix.
