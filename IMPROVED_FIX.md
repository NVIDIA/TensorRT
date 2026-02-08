# Improved Fix for Issue #4603

## Problem Analysis

The original fix using `broadcastTensors()` is **insufficient** because:

1. `broadcastTensors()` only handles **rank mismatch** (e.g., `[3]` vs `[2, 3]`)
2. It does NOT handle **dimension size mismatch** (e.g., `[1]` vs `[2]`)
3. The DINOv3 error shows shapes `[2]` and `[1]` - same rank, different size

## Root Cause

TensorRT's `IIfConditionalOutputLayer` requires **exact shape match**, but:
- ONNX allows broadcasting where dimension size 1 can expand to any size N
- Example: `[1]` can broadcast to `[2]` by repeating the element

## Correct Solution

We need to implement full ONNX-style broadcasting using the pattern from the `Expand` operator:

### Algorithm
1. Align ranks by prepending dimensions of size 1
2. For each dimension, compute the broadcast size: `max(dim1, dim2)`
3. For each tensor, if `dim[i] == 1` and `broadcast_dim[i] > 1`, use stride=0 to expand
4. Use `ISliceLayer` with computed strides to perform the expansion

### Implementation Options

#### Option 1: Create a new `broadcastTensorsForConditional()` function
Add to `importerUtils.cpp`:

```cpp
void broadcastTensorsForConditional(ImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
{
    // First, align ranks
    broadcastTensors(ctx, t1, t2);
    
    // Now both tensors have the same rank
    auto shape1 = shapeOf(*t1);
    auto shape2 = shapeOf(*t2);
    
    // Compute broadcast shape: max(shape1, shape2) element-wise
    ShapeTensor broadcastShape = broadcast(ctx, shape1, shape2);
    
    // Expand t1 if needed
    if (!shape1.allValuesKnown() || !broadcastShape.allValuesKnown() || 
        !std::equal(shape1.begin(), shape1.end(), broadcastShape.begin()))
    {
        // Compute strides: stride[i] = (shape1[i] > 1) ? 1 : 0
        ShapeTensor const zero = similar(ctx, shape1, 0);
        ShapeTensor const one = similar(ctx, shape1, 1);
        ShapeTensor const strides1 = min(ctx, one, max(ctx, sub(ctx, shape1, one), zero));
        
        nvinfer1::ISliceLayer* slice1 = addSlice(ctx, *t1, zero, broadcastShape, strides1);
        ctx->registerLayer(slice1, "ONNXTRT_BroadcastForConditional", nullptr);
        t1 = N_CHECK(slice1->getOutput(0));
    }
    
    // Expand t2 if needed
    if (!shape2.allValuesKnown() || !broadcastShape.allValuesKnown() || 
        !std::equal(shape2.begin(), shape2.end(), broadcastShape.begin()))
    {
        // Compute strides: stride[i] = (shape2[i] > 1) ? 1 : 0
        ShapeTensor const zero = similar(ctx, shape2, 0);
        ShapeTensor const one = similar(ctx, shape2, 1);
        ShapeTensor const strides2 = min(ctx, one, max(ctx, sub(ctx, shape2, one), zero));
        
        nvinfer1::ISliceLayer* slice2 = addSlice(ctx, *t2, zero, broadcastShape, strides2);
        ctx->registerLayer(slice2, "ONNXTRT_BroadcastForConditional", nullptr);
        t2 = N_CHECK(slice2->getOutput(0));
    }
}
```

#### Option 2: Simpler approach using element-wise trick
Since TensorRT's `IElementWiseLayer` handles broadcasting automatically, we can:
1. Add a dummy element-wise operation (e.g., multiply by 1)
2. Let TensorRT handle the broadcasting
3. Extract the broadcast result

However, this is a hack and may not work reliably.

#### Option 3: Use existing Expand logic inline
Directly apply the Expand operator's logic in the If operator:

```cpp
for (size_t i = 0; i < thenSubgraphTensors.size(); i++)
{
    auto* thenOut = &convertToTensor(thenSubgraphTensors[i], ctx);
    auto* elseOut = &convertToTensor(elseSubgraphTensors[i], ctx);
    
    // Align ranks first
    broadcastTensors(ctx, thenOut, elseOut);
    
    // Now handle dimension size broadcasting
    auto thenShape = shapeOf(*thenOut);
    auto elseShape = shapeOf(*elseOut);
    ShapeTensor broadcastShape = broadcast(ctx, thenShape, elseShape);
    
    // Expand thenOut if needed
    {
        ShapeTensor const zero = similar(ctx, thenShape, 0);
        ShapeTensor const one = similar(ctx, thenShape, 1);
        ShapeTensor const strides = min(ctx, one, max(ctx, sub(ctx, thenShape, one), zero));
        nvinfer1::ISliceLayer* slice = addSlice(ctx, *thenOut, zero, broadcastShape, strides);
        ctx->registerLayer(slice, "ONNXTRT_BroadcastThen", nullptr);
        thenOut = N_CHECK(slice->getOutput(0));
    }
    
    // Expand elseOut if needed
    {
        ShapeTensor const zero = similar(ctx, elseShape, 0);
        ShapeTensor const one = similar(ctx, elseShape, 1);
        ShapeTensor const strides = min(ctx, one, max(ctx, sub(ctx, elseShape, one), zero));
        nvinfer1::ISliceLayer* slice = addSlice(ctx, *elseOut, zero, broadcastShape, strides);
        ctx->registerLayer(slice, "ONNXTRT_BroadcastElse", nullptr);
        elseOut = N_CHECK(slice->getOutput(0));
    }
    
    auto* outputLayer = N_CHECK(conditional->addOutput(*thenOut, *elseOut));
    ctx->registerLayer(outputLayer, std::string(conditional->getName()) + "_OutputLayer", &node);
    graphOutputs.emplace_back(N_CHECK(outputLayer->getOutput(0)));
}
```

## Recommended Approach

**Option 3** is recommended because:
1. It's self-contained in the If operator
2. Uses proven logic from the Expand operator
3. Handles both rank and dimension size broadcasting
4. No need to modify importerUtils.cpp

## Testing

Test cases needed:
1. `[1]` vs `[2]` - dimension size broadcast
2. `[1, 3]` vs `[2, 3]` - partial dimension broadcast
3. `[3]` vs `[2, 3]` - rank broadcast
4. `[1]` vs `[2, 3]` - combined rank and dimension broadcast
5. `[2, 3]` vs `[2, 3]` - no broadcast needed (existing case)
