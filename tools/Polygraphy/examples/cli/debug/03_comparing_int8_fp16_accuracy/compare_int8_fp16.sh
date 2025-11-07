#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Complete workflow script for comparing INT8 and FP16 accuracy
# and fixing problematic layers

set -e  # Exit on error

# Configuration
MODEL="model.onnx"
CALIBRATION_SCRIPT="calibration_data_loader.py"
CALIBRATION_CACHE="calibration.cache"
FP16_ENGINE="model_fp16.engine"
INT8_ENGINE="model_int8.engine"
INT8_FIXED_ENGINE="model_int8_fixed.engine"
TEST_INPUTS="test_inputs.json"
FP16_OUTPUTS="fp16_outputs.json"
INT8_OUTPUTS="int8_outputs.json"
FP16_LAYER_OUTPUTS="fp16_layer_outputs.json"
INT8_LAYER_OUTPUTS="int8_layer_outputs.json"
GOLDEN_OUTPUTS="golden_outputs.json"
FIX_SCRIPT="fix_precision.py"

echo "=========================================="
echo "INT8 vs FP16 Accuracy Comparison Workflow"
echo "=========================================="
echo ""

# Step 1: Build FP16 engine
echo "Step 1: Building FP16 engine..."
polygraphy convert "$MODEL" \
    --fp16 \
    -o "$FP16_ENGINE"
echo "✓ FP16 engine built: $FP16_ENGINE"
echo ""

# Step 2: Build INT8 engine with calibration
echo "Step 2: Building INT8 engine with calibration..."
if [ ! -f "$CALIBRATION_CACHE" ]; then
    echo "Generating calibration cache..."
    polygraphy convert "$MODEL" \
        --int8 \
        --data-loader-script "$CALIBRATION_SCRIPT" \
        --calibration-cache "$CALIBRATION_CACHE" \
        -o "$INT8_ENGINE"
else
    echo "Using existing calibration cache: $CALIBRATION_CACHE"
    polygraphy convert "$MODEL" \
        --int8 \
        --calibration-cache "$CALIBRATION_CACHE" \
        -o "$INT8_ENGINE"
fi
echo "✓ INT8 engine built: $INT8_ENGINE"
echo ""

# Step 3: Generate test inputs
echo "Step 3: Generating test inputs..."
polygraphy run "$MODEL" --onnxrt \
    --save-inputs "$TEST_INPUTS"
echo "✓ Test inputs saved: $TEST_INPUTS"
echo ""

# Step 4: Compare overall accuracy
echo "Step 4: Comparing overall accuracy..."
echo "Running FP16 engine..."
polygraphy run "$FP16_ENGINE" --trt \
    --load-inputs "$TEST_INPUTS" \
    --save-outputs "$FP16_OUTPUTS"

echo "Running INT8 engine and comparing..."
if polygraphy run "$INT8_ENGINE" --trt \
    --load-inputs "$TEST_INPUTS" \
    --save-outputs "$INT8_OUTPUTS" \
    --load-outputs "$FP16_OUTPUTS"; then
    echo "✓ INT8 accuracy is acceptable!"
    echo "No further action needed."
    exit 0
else
    echo "✗ INT8 accuracy is not acceptable. Proceeding with layer-wise analysis..."
fi
echo ""

# Step 5: Compare layer-wise outputs
echo "Step 5: Comparing layer-wise outputs..."
echo "Saving all FP16 layer outputs..."
polygraphy run "$MODEL" --trt --fp16 \
    --trt-outputs mark all \
    --load-inputs "$TEST_INPUTS" \
    --save-outputs "$FP16_LAYER_OUTPUTS"

echo "Saving all INT8 layer outputs..."
polygraphy run "$MODEL" --trt --int8 \
    --calibration-cache "$CALIBRATION_CACHE" \
    --trt-outputs mark all \
    --load-inputs "$TEST_INPUTS" \
    --save-outputs "$INT8_LAYER_OUTPUTS"

echo "✓ Layer outputs saved"
echo ""

# Step 6: Analyze layer errors
echo "Step 6: Analyzing layer-wise errors..."
python3 parse_layer_errors.py \
    --fp16-outputs "$FP16_LAYER_OUTPUTS" \
    --int8-outputs "$INT8_LAYER_OUTPUTS" \
    --threshold 0.1 \
    --top-k 10
echo ""

# Step 7: Use debug precision tool (optional, automated approach)
echo "Step 7: Using debug precision tool to automatically identify problematic layers..."
echo "Generating golden outputs from ONNX-Runtime..."
polygraphy run "$MODEL" --onnxrt \
    --load-inputs "$TEST_INPUTS" \
    --save-outputs "$GOLDEN_OUTPUTS"

echo "Running debug precision tool..."
echo "This may take some time as it iteratively builds engines..."
polygraphy debug precision "$MODEL" \
    --int8 \
    --calibration-cache "$CALIBRATION_CACHE" \
    --precision float32 \
    --mode bisect \
    --dir forward \
    --check "polygraphy run polygraphy_debug.engine --trt \
        --load-inputs $TEST_INPUTS \
        --load-outputs $GOLDEN_OUTPUTS" || true
echo ""

# Step 8: Manual fix (if debug precision didn't work or for fine-tuning)
echo "Step 8: Applying precision constraints..."
echo "Please edit $FIX_SCRIPT to add the problematic layer names identified above."
echo "Press Enter to continue after editing, or Ctrl+C to exit..."
read -r

if [ -f "$FIX_SCRIPT" ]; then
    echo "Building INT8 engine with precision constraints..."
    polygraphy convert "$MODEL" \
        --int8 \
        --calibration-cache "$CALIBRATION_CACHE" \
        --trt-network-postprocess-script "$FIX_SCRIPT" \
        --precision-constraints obey \
        -o "$INT8_FIXED_ENGINE"
    echo "✓ Fixed INT8 engine built: $INT8_FIXED_ENGINE"
    echo ""
    
    # Step 9: Verify the fix
    echo "Step 9: Verifying the fixed engine..."
    if polygraphy run "$INT8_FIXED_ENGINE" --trt \
        --load-inputs "$TEST_INPUTS" \
        --load-outputs "$GOLDEN_OUTPUTS"; then
        echo "✓ Fixed INT8 engine has acceptable accuracy!"
    else
        echo "✗ Fixed INT8 engine still has accuracy issues."
        echo "Consider:"
        echo "  1. Adding more layers to FP32"
        echo "  2. Improving calibration data quality"
        echo "  3. Using Quantization-Aware Training (QAT)"
    fi
else
    echo "✗ $FIX_SCRIPT not found. Please create it based on the analysis above."
fi
echo ""

# Step 10: Performance comparison
echo "Step 10: Performance comparison..."
echo "FP16 engine performance:"
trtexec --loadEngine="$FP16_ENGINE" --iterations=100 --avgRuns=10 || true
echo ""

echo "INT8 engine performance:"
trtexec --loadEngine="$INT8_ENGINE" --iterations=100 --avgRuns=10 || true
echo ""

if [ -f "$INT8_FIXED_ENGINE" ]; then
    echo "Fixed INT8 engine performance:"
    trtexec --loadEngine="$INT8_FIXED_ENGINE" --iterations=100 --avgRuns=10 || true
    echo ""
fi

echo "=========================================="
echo "Workflow complete!"
echo "=========================================="
echo ""
echo "Summary of generated files:"
echo "  - $FP16_ENGINE: FP16 baseline engine"
echo "  - $INT8_ENGINE: Original INT8 engine"
echo "  - $INT8_FIXED_ENGINE: Fixed INT8 engine (if created)"
echo "  - $CALIBRATION_CACHE: Calibration cache (reusable)"
echo "  - $FP16_LAYER_OUTPUTS: FP16 layer-wise outputs"
echo "  - $INT8_LAYER_OUTPUTS: INT8 layer-wise outputs"
echo ""
echo "Next steps:"
echo "  1. Review the layer error analysis output"
echo "  2. Update $FIX_SCRIPT with problematic layer names"
echo "  3. Re-run this script or manually build the fixed engine"
echo "  4. Validate on your full test dataset"
echo ""
