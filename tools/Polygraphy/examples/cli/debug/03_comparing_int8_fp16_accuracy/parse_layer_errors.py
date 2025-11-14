#!/usr/bin/env python3
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

"""
Script to parse and analyze layer-wise errors between INT8 and FP16 TensorRT engines.

This script loads Polygraphy JSON output files from INT8 and FP16 engine runs,
computes error metrics for each layer, and identifies layers with the highest errors.
"""

import argparse
import numpy as np
from polygraphy.comparator import RunResults
from typing import Dict, Tuple


def compute_error_metrics(fp16_array: np.ndarray, int8_array: np.ndarray) -> Dict[str, float]:
    """
    Compute various error metrics between two arrays.
    
    Args:
        fp16_array: Reference FP16 output
        int8_array: INT8 output to compare
        
    Returns:
        Dictionary containing error metrics
    """
    # Ensure arrays have the same shape
    if fp16_array.shape != int8_array.shape:
        print(f"Warning: Shape mismatch - FP16: {fp16_array.shape}, INT8: {int8_array.shape}")
        return None
    
    # Compute absolute difference
    abs_diff = np.abs(fp16_array - int8_array)
    
    # Compute relative difference (avoid division by zero)
    rel_diff = abs_diff / (np.abs(fp16_array) + 1e-8)
    
    # Compute various statistics
    metrics = {
        'max_abs_error': float(np.max(abs_diff)),
        'mean_abs_error': float(np.mean(abs_diff)),
        'median_abs_error': float(np.median(abs_diff)),
        'std_abs_error': float(np.std(abs_diff)),
        'max_rel_error': float(np.max(rel_diff)),
        'mean_rel_error': float(np.mean(rel_diff)),
        'median_rel_error': float(np.median(rel_diff)),
        'std_rel_error': float(np.std(rel_diff)),
        'fp16_mean': float(np.mean(fp16_array)),
        'fp16_std': float(np.std(fp16_array)),
        'int8_mean': float(np.mean(int8_array)),
        'int8_std': float(np.std(int8_array)),
        'cosine_similarity': float(
            np.dot(fp16_array.flatten(), int8_array.flatten()) /
            (np.linalg.norm(fp16_array.flatten()) * np.linalg.norm(int8_array.flatten()) + 1e-8)
        ),
    }
    
    return metrics


def analyze_layer_errors(
    fp16_json: str,
    int8_json: str,
    threshold: float = 0.1,
    top_k: int = 10
) -> None:
    """
    Analyze and report layer-wise errors between INT8 and FP16 engines.
    
    Args:
        fp16_json: Path to FP16 outputs JSON file
        int8_json: Path to INT8 outputs JSON file
        threshold: Threshold for flagging high errors
        top_k: Number of top error layers to report
    """
    print("=" * 80)
    print("INT8 vs FP16 Layer-wise Error Analysis")
    print("=" * 80)
    print()
    
    # Load the JSON files
    print(f"Loading FP16 outputs from: {fp16_json}")
    fp16_results = RunResults.load(fp16_json)
    
    print(f"Loading INT8 outputs from: {int8_json}")
    int8_results = RunResults.load(int8_json)
    print()
    
    # Get runner names
    fp16_runner = list(fp16_results.keys())[0]
    int8_runner = list(int8_results.keys())[0]
    
    print(f"FP16 Runner: {fp16_runner}")
    print(f"INT8 Runner: {int8_runner}")
    print()
    
    # Get outputs from first iteration
    fp16_outputs = fp16_results[fp16_runner][0]
    int8_outputs = int8_results[int8_runner][0]
    
    # Compute errors for each layer
    layer_errors = {}
    
    print("Computing error metrics for each layer...")
    print()
    
    for layer_name in fp16_outputs.keys():
        if layer_name not in int8_outputs:
            print(f"Warning: Layer '{layer_name}' not found in INT8 outputs")
            continue
        
        fp16_array = fp16_outputs[layer_name]
        int8_array = int8_outputs[layer_name]
        
        metrics = compute_error_metrics(fp16_array, int8_array)
        if metrics is not None:
            layer_errors[layer_name] = metrics
    
    # Sort layers by max absolute error
    sorted_layers = sorted(
        layer_errors.items(),
        key=lambda x: x[1]['max_abs_error'],
        reverse=True
    )
    
    # Report summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total layers analyzed: {len(layer_errors)}")
    
    # Count layers exceeding threshold
    high_error_layers = [
        name for name, metrics in layer_errors.items()
        if metrics['max_abs_error'] > threshold
    ]
    print(f"Layers with max absolute error > {threshold}: {len(high_error_layers)}")
    print()
    
    # Report top-k layers with highest errors
    print("=" * 80)
    print(f"TOP {top_k} LAYERS WITH HIGHEST ERRORS")
    print("=" * 80)
    print()
    
    for i, (layer_name, metrics) in enumerate(sorted_layers[:top_k], 1):
        print(f"{i}. Layer: {layer_name}")
        print(f"   Max Absolute Error:    {metrics['max_abs_error']:.6f}")
        print(f"   Mean Absolute Error:   {metrics['mean_abs_error']:.6f}")
        print(f"   Median Absolute Error: {metrics['median_abs_error']:.6f}")
        print(f"   Max Relative Error:    {metrics['max_rel_error']:.6f}")
        print(f"   Mean Relative Error:   {metrics['mean_rel_error']:.6f}")
        print(f"   Cosine Similarity:     {metrics['cosine_similarity']:.6f}")
        print(f"   FP16 Mean/Std:         {metrics['fp16_mean']:.6f} / {metrics['fp16_std']:.6f}")
        print(f"   INT8 Mean/Std:         {metrics['int8_mean']:.6f} / {metrics['int8_std']:.6f}")
        print()
    
    # Report layers exceeding threshold
    if high_error_layers:
        print("=" * 80)
        print(f"LAYERS EXCEEDING THRESHOLD ({threshold})")
        print("=" * 80)
        print()
        print("Consider setting these layers to FP32 precision:")
        print()
        for layer_name in high_error_layers[:20]:  # Limit to 20 for readability
            metrics = layer_errors[layer_name]
            print(f"  - {layer_name:50s} (max_abs_error: {metrics['max_abs_error']:.6f})")
        
        if len(high_error_layers) > 20:
            print(f"  ... and {len(high_error_layers) - 20} more layers")
        print()
        
        # Generate sample postprocessing script
        print("=" * 80)
        print("SAMPLE POSTPROCESSING SCRIPT")
        print("=" * 80)
        print()
        print("Save the following as 'fix_precision.py':")
        print()
        print("```python")
        print("import tensorrt as trt")
        print()
        print("def postprocess(network):")
        print("    \"\"\"Set problematic layers to FP32 precision.\"\"\"")
        print("    fp32_layers = [")
        for layer_name in high_error_layers[:10]:  # Top 10 layers
            print(f"        \"{layer_name}\",")
        print("    ]")
        print()
        print("    for layer in network:")
        print("        if layer.name in fp32_layers:")
        print("            print(f\"Setting {{layer.name}} to FP32\")")
        print("            layer.precision = trt.float32")
        print("            for i in range(layer.num_outputs):")
        print("                layer.set_output_type(i, trt.float32)")
        print("```")
        print()
        print("Then rebuild the engine with:")
        print()
        print("polygraphy convert model.onnx \\")
        print("    --int8 \\")
        print("    --calibration-cache calibration.cache \\")
        print("    --trt-network-postprocess-script fix_precision.py \\")
        print("    --precision-constraints obey \\")
        print("    -o model_int8_fixed.engine")
        print()
    else:
        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print()
        print(f"No layers exceed the threshold of {threshold}.")
        print("The INT8 model appears to have acceptable accuracy.")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise errors between INT8 and FP16 TensorRT engines"
    )
    parser.add_argument(
        "--fp16-outputs",
        required=True,
        help="Path to FP16 outputs JSON file (from polygraphy run with --save-outputs)"
    )
    parser.add_argument(
        "--int8-outputs",
        required=True,
        help="Path to INT8 outputs JSON file (from polygraphy run with --save-outputs)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for flagging high errors (default: 0.1)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top error layers to report (default: 10)"
    )
    
    args = parser.parse_args()
    
    analyze_layer_errors(
        args.fp16_outputs,
        args.int8_outputs,
        args.threshold,
        args.top_k
    )


if __name__ == "__main__":
    main()
