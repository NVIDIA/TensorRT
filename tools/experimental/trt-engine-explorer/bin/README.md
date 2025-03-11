# trex Command-line Tool

The `trex` command-line tool (not to be confused with the `trex` package) provides a convinient interface to some of the utilities in the `utils` directory. It is installed with the `trex` package.

## trex draw
Draw a graph diagram of a TensorRT engine graph JSON file.<br>

Example:
```
$ trex draw ./examples/pytorch/resnet/A100/fp32/resnet.onnx.engine.graph.json --display_regions --no_layer_names
```

There are many rendering configuration flags including `--query` which draws the induced subgraph of all nodes whose name or metadata matches the regular expression, which is useful for engines graphs that are too large for graphviz to render.
```
$ trex draw mymodel.graph.json -pj=mymodel.profile.json --display_metadata --display_regions --display_constant --query="transformer_blocks.(12|37).*"
```

## trex process
Build, profile and draw a TensorRT engine.
```
$ trex process ./examples/pytorch/resnet/generated/resnet.onnx ./examples/pytorch/resnet/A100/fp32/
```

## trex summary
Print a tabularized summary of the tactics
```
$ trex summary ./examples/pytorch/resnet/A100/fp32/resnet.onnx.engine.graph.json --profiling_json=./examples/pytorch/resnet/A100/fp32/resnet.onnx.engine.profile.json --sort_key=latency --group_tactics

+----+-----------------------------------------------------------------------------------------------------------------------------------+-------------+---------+
|    | tactic                                                                                                                            |   latency % |   count |
|----+-----------------------------------------------------------------------------------------------------------------------------------+-------------+---------|
|  1 | ampere_scudnn_128x64_relu_xregs_large_nn_v                                                                                        |   24.5902   |       1 |
|  7 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x2x1_g1_tensor16x8x8_t1r3s       |   21.7454   |       4 |
|  9 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_t1r3s       |   15.188    |       4 |
|  5 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize256x128x32_stage3_warpsize4x2x1_g1_tensor16x8x8_t1r3s      |   13.838    |       4 |
|  4 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage4_warpsize2x2x1_g1_tensor16x8x             |   13.7898   |       4 |
|  0 | TensorR                                                                                                                           |    6.50916  |       2 |
|  3 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_t1r1s      |    1.44648  |       1 |
|  6 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x2x1_g1_tensor16x8x8_t1r1s       |    0.96432  |       1 |
|  8 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_t1r1s       |    0.819672 |       1 |
| 10 | sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_simple_t1r1s |    0.626808 |       1 |
|  2 | sm50_xmma_pooling_fw_4d_FP32FP32NHWC_Average_FastDiv_CAlign                                                                       |    0.48216  |       1 |
+----+-----------------------------------------------------------------------------------------------------------------------------------+-------------+---------+

```
