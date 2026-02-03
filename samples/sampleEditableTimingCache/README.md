# Create a deterministic build using editable timing cache
**Table of Contents**

- [Create a deterministic build using editable timing cache](#create-a-deterministic-build-using-editable-timing-cache)
  - [Description](#description)
  - [Running the sample](#running-the-sample)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `sampleEditableTimingCache`, illustrates how to build an engine with the desired tactics by modifying the timing cache.

In TensorRT some layers may have multiple implementations, which are called tactics. When building an engine, all of the tactics will be profiled and the fastest one will be chosen and will be written into the TimingCache. In some circumastances, the expected tactic is not the fastest one, and the user needs to replace the best tactic with another tactic. This requirement can be satisfied by editing the timing cache. This sample demonstrates how to achieve this using the Timing Cache editing API and the profiling log.

In this sample, we construct a simple network with 3 nodes: MatMul->Softmax->MatMul. The two MatMuls are identical in all properties except for their names.

First, we construct the network and build an engine from it. The `BuilderConfig` was configured to enable the editable timing cache, so TensorRT outputs the profiling information in logs. Also, it records the decisions on which tactics to use in the Timing Cache.

Then we choose a different tactic from the previously used for the first MatMul and add it to the cache.

Finally, we build the engine again. At this time, the cache is reused, so TensorRT doesn't do profiling. Rather, it uses the tactics recorded in the cache. This way, apart from the tactics used by the first MatMul, all the others are the same as before.

## Running the sample

1. The sample gets compiled when building the TensorRT OSS following the [instructions](https://github.com/NVIDIA/TensorRT). The binary named `sample_editable_timing_cache` will be created in the output directory.

2. Run the sample and observe the logs.

    ```
    ./sample_editable_timing_cache
    ```

3.  Verify that the sample has run successfully.

    This sample will ouput a lot of logs. You should see something similar to the following:

    ```
    Autotuning op matMul1(key: 0x1814870c44ff0f8574df6e3dda04cbd7):
    Sorted table of all evaluated tactics:
    tactic_id, cost(in ms), cost/fastest_cost, prediction_correlation, kernel_name, tactic_hash, tunable_parameter
       3, 0.0112640, 1.00000, 0.50673, sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8, 0x665ded9abbf88,
       5, 0.0118784, 1.05455, 0.51157, sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage4_warpsize2x1x2_tensor16x8x8, 0x393e4ef8ad243,
       6, 0.0123904, 1.10000, 0.50600, sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage5_warpsize2x2x1_tensor16x8x8, 0x2ad3a182fb05c,
    ...
    The selected tactic is (tactic hash, cost(in ms)):0x665ded9abbf88,  0.011264
    Writing the best tactic (0x665ded9abbf88) to cache
    ```
    It reports the name of the profiled operator, the key, the available tactics and the finally used one.

    Also, yous should see something like this:
    ```
    Name: matMul1_myl0_0, LayerType: gemm, Inputs: [ { Name: input, Dimensions: [128,128], Format/Datatype: Float }, { Name: weight1, Dimensions: [128,128], Format/Datatype: Float }, { Name: __mye34matMul1_alpha, Dimensions: [1], Format/Datatype: Float }, { Name: __mye35matMul1_beta, Dimensions: [1], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_4, Dimensions: [128,128], Format/Datatype: Float }], TacticName: sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8, StreamId: 0, Metadata:
    ```
    It reports the information about layer `matMul1_myl0_0` in the engine.

    The above logs output by TensorRT aren't very intuitive. For better understanding, a concise version is placed at the very end.
    ```
    Layers of the first engine:
    #0: matMul1_myl0_0                 =uses=> sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8
    #1: __myl_TraMaxSubExpSum_myl0_1   =uses=> __myl_TraMaxSubExpSum_0xcbcb71f14cb4526fd18f61134658c571
    #2: __myl_DivMul_myl0_2            =uses=> __myl_DivMul_0x80125aec9f1e9979e47ef2b407811651
    #3: matMul2_myl0_3                 =uses=> sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8

    Profiling table:
        op: matMul1
            key: 0x1814870c44ff0f8574df6e3dda04cbd7
            selected: 0x665ded9abbf88
            available tactics:
                0x665ded9abbf88 sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8
                0x393e4ef8ad243 sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage4_warpsize2x1x2_tensor16x8x8
                0x2ad3a182fb05c sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage5_warpsize2x2x1_tensor16x8x8
    ...

        op: matMul2
            key: 0xb222b0832016f1115ff61116c094875a
            selected: 0x665ded9abbf88
            available tactics:
                0x665ded9abbf88 sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8
                0x2ad3a182fb05c sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage5_warpsize2x2x1_tensor16x8x8
                0x393e4ef8ad243 sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage4_warpsize2x1x2_tensor16x8x8
    ...

    Originally, layer `matMul1_myl0_0` used kernel `sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8`.
    Now, it should use the new kernel `sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage4_warpsize2x1x2_tensor16x8x8.`

    Layers of the second engine:
    #0: matMul1_myl0_0                 =uses=> sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x32x64_stage4_warpsize2x1x2_tensor16x8x8
    #1: __myl_TraMaxSubExpSum_myl0_1   =uses=> __myl_TraMaxSubExpSum_0xcbcb71f14cb4526fd18f61134658c571
    #2: __myl_DivMul_myl0_2            =uses=> __myl_DivMul_0x80125aec9f1e9979e47ef2b407811651
    #3: matMul2_myl0_3                 =uses=> sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize32x32x64_stage3_warpsize2x1x2_tensor16x8x8
    ```

    If the sample runs successfully, you should see the following text:
    ```
    &&&& PASSED TensorRT.sample_editable_timing_cache [TensorRT v100800] [b18] # sample_editable_timing_cache
    ```

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

October 2025
  - Migrate to strongly typed APIs.

# Known issues
