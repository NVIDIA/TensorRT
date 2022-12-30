# Fused multi-head attention with relative position bias kernel compilation

This is used for swinQKVToContextPlugin in SWIN when run in fp16/int8 precision. 
Three conditions for running the plugin:
1. GPU device is of sm 75/80/86
2. headSize = 32 
3. seqlen <= 256.

## Steps for generating fused_mha_with_position_bias kernels (We can generate the kernels in computelab):

1. Clone xmma source code
   ```
   git clone https://gitlab-master.nvidia.com/jackch/fmha_v2/ -b relative_pos_bias_dev relative_pos_bias_dev
   ```
   
2. Generate the cu files (into the folder generate)
   ```
   python3 setup.py
   ```
   
3. Generate cubins
   ```
   make cubin_demoswin -j
   ```
   
4. Generate sMhaKernelMetaInfosV2 for fmha_with_position_bias_v2.h 
   ```
   ./bin/print_traits.exe | python format_traits.py
   ```

5. Copy cubins to TRT
   Run command like this for all the new kernels:
      ```
      cp cubin/fused_mha_with_relPosBias_fp16_128_32_kernel.sm75.cpp plugin/swinQKVToContextPlugin/fmha_with_position_bias/
      ```

        