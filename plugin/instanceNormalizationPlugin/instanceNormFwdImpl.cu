/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <stdio.h>
 #include <assert.h>
 #include <type_traits>
 
 #include "instanceNormFwd.h"
 #include "instanceNormCommon.h"
 
 namespace instance_norm_impl
 {

 static inline int div_up(int m, int n) {
     return (m + n - 1) / n;
 }
 
 
 using kernel_params_32 = Instance_norm_kernel_params<uint16_t, uint16_t, uint16_t, 512, 8, 32>;
 using kernel_params_64 = Instance_norm_kernel_params<uint16_t, uint16_t, uint16_t, 512, 16, 64>;
 
 using kernel_params_32_int8 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32>;
 using kernel_params_32_int8_sm_700 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32, 700>;
 using kernel_params_32_int8_sm_720 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32, 720>;
 using kernel_params_32_int8_sm_750 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32, 750>;
 using kernel_params_32_int8_sm_800 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32, 800>;
 using kernel_params_32_int8_sm_860 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32, 860>;
 // debug :
 //using kernel_params_32_int8 = Instance_norm_kernel_params<int8_t, int8_t, 256, 8, 32>;
 using kernel_params_32_fp16_int8 = Instance_norm_kernel_params<uint16_t, int8_t, float, 512, 8, 32>;
 //using kernel_params_32_int8 = Instance_norm_kernel_params<int8_t, int8_t, 512, 4, 16>;
 
 template< 
     typename Storage,
     typename Input_Data_Type,
     typename Output_Data_Type,
     int THREADS_PER_CTA, 
     int THREADS_PER_PIXEL, 
     int PIXELS_PER_THREAD_IN_REGISTERS, 
     int PIXELS_PER_THREAD_IN_SMEM,
     int ELEMENTS_PER_LDG,
     int USE_ONLINE_APPROACH,
     int OUTER_LOOPS_,
     int DESIRED_OCCUPANCY
 >
 __global__ __launch_bounds__(THREADS_PER_CTA, DESIRED_OCCUPANCY)
     void instance_norm_fwd(InstanceNormFwdParams params) {
 
     // Single pass numerically stable algorithm, see:
     // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
     //
     // n = 0, mean = 0.0, M2 = 0.0
     // 
     // for x in data:
     //     n += 1
     //     delta = x - mean
     //     mean += delta/n
     //     delta2 = x - mean
     //     M2 += delta*delta2
     // 
     // if n < 2:
     //     return float('nan')
     // else:
     //     return M2 / (n - 1)
 
     const bool IS_INPUT_INT8 = std::is_same<Input_Data_Type, int8_t>::value;
     const bool IS_OUTPUT_INT8 = std::is_same<Output_Data_Type, int8_t>::value;
 
     // The number of pixels loaded in a single LDG.
     const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
     // The number of pixels computed per CTA stored in registers.
     const int PIXELS_PER_CTA_IN_REGISTERS = PIXELS_PER_THREAD_IN_REGISTERS * PIXELS_PER_LDG;
     // The number of pixels computed per CTA stored in SMEM.
     const int PIXELS_PER_CTA_IN_SMEM = PIXELS_PER_THREAD_IN_SMEM*PIXELS_PER_LDG;
     // The number of C elements per CTA.
     const int C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL*ELEMENTS_PER_LDG;
 
     // Shared memory to do CTA-wide parallel sums.
     __shared__ float smem[ELEMENTS_PER_LDG*THREADS_PER_CTA];
 
     // The position in the NHW dimension where the CTA starts.
     int cta_nhw_regs = blockIdx.x * PIXELS_PER_CTA_IN_REGISTERS;
     // The position in the NHW dimension where the CTA starts for the portion in SMEM.
     int cta_nhw_smem = blockIdx.x * PIXELS_PER_CTA_IN_SMEM;
     // Compute the NHW coordinate of the thread in the CTA.
     const int thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;
 
     for (int nc_blk_index = blockIdx.y; nc_blk_index < params.c_blks * params.n; nc_blk_index += gridDim.y) {
     
     int n_blk_index = nc_blk_index / params.c_blks;
     int c_blk_index = nc_blk_index % params.c_blks;
 
     // The position in the C dimension where the CTA starts.
     const int cta_c = c_blk_index * C_ELEMENTS_PER_CTA;
     // Compute the C coordinate of the thread in the CTA. 
     const int thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
     // Compute the C coordinate of the thread.
     const int thread_c = cta_c + thread_in_cta_c*ELEMENTS_PER_LDG;
 
     // Is the thread working on a valid C dimension?
     const int is_valid_c = thread_c < params.c;
 
     // The adapter for the storage.
     typedef PackedStorage<Storage, ELEMENTS_PER_LDG> PackedStorage_;
     // The data type for packed storage in SMEM.
     typedef typename PackedStorage_::Type PackedStorageType;
     // The number of elements in the packed storage.
     const int PACKED_ELEMENTS_PER_LDG = PackedStorage_::PACKED_ELEMENTS_PER_LDG;
     // Registers to keep the data live for the persistent approach.
     PackedStorageType x_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];
 
     // Shared memory buffer to store the extra pixels.
     extern __shared__ char smem_storage_[];
     PackedStorageType * smem_storage = reinterpret_cast<PackedStorageType *>(smem_storage_);
 
     float int8_in_scale = params.in_scale;
     float int8_out_scale = params.out_scale;
 
     // Register to store the number of elements read so far.
     float count = 0.f, mean[ELEMENTS_PER_LDG], m2[ELEMENTS_PER_LDG];
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         mean[i] = 0.f;
         m2  [i] = 0.f;
     }
 
     // The number of elements loaded by this CTA.
     int cta_count = 0;
     int global_batch_offset = n_blk_index * params.nhw * params.c;
     // int8 relevant
     // int8 output implies we have NC/32DHW32 input for bath fp16 and int8
     int global_thread_c_input = ( IS_INPUT_INT8 || IS_OUTPUT_INT8 )? thread_in_cta_c*ELEMENTS_PER_LDG 
                                       + (cta_c % 32)  // handle C_ELEMENTS_PER_CTA == 16 case
                                       + (cta_c / 32) * 32 * params.nhw : thread_c;
     int stride_c_input = ( IS_INPUT_INT8 || IS_OUTPUT_INT8 )? 32 : params.c;
     int global_thread_c_output = ( IS_OUTPUT_INT8 )? thread_in_cta_c*ELEMENTS_PER_LDG 
                                       + (cta_c % 32)  // handle C_ELEMENTS_PER_CTA == 16 case
                                       + (cta_c / 32) * 32 * params.nhw : thread_c;
     int stride_c_output = ( IS_OUTPUT_INT8 )? 32 : params.c;
     // The base pointer to load from.
     const Input_Data_Type *gmem_src = &reinterpret_cast<Input_Data_Type *>(params.gmem_src)[global_thread_c_input + global_batch_offset];
 
     // Load the batch of elements. Compute the mean/var across those elements.
     const int pixels_per_iteration = PIXELS_PER_CTA_IN_REGISTERS*gridDim.x;
 
     // outer loops
     int OUTER_LOOPS = OUTER_LOOPS_ == 1? 1 : params.outer_loops;
 
     #pragma unroll 1
     for( int loop_i = 0; loop_i < OUTER_LOOPS; ++loop_i ) {
         // The nhw position.
         int nhw_regs = cta_nhw_regs + loop_i*pixels_per_iteration;
 
         cta_count += max(min(nhw_regs + PIXELS_PER_CTA_IN_REGISTERS, params.nhw) - max(nhw_regs, 0), 0);
 
         // Load the data and compute the local mean/sum and the variance.
         if( USE_ONLINE_APPROACH ) {
             // Read the elements from memory.
             float is_valid[PIXELS_PER_THREAD_IN_REGISTERS];
             #pragma unroll
             for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
                 const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                 zero(x_storage[i]);
                 is_valid[i] = 0.f;
                 if( idx < params.nhw && is_valid_c ) {
                     ldg_stream(x_storage[i], &gmem_src[idx*stride_c_input]);
                     is_valid[i] = 1.f;
                 }
             }
 
             // Do the math.
             #pragma unroll
             for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
                 // Convert to float.
                 float x_math[ELEMENTS_PER_LDG];
                 to_float<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);
 
                 // Update the count.
                 count += is_valid[i];
                 // Invert the count.
                 float inv_count = is_valid[i] ? 1.f / count : 0.f;
 
                 // Update the mean and m2 using deltas.
                 #pragma unroll
                 for( int j = 0; j < ELEMENTS_PER_LDG; ++j ) {
                     float delta0 = x_math[j] - mean[j];
                     mean[j] += delta0 * inv_count;
                     float delta1 = x_math[j] - mean[j];
                     m2[j] += delta0 * delta1 * is_valid[i];
                 }
             }
         } else {
             // Read the elements from memory.
             #pragma unroll
             for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
                 const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                 zero(x_storage[i]);
                 if( idx < params.nhw && is_valid_c ) {
                     ldg_stream(x_storage[i], &gmem_src[idx * stride_c_input]);
                     count += 1.f;
                 }
             }
 
             // Sum the elements in registers.
             #pragma unroll
             for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
                 // Convert to float.
                 float x_math[ELEMENTS_PER_LDG];
                 to_float<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);
 
                 // Update the mean and m2 using deltas.
                 #pragma unroll
                 for( int j = 0; j < ELEMENTS_PER_LDG; ++j ) {
                     mean[j] += x_math[j];
                 }
             }
             
             // Compute the mean.
             float inv_count = 1.f / count;
             #pragma unroll
             for( int j = 0; j < ELEMENTS_PER_LDG; ++j ) {
                 mean[j] *= inv_count;
             }
             
             // Compute the variance.
             #pragma unroll
             for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
                 // Convert to float.
                 float x_math[ELEMENTS_PER_LDG];
                 to_float<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);
 
                 // Is it a valid pixel?
                 float is_valid = i < (int) count ? 1.f : 0.f;
                 // Update the mean and m2 using deltas.
                 #pragma unroll
                 for( int j = 0; j < ELEMENTS_PER_LDG; ++j ) {
                     m2[j] += (x_math[j] - mean[j]) * (x_math[j] - mean[j]) * is_valid; 
                 }
             }
         }
     }
 
     // The elements to load and store in SMEM.
     int smem_nhw = OUTER_LOOPS*pixels_per_iteration + cta_nhw_smem; 
     // Load elements from SMEM, update the CTA count.
     int pixels_in_smem = min(smem_nhw + PIXELS_PER_CTA_IN_SMEM, params.nhw) - max(smem_nhw, 0);
     if( pixels_in_smem > 0 ) {
         cta_count += pixels_in_smem;
         for( int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i ) {
             const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
             float is_pixel_valid = (idx < params.nhw && is_valid_c) ? 1.f : 0.f;
 
             PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG];
             ldg_stream(x_storage_local, &gmem_src[(is_pixel_valid ? idx : 0) * stride_c_input]);
 
             // The offset to store in SMEM.
             const int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
             // Store in SMEM.
             write_to_smem(&smem_storage[offset], threadIdx.x, x_storage_local);
             // Update the count.
             count += is_pixel_valid;
             // Invert the count.
             float inv_count = is_pixel_valid ? 1.f / count : 0.f;
 
             float x_math[ELEMENTS_PER_LDG];
             to_float<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage_local, int8_in_scale);
             // Update the mean and m2 using deltas.
             #pragma unroll
             for( int j = 0; j < ELEMENTS_PER_LDG; ++j ) {
                 float delta0 = x_math[j] - mean[j];
                 mean[j] += delta0 * inv_count;
                 float delta1 = x_math[j] - mean[j];
                 m2[j] += delta0 * delta1 * is_pixel_valid;
             }
         }
     }
 
     // We scale the mean by the number of elements. It brings more stability.
     float m1[ELEMENTS_PER_LDG];
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         m1[i] = mean[i] * count;
     }
 
     // Run the parallel sum accross the CTA to get the local sum.
     ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
         smem, m1, thread_in_cta_nhw); 
     __syncthreads();
 
     // The values in shared memory correspond to the CTA-wide sums.
     read_from_smem(m1, smem, thread_in_cta_c); 
     __syncthreads();
 
     // Adjust the variance.
     float inv_cta_count = 1.f / (float) cta_count;
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         float mean_diff = m1[i]*inv_cta_count - mean[i];
         m2[i] = m2[i] + mean_diff * mean_diff * count;
     }
         
     // Run the parallel sum accross the CTA to get the local adjusted variance.
     ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
         smem, m2, thread_in_cta_nhw); 
 
     // The workspace in global memory is distributed across the different CTA. 
     int gmem_sums_offset = nc_blk_index*gridDim.x*C_ELEMENTS_PER_CTA*2;
 
     // Write the data for the CTA to global memory.
     GMEM_SUMS_TYPE *gmem_sums = &params.gmem_sums[gmem_sums_offset];
     if( threadIdx.x < THREADS_PER_PIXEL ) {
         const int idx = blockIdx.x*THREADS_PER_PIXEL + threadIdx.x;
         write_to_gmem(&gmem_sums[                           0], idx, m1);
         write_to_gmem(&gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx, m2);
     }
 
     // The memory location to store the number of pixels per CTA.
     int *gmem_counts = &params.gmem_counts[nc_blk_index*gridDim.x];
     if( threadIdx.x == 0 ) {
         //gmem_counts[0] = cta_count;
         gmem_counts[blockIdx.x] = cta_count;
     }
 
     // Read the bias and scale.
     float bias[ELEMENTS_PER_LDG];
     float scale[ELEMENTS_PER_LDG];
     read_from_gmem(bias, &params.gmem_bias[cta_c], thread_in_cta_c);
     read_from_gmem(scale, &params.gmem_scale[cta_c], thread_in_cta_c);
 
     // The counters to count how many CTAs have retired at this point. One per chunk of C.
     int *gmem_retired_ctas = &params.gmem_retired_ctas[nc_blk_index];
 
     // Make sure the threads are done and reconverged.
     __syncthreads();
 
     // Register the CTA.
     int expected_count = gridDim.x;
     if( threadIdx.x == 0 ) {
         // Issue the membar.
         __threadfence();
         // Notify that the CTA is done.
         int val_to_add = 1;
         if (blockIdx.x == 0) {
             val_to_add = -(expected_count - 1);
         }
         atomicAdd(gmem_retired_ctas, val_to_add);
     }
 
     // Are all CTAs done?
     if (threadIdx.x == 0) {
         int retired_ctas = -1;
         do {
             __threadfence();
             asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(retired_ctas) : "l"(gmem_retired_ctas));
         } while (retired_ctas != 0);
     }
     __threadfence();
     __syncthreads();
 
     // Reset the mean to compute the global mean.
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         m1[i] = 0.f;
     }
 
     // Build the global mean.
     #pragma unroll 1
     for( int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA ) {
         float tmp[ELEMENTS_PER_LDG];
         read_from_gmem(tmp, gmem_sums, idx);
 
         #pragma unroll
         for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
             m1[i] += tmp[i];
         }
     }
 
     // Run the parallel sum accross the CTA to get the local sum.
     ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
         smem, m1, thread_in_cta_nhw); 
     __syncthreads();
 
     // The values in shared memory correspond to the CTA-wide sums.
     read_from_smem(m1, smem, thread_in_cta_c); 
     __syncthreads();
 
     // Normalize the mean.
     float inv_count = 1.f / (float) params.nhw;
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         m1[i] = m1[i] * inv_count;
     }
 
     // Reset the variance.
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         m2[i] = 0.f;
     }
 
     // Build the global variance.
     #pragma unroll 1
     for( int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA ) {
 
         // Read the means computed by different CTAs (again). Reuse tmp if we have 1 iteration.
         float tmp_mean[ELEMENTS_PER_LDG], tmp_var[ELEMENTS_PER_LDG];
         read_from_gmem(tmp_mean, &gmem_sums[                           0], idx);
         read_from_gmem(tmp_var,  &gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx);
 
         // Read the number of pixels visited by a given CTA.
         cta_count = __ldg(&gmem_counts[idx / THREADS_PER_PIXEL]); 
 
         // Compute the diff to update the variance.
         float mean_diff[ELEMENTS_PER_LDG], inv_cta_count = 1.f / (float) cta_count;
         #pragma unroll 
         for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
             mean_diff[i] = m1[i] - tmp_mean[i]*inv_cta_count;
         }
 
         // Update the variance.
         #pragma unroll 
         for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
             m2[i] += tmp_var[i] + mean_diff[i]*mean_diff[i]*(float) cta_count;
         }
     }
 
     // Run the parallel sum accross the CTA to get the local sum.
     ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
         smem, m2, thread_in_cta_nhw); 
     __syncthreads();
 
     read_from_smem(m2, smem, thread_in_cta_c); 
     __syncthreads();
 
     // Finalize the stddev.
     #pragma unroll 
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         m2[i] *= inv_count;
     }
 
     // store the saved mean/var
     float svarinv[ELEMENTS_PER_LDG];
     bool is_valid_for_saving = is_valid_c && blockIdx.x == 0 && thread_in_cta_nhw == 0;
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         svarinv[i] = rsqrtf(m2[i] + params.var_eps);
     }
 
 #if !DISABLE_MEAN_VAR_OUTPUT
     int global_stats_offset =  n_blk_index * params.c;
     if( is_valid_for_saving ) {
         write_to_gmem(params.gmem_saved_mean + global_stats_offset, \
                       thread_c/ELEMENTS_PER_LDG, m1);
         write_to_gmem(params.gmem_saved_var + global_stats_offset, \
                       thread_c/ELEMENTS_PER_LDG, svarinv);
     }
 
     // store the running mean/var
     float rmean[ELEMENTS_PER_LDG];
     float rvar[ELEMENTS_PER_LDG];
     zero(rmean);
     zero(rvar);
     if( params.exp_avg_factor != 1.f && is_valid_for_saving ) {
         read_from_gmem(rmean, params.gmem_running_mean + global_stats_offset, \
                        thread_c/ELEMENTS_PER_LDG);
         read_from_gmem(rvar, params.gmem_running_var + global_stats_offset, \
                        thread_c/ELEMENTS_PER_LDG);
     }
     #pragma unroll
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         rmean[i] = (1.f - params.exp_avg_factor) * rmean[i] +   \
             params.exp_avg_factor * m1[i];
         rvar[i] = (1.f - params.exp_avg_factor) * rvar[i] +     \
             params.exp_avg_factor * m2[i];
     }
     if( is_valid_for_saving ) {
         write_to_gmem(params.gmem_running_mean + global_stats_offset, thread_c/ELEMENTS_PER_LDG, rmean);
         write_to_gmem(params.gmem_running_var + global_stats_offset, thread_c/ELEMENTS_PER_LDG, rvar);
     }
 #endif 
 
     // Update the scale with the stddev and eps.
     #pragma unroll 
     for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
         scale[i] *= svarinv[i];
     }
 
     // The base pointer to write to.
     Output_Data_Type *const gmem_dst = &reinterpret_cast<Output_Data_Type *>(params.gmem_dst)[global_thread_c_output + global_batch_offset];
 
     // Store the elements in registers.
     #pragma unroll 1
     for( int loop_i = OUTER_LOOPS-1; loop_i >= 0; --loop_i ) {
 
         // The value for nhw.
         int out_nhw = cta_nhw_regs + loop_i*pixels_per_iteration;
 
         // Normalize the elements and write to memory.
         #pragma unroll 
         for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
             // Convert to float.
             float x_math[ELEMENTS_PER_LDG];
             to_float<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);
 
             // Normalize and apply activation function
             normalize(x_math, bias, scale, m1);
             if( params.use_relu ) {
                 relu_activation(x_math, params.relu_alpha);
             }
 
             // Write back.
             const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
             if( (unsigned) idx < params.nhw && is_valid_c ) {
                 stg_stream(&gmem_dst[idx*stride_c_output], x_math, int8_out_scale);
             }
         }
 
         // The next value of nhw.
         out_nhw -= pixels_per_iteration;
 
         // Read the next elements from memory.
         #pragma unroll 
         for( int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i ) {
             const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
             if( (unsigned) idx < params.nhw && is_valid_c ) {
                 ldg_stream(x_storage[i], &gmem_src[idx*stride_c_output]);
             }
         }
     }
 
     // Normalize the elements from SMEM and write them out.
     if( pixels_in_smem > 0 ) {
         for( int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i ) {
             // Read from SMEM.
             const int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
             float x_math[ELEMENTS_PER_LDG];
             PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG];
             read_from_smem(x_storage_local, &smem_storage[offset], threadIdx.x);
             to_float<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage_local, int8_in_scale);
 
             // Normalize and apply activation function
             normalize(x_math, bias, scale, m1);
             if( params.use_relu ) {
                 relu_activation(x_math, params.relu_alpha);
             }
 
             // Write back.
             const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
             if( (unsigned) idx < params.nhw && is_valid_c ) {
                 stg_stream(&gmem_dst[idx*stride_c_output], x_math, int8_out_scale);
             }
         }
     }
     __syncthreads();
 } // blockIdx.y loop
 
 }
 
 
 template <typename Kernel_params>
 dim3 estimate_in_grid_dim(const InstanceNormFwdParams& params)
 {
     dim3 grid_dim;
     grid_dim.x = div_up(params.nhw, Kernel_params::MIN_PIXELS_PER_CTA); // PIXELS_PER_CTA
     grid_dim.y = div_up(params.c * params.n, Kernel_params::C_ELEMENTS_PER_CTA);
     grid_dim.z = 1; //params.n;
 
     return grid_dim;
 }
 
 
 template <typename Kernel_params>
 void instance_norm_buffer_sizes(const InstanceNormFwdParams& params, 
                                 size_t &size_sums, size_t &size_counts, size_t &size_retired_ctas)
 {
     dim3 grid_dim = estimate_in_grid_dim<Kernel_params>(params);
 
     size_sums = grid_dim.z*grid_dim.y*grid_dim.x*Kernel_params::THREADS_PER_PIXEL*Kernel_params::ELEMENTS_PER_LDG*2*sizeof(GMEM_SUMS_TYPE);
     size_counts = grid_dim.z*grid_dim.y*grid_dim.x*sizeof(int);
     size_retired_ctas = grid_dim.z*grid_dim.y*sizeof(int);
 
     size_sums = div_up(size_sums, 256) * 256;
     size_counts = div_up(size_counts, 256) * 256;
     size_retired_ctas = div_up(size_retired_ctas, 256) * 256;
 }
 
 
 
 
 
 template <typename Kernel_params>
 int instance_norm_fwd_launch(const InstanceNormFwdContext& context, InstanceNormFwdParams& params, cudaStream_t stream)
 {
 
     size_t smem_size = Kernel_params::PIXELS_PER_THREAD_IN_SMEM * 
                         Kernel_params::THREADS_PER_CTA * 
                         Kernel_params::ELEMENTS_PER_LDG * sizeof(typename Kernel_params::StorageType);
 
     dim3 grid_dim = estimate_in_grid_dim<Kernel_params>(params);
 
     params.c_blks = div_up(params.c, Kernel_params::C_ELEMENTS_PER_CTA);
 
     size_t size_retired_ctas = grid_dim.z*grid_dim.y*sizeof(int);
 
     #define KERNEL_RUN(OUTER_LOOPS, DESIRED_OCCUPANCY)  \
         {                            \
             CHECK_CUDA(cudaMemsetAsync(params.gmem_retired_ctas, 0, size_retired_ctas, stream)); \
             if( smem_size > 0 ) \
                 CHECK_CUDA(cudaFuncSetAttribute( \
                     instance_norm_fwd< \
                             typename Kernel_params::StorageType, \
                             typename Kernel_params::Input_Data_Type, \
                             typename Kernel_params::Output_Data_Type, \
                             Kernel_params::THREADS_PER_CTA, \
                             Kernel_params::THREADS_PER_PIXEL, \
                             Kernel_params::PIXELS_PER_THREAD_IN_REGISTERS, \
                             Kernel_params::PIXELS_PER_THREAD_IN_SMEM, \
                             Kernel_params::ELEMENTS_PER_LDG, \
                             Kernel_params::USE_ONLINE_APPROACH, \
                             OUTER_LOOPS, \
                             DESIRED_OCCUPANCY>, \
                     cudaFuncAttributeMaxDynamicSharedMemorySize, \
                     smem_size));                                 \
             instance_norm_fwd< \
                 typename Kernel_params::StorageType, \
                 typename Kernel_params::Input_Data_Type, \
                 typename Kernel_params::Output_Data_Type, \
                 Kernel_params::THREADS_PER_CTA, \
                 Kernel_params::THREADS_PER_PIXEL, \
                 Kernel_params::PIXELS_PER_THREAD_IN_REGISTERS, \
                 Kernel_params::PIXELS_PER_THREAD_IN_SMEM, \
                 Kernel_params::ELEMENTS_PER_LDG, \
                 Kernel_params::USE_ONLINE_APPROACH, \
                 OUTER_LOOPS, \
                 DESIRED_OCCUPANCY><<<grid_dim,Kernel_params::THREADS_PER_CTA, smem_size, stream>>>(params); }
 
     size_t total_smem_bytes = smem_size + Kernel_params::ELEMENTS_PER_LDG * Kernel_params::THREADS_PER_CTA * sizeof(float);
     int smem_driven_fwd_occupancy = min(int(context.sm_shared_size) / (int)total_smem_bytes, (int)2);
     int max_grid                  = context.sm_count * smem_driven_fwd_occupancy;
     if ((context.sm_version >= 700 ) && (context.sm_version < 800))
     {
         max_grid = max_grid - 4;
     }
 
     if (max_grid / int(grid_dim.x) > 1) {
         grid_dim.y = max_grid / int(grid_dim.x);
         grid_dim.y = int(grid_dim.y) > params.c_blks * params.n ? params.c_blks * params.n : int(grid_dim.y);
     } else {
         grid_dim.y = 1;
     }
 
     int loop = 1;
     if( grid_dim.x <= max_grid ) {
         if (smem_driven_fwd_occupancy >= 2) {
             KERNEL_RUN(1, 2);
         } else {
             KERNEL_RUN(1, 1);
         }
     } else {
         grid_dim.x = max_grid;
         int nhw_in_regs = params.nhw - Kernel_params::PIXELS_PER_THREAD_IN_SMEM*Kernel_params::PIXELS_PER_LDG*grid_dim.x;
         int pixels_per_iteration = Kernel_params::PIXELS_PER_THREAD_IN_REGISTERS*Kernel_params::PIXELS_PER_LDG*grid_dim.x;
         nhw_in_regs = (nhw_in_regs <= 0)? pixels_per_iteration : nhw_in_regs;
         if (nhw_in_regs < 0)
         {
             nhw_in_regs = pixels_per_iteration;
             // make PIXELS_PER_THREAD_IN_SMEM <= PIXELS_PER_THREAD_IN_REGISTERS if the assert fails
             assert(pixels_per_iteration >= params.nhw);
         }
 
         loop = div_up(nhw_in_regs, pixels_per_iteration);
         params.outer_loops = loop;
         assert(loop >= 1);
 
         if( loop == 1 ) {
             if (smem_driven_fwd_occupancy >= 2) {
                 KERNEL_RUN(1, 2);
             } else {
                 KERNEL_RUN(1, 1);
             }
         } else {
             if (smem_driven_fwd_occupancy >= 2) {
                 KERNEL_RUN(0, 2);
             } else {
                 KERNEL_RUN(0, 1);
             }
         }
     }
     return loop;
 }
 
 static int c_cond_g = 32;
 
 void instance_norm_buffer_sizes_dispatch(const InstanceNormFwdContext& context, const InstanceNormFwdParams& params, 
                                 size_t &size_sums, size_t &size_counts, size_t &size_retired_ctas,
                                 int input_data_type, int output_data_type)
 {
     if (input_data_type == 2 && output_data_type == 2) {
         switch (context.sm_version)
         {
             case 700: return instance_norm_buffer_sizes<kernel_params_32_int8_sm_700>(params, size_sums, size_counts, size_retired_ctas); break;
             case 720: return instance_norm_buffer_sizes<kernel_params_32_int8_sm_720>(params, size_sums, size_counts, size_retired_ctas); break;
             case 750: return instance_norm_buffer_sizes<kernel_params_32_int8_sm_750>(params, size_sums, size_counts, size_retired_ctas); break;
             case 800: return instance_norm_buffer_sizes<kernel_params_32_int8_sm_800>(params, size_sums, size_counts, size_retired_ctas); break;
             case 860: return instance_norm_buffer_sizes<kernel_params_32_int8_sm_860>(params, size_sums, size_counts, size_retired_ctas); break;
             default: return instance_norm_buffer_sizes<kernel_params_32_int8>(params, size_sums, size_counts, size_retired_ctas); break;
         }
         return instance_norm_buffer_sizes<kernel_params_32_int8>(params, size_sums, size_counts, size_retired_ctas);
     } else if (input_data_type == 1 && output_data_type == 2) {
         return instance_norm_buffer_sizes<kernel_params_32_fp16_int8>(params, size_sums, size_counts, size_retired_ctas);
     } else if (input_data_type == 1 && output_data_type == 1) {
         if (params.c <= c_cond_g) {
             return instance_norm_buffer_sizes<kernel_params_32>(params, size_sums, size_counts, size_retired_ctas);
         }
         else {
             return instance_norm_buffer_sizes<kernel_params_64>(params, size_sums, size_counts, size_retired_ctas);
         }
     } else {
         fprintf(stderr, "Unsupported format combination by the instance norm kernel\n");
         assert(0);
     }
 }
 
 
 int instance_norm_fwd_dispatch(const InstanceNormFwdContext& context, InstanceNormFwdParams& params, cudaStream_t stream, 
                                int input_data_type, int output_data_type)
 {
     assert(context.sm_version >= 600);
     if (input_data_type == 2 && output_data_type == 2) {
         switch (context.sm_version)
         {
             case 700: return instance_norm_fwd_launch<kernel_params_32_int8_sm_700>(context, params, stream); break;
             case 720: return instance_norm_fwd_launch<kernel_params_32_int8_sm_720>(context, params, stream); break;
             case 750: return instance_norm_fwd_launch<kernel_params_32_int8_sm_750>(context, params, stream); break;
             case 800: return instance_norm_fwd_launch<kernel_params_32_int8_sm_800>(context, params, stream); break;
             case 860: return instance_norm_fwd_launch<kernel_params_32_int8_sm_860>(context, params, stream); break;
             default: return instance_norm_fwd_launch<kernel_params_32_int8>(context, params, stream); break;
         }
     } else if (input_data_type == 1 && output_data_type == 2) {
         return instance_norm_fwd_launch<kernel_params_32_fp16_int8>(context, params, stream);
     } else if (input_data_type == 1 && output_data_type == 1) {
         if (params.c <= c_cond_g) {
             return instance_norm_fwd_launch<kernel_params_32>(context, params, stream);
         }
         else {
             return instance_norm_fwd_launch<kernel_params_64>(context, params, stream);
         }
     } else {
         fprintf(stderr, "Unsupported format combination by the instance norm kernel\n");
         assert(0);
     }
     
     return 0;
 }
 
 } // namespace instance_norm_impl