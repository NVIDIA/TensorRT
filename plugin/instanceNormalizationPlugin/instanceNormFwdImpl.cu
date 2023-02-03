/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
#include <cuda.h>

#include "instanceNormFwd.h"
#include "instanceNormCommon.h"

namespace instance_norm_impl
{

static inline int32_t divUp(int32_t m, int32_t n)
{
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
using kernel_params_32_int8_sm_870 = Instance_norm_kernel_params<int8_t, int8_t, int8_t, 512, 8, 32, 870>;
using kernel_params_32_fp16_int8 = Instance_norm_kernel_params<uint16_t, int8_t, float, 512, 8, 32>;

template <typename Storage, typename Input_Data_Type, typename Output_Data_Type, int32_t THREADS_PER_CTA,
    int32_t THREADS_PER_PIXEL, int32_t PIXELS_PER_THREAD_IN_REGISTERS, int32_t PIXELS_PER_THREAD_IN_SMEM,
    int32_t ELEMENTS_PER_LDG, int32_t USE_ONLINE_APPROACH, int32_t OUTER_LOOPS_, int32_t DESIRED_OCCUPANCY>
__global__ __launch_bounds__(THREADS_PER_CTA, DESIRED_OCCUPANCY) void instanceNormFwd(InstanceNormFwdParams params)
{

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

    bool const IS_INPUT_INT8 = std::is_same<Input_Data_Type, int8_t>::value;
    bool const IS_OUTPUT_INT8 = std::is_same<Output_Data_Type, int8_t>::value;

    // The number of pixels loaded in a single LDG.
    int32_t const PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
    // The number of pixels computed per CTA stored in registers.
    int32_t const PIXELS_PER_CTA_IN_REGISTERS = PIXELS_PER_THREAD_IN_REGISTERS * PIXELS_PER_LDG;
    // The number of pixels computed per CTA stored in SMEM.
    int32_t const PIXELS_PER_CTA_IN_SMEM = PIXELS_PER_THREAD_IN_SMEM * PIXELS_PER_LDG;
    // The number of C elements per CTA.
    int32_t const C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL * ELEMENTS_PER_LDG;

    // Shared memory to do CTA-wide parallel sums.
    __shared__ float smem[ELEMENTS_PER_LDG * THREADS_PER_CTA];

    // The position in the NHW dimension where the CTA starts.
    int32_t cta_nhw_regs = blockIdx.x * PIXELS_PER_CTA_IN_REGISTERS;
    // The position in the NHW dimension where the CTA starts for the portion in SMEM.
    int32_t cta_nhw_smem = blockIdx.x * PIXELS_PER_CTA_IN_SMEM;
    // Compute the NHW coordinate of the thread in the CTA.
    int32_t const thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;

    for (int32_t nc_blk_index = blockIdx.y; nc_blk_index < params.c_blks * params.n; nc_blk_index += gridDim.y)
    {

        int32_t n_blk_index = nc_blk_index / params.c_blks;
        int32_t c_blk_index = nc_blk_index % params.c_blks;

        // The position in the C dimension where the CTA starts.
        int32_t const cta_c = c_blk_index * C_ELEMENTS_PER_CTA;
        // Compute the C coordinate of the thread in the CTA.
        int32_t const thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
        // Compute the C coordinate of the thread.
        int32_t const thread_c = cta_c + thread_in_cta_c * ELEMENTS_PER_LDG;

        // Is the thread working on a valid C dimension?
        int32_t const is_valid_c = thread_c < params.c;

        // The adapter for the storage.
        typedef PackedStorage<Storage, ELEMENTS_PER_LDG> PackedStorage_;
        // The data type for packed storage in SMEM.
        typedef typename PackedStorage_::Type PackedStorageType;
        // The number of elements in the packed storage.
        int32_t const PACKED_ELEMENTS_PER_LDG = PackedStorage_::PACKED_ELEMENTS_PER_LDG;
        // Registers to keep the data live for the persistent approach.
        PackedStorageType x_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];

        // Shared memory buffer to store the extra pixels.
        extern __shared__ char smem_storage_[];
        PackedStorageType* smem_storage = reinterpret_cast<PackedStorageType*>(smem_storage_);

        float int8_in_scale = params.in_scale;
        float int8_out_scale = params.out_scale;

        // Register to store the number of elements read so far.
        float count = 0.f, mean[ELEMENTS_PER_LDG], m2[ELEMENTS_PER_LDG];
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            mean[i] = 0.f;
            m2[i] = 0.f;
        }

        // The number of elements loaded by this CTA.
        int32_t cta_count = 0;
        int32_t global_batch_offset = n_blk_index * params.nhw * params.c;
        // int8 relevant
        // int8 output implies we have NC/32DHW32 input for bath fp16 and int8
        int32_t global_thread_c_input = (IS_INPUT_INT8 || IS_OUTPUT_INT8)
            ? thread_in_cta_c * ELEMENTS_PER_LDG + (cta_c % 32) // handle C_ELEMENTS_PER_CTA == 16 case
                + (cta_c / 32) * 32 * params.nhw
            : thread_c;
        int32_t stride_c_input = (IS_INPUT_INT8 || IS_OUTPUT_INT8) ? 32 : params.c;
        int32_t global_thread_c_output = (IS_OUTPUT_INT8)
            ? thread_in_cta_c * ELEMENTS_PER_LDG + (cta_c % 32) // handle C_ELEMENTS_PER_CTA == 16 case
                + (cta_c / 32) * 32 * params.nhw
            : thread_c;
        int32_t stride_c_output = (IS_OUTPUT_INT8) ? 32 : params.c;
        // The base pointer to load from.
        Input_Data_Type const* gmem_src
            = &reinterpret_cast<Input_Data_Type const*>(params.gmem_src)[global_thread_c_input + global_batch_offset];

        // Load the batch of elements. Compute the mean/var across those elements.
        int32_t const pixels_per_iteration = PIXELS_PER_CTA_IN_REGISTERS * gridDim.x;

        // outer loops
        int32_t OUTER_LOOPS = OUTER_LOOPS_ == 1 ? 1 : params.outer_loops;

#pragma unroll 1
        for (int32_t loop_i = 0; loop_i < OUTER_LOOPS; ++loop_i)
        {
            // The nhw position.
            int32_t nhw_regs = cta_nhw_regs + loop_i * pixels_per_iteration;

            cta_count += max(min(nhw_regs + PIXELS_PER_CTA_IN_REGISTERS, params.nhw) - max(nhw_regs, 0), 0);

            // Load the data and compute the local mean/sum and the variance.
            if (USE_ONLINE_APPROACH)
            {
                // Read the elements from memory.
                float is_valid[PIXELS_PER_THREAD_IN_REGISTERS];
#pragma unroll
                for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
                {
                    int32_t const idx = nhw_regs + thread_in_cta_nhw + i * PIXELS_PER_LDG;
                    zero(x_storage[i]);
                    is_valid[i] = 0.f;
                    if (idx < params.nhw && is_valid_c)
                    {
                        ldgStream(x_storage[i], &gmem_src[idx * stride_c_input]);
                        is_valid[i] = 1.f;
                    }
                }

// Do the math.
#pragma unroll
                for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
                {
                    // Convert to float.
                    float x_math[ELEMENTS_PER_LDG];
                    toFloat<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);

                    // Update the count.
                    count += is_valid[i];
                    // Invert the count.
                    float inv_count = is_valid[i] ? 1.f / count : 0.f;

// Update the mean and m2 using deltas.
#pragma unroll
                    for (int32_t j = 0; j < ELEMENTS_PER_LDG; ++j)
                    {
                        float delta0 = x_math[j] - mean[j];
                        mean[j] += delta0 * inv_count;
                        float delta1 = x_math[j] - mean[j];
                        m2[j] += delta0 * delta1 * is_valid[i];
                    }
                }
            }
            else
            {
// Read the elements from memory.
#pragma unroll
                for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
                {
                    int32_t const idx = nhw_regs + thread_in_cta_nhw + i * PIXELS_PER_LDG;
                    zero(x_storage[i]);
                    if (idx < params.nhw && is_valid_c)
                    {
                        ldgStream(x_storage[i], &gmem_src[idx * stride_c_input]);
                        count += 1.f;
                    }
                }

// Sum the elements in registers.
#pragma unroll
                for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
                {
                    // Convert to float.
                    float x_math[ELEMENTS_PER_LDG];
                    toFloat<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);

// Update the mean and m2 using deltas.
#pragma unroll
                    for (int32_t j = 0; j < ELEMENTS_PER_LDG; ++j)
                    {
                        mean[j] += x_math[j];
                    }
                }

                // Compute the mean.
                float inv_count = 1.f / count;
#pragma unroll
                for (int32_t j = 0; j < ELEMENTS_PER_LDG; ++j)
                {
                    mean[j] *= inv_count;
                }

// Compute the variance.
#pragma unroll
                for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
                {
                    // Convert to float.
                    float x_math[ELEMENTS_PER_LDG];
                    toFloat<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);

                    // Is it a valid pixel?
                    float is_valid = i < (int32_t) count ? 1.f : 0.f;
// Update the mean and m2 using deltas.
#pragma unroll
                    for (int32_t j = 0; j < ELEMENTS_PER_LDG; ++j)
                    {
                        m2[j] += (x_math[j] - mean[j]) * (x_math[j] - mean[j]) * is_valid;
                    }
                }
            }
        }

        // The elements to load and store in SMEM.
        int32_t smem_nhw = OUTER_LOOPS * pixels_per_iteration + cta_nhw_smem;
        // Load elements from SMEM, update the CTA count.
        int32_t pixels_in_smem = min(smem_nhw + PIXELS_PER_CTA_IN_SMEM, params.nhw) - max(smem_nhw, 0);
        if (pixels_in_smem > 0)
        {
            cta_count += pixels_in_smem;
            for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i)
            {
                int32_t const idx = smem_nhw + thread_in_cta_nhw + i * PIXELS_PER_LDG;
                float is_pixel_valid = (idx < params.nhw && is_valid_c) ? 1.f : 0.f;

                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG];
                ldgStream(x_storage_local, &gmem_src[(is_pixel_valid ? idx : 0) * stride_c_input]);

                // The offset to store in SMEM.
                int32_t const offset = i * THREADS_PER_CTA * PACKED_ELEMENTS_PER_LDG;
                // Store in SMEM.
                writeToSmem(&smem_storage[offset], threadIdx.x, x_storage_local);
                // Update the count.
                count += is_pixel_valid;
                // Invert the count.
                float inv_count = is_pixel_valid ? 1.f / count : 0.f;

                float x_math[ELEMENTS_PER_LDG];
                toFloat<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage_local, int8_in_scale);
// Update the mean and m2 using deltas.
#pragma unroll
                for (int32_t j = 0; j < ELEMENTS_PER_LDG; ++j)
                {
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
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            m1[i] = mean[i] * count;
        }

        // Run the parallel sum accross the CTA to get the local sum.
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(smem, m1, thread_in_cta_nhw);
        __syncthreads();

        // The values in shared memory correspond to the CTA-wide sums.
        readFromSmem(m1, smem, thread_in_cta_c);
        __syncthreads();

        // Adjust the variance.
        float inv_cta_count = 1.f / (float) cta_count;
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            float mean_diff = m1[i] * inv_cta_count - mean[i];
            m2[i] = m2[i] + mean_diff * mean_diff * count;
        }

        // Run the parallel sum accross the CTA to get the local adjusted variance.
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(smem, m2, thread_in_cta_nhw);

        // The workspace in global memory is distributed across the different CTA.
        int32_t gmem_sums_offset = nc_blk_index * gridDim.x * C_ELEMENTS_PER_CTA * 2;

        // Write the data for the CTA to global memory.
        GMEM_SUMS_TYPE* gmem_sums = &params.gmem_sums[gmem_sums_offset];
        if (threadIdx.x < THREADS_PER_PIXEL)
        {
            int32_t const idx = blockIdx.x * THREADS_PER_PIXEL + threadIdx.x;
            writeToGmem(&gmem_sums[0], idx, m1);
            writeToGmem(&gmem_sums[C_ELEMENTS_PER_CTA * gridDim.x], idx, m2);
        }

        // The memory location to store the number of pixels per CTA.
        int32_t* gmem_counts = &params.gmem_counts[nc_blk_index * gridDim.x];
        if (threadIdx.x == 0)
        {
            // gmem_counts[0] = cta_count;
            gmem_counts[blockIdx.x] = cta_count;
        }

        // Read the bias and scale.
        float bias[ELEMENTS_PER_LDG];
        float scale[ELEMENTS_PER_LDG];
        if (is_valid_c)
        {
            readFromGmem(bias, &params.gmem_bias[cta_c], thread_in_cta_c);
            readFromGmem(scale, &params.gmem_scale[cta_c], thread_in_cta_c);
        }

        // The counters to count how many CTAs have retired at this point. One per chunk of C.
        int32_t* gmem_retired_ctas = &params.gmem_retired_ctas[nc_blk_index];

        // Make sure the threads are done and reconverged.
        __syncthreads();

        // Register the CTA.
        int32_t expected_count = gridDim.x;
        if (threadIdx.x == 0)
        {
            // Issue the membar.
            __threadfence();
            // Notify that the CTA is done.
            int32_t val_to_add = 1;
            if (blockIdx.x == 0)
            {
                val_to_add = -(expected_count - 1);
            }
            atomicAdd(gmem_retired_ctas, val_to_add);
        }

        // Are all CTAs done?
        if (threadIdx.x == 0)
        {
            int32_t retired_ctas = -1;
            do
            {
                __threadfence();
                asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(retired_ctas) : "l"(gmem_retired_ctas));
            } while (retired_ctas != 0);
        }
        __threadfence();
        __syncthreads();

// Reset the mean to compute the global mean.
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            m1[i] = 0.f;
        }

// Build the global mean.
#pragma unroll 1
        for (int32_t idx = threadIdx.x; idx < THREADS_PER_PIXEL * gridDim.x; idx += THREADS_PER_CTA)
        {
            float tmp[ELEMENTS_PER_LDG];
            readFromGmem(tmp, gmem_sums, idx);

#pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
            {
                m1[i] += tmp[i];
            }
        }

        // Run the parallel sum accross the CTA to get the local sum.
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(smem, m1, thread_in_cta_nhw);
        __syncthreads();

        // The values in shared memory correspond to the CTA-wide sums.
        readFromSmem(m1, smem, thread_in_cta_c);
        __syncthreads();

        // Normalize the mean.
        float inv_count = 1.f / (float) params.nhw;
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            m1[i] = m1[i] * inv_count;
        }

// Reset the variance.
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            m2[i] = 0.f;
        }

// Build the global variance.
#pragma unroll 1
        for (int32_t idx = threadIdx.x; idx < THREADS_PER_PIXEL * gridDim.x; idx += THREADS_PER_CTA)
        {

            // Read the means computed by different CTAs (again). Reuse tmp if we have 1 iteration.
            float tmp_mean[ELEMENTS_PER_LDG], tmp_var[ELEMENTS_PER_LDG];
            readFromGmem(tmp_mean, &gmem_sums[0], idx);
            readFromGmem(tmp_var, &gmem_sums[C_ELEMENTS_PER_CTA * gridDim.x], idx);

            // Read the number of pixels visited by a given CTA.
            cta_count = __ldg(&gmem_counts[idx / THREADS_PER_PIXEL]);

            // Compute the diff to update the variance.
            float mean_diff[ELEMENTS_PER_LDG], inv_cta_count = 1.f / (float) cta_count;
#pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
            {
                mean_diff[i] = m1[i] - tmp_mean[i] * inv_cta_count;
            }

// Update the variance.
#pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
            {
                m2[i] += tmp_var[i] + mean_diff[i] * mean_diff[i] * (float) cta_count;
            }
        }

        // Run the parallel sum accross the CTA to get the local sum.
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(smem, m2, thread_in_cta_nhw);
        __syncthreads();

        readFromSmem(m2, smem, thread_in_cta_c);
        __syncthreads();

// Finalize the stddev.
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            m2[i] *= inv_count;
        }

        // store the saved mean/var
        float svarinv[ELEMENTS_PER_LDG];
        bool is_valid_for_saving = is_valid_c && blockIdx.x == 0 && thread_in_cta_nhw == 0;
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            svarinv[i] = rsqrtf(m2[i] + params.var_eps);
        }

#if ACCUM_MEAN_VAR_IN_FLOAT
        int32_t global_stats_offset = n_blk_index * params.c;
        if (is_valid_for_saving)
        {
            writeToGmem(params.gmem_saved_mean + global_stats_offset, thread_c / ELEMENTS_PER_LDG, m1);
            writeToGmem(params.gmem_saved_var + global_stats_offset, thread_c / ELEMENTS_PER_LDG, svarinv);
        }

        // store the running mean/var
        float rmean[ELEMENTS_PER_LDG];
        float rvar[ELEMENTS_PER_LDG];
        zero(rmean);
        zero(rvar);
        if (params.exp_avg_factor != 1.f && is_valid_for_saving)
        {
            readFromGmem(rmean, params.gmem_running_mean + global_stats_offset, thread_c / ELEMENTS_PER_LDG);
            readFromGmem(rvar, params.gmem_running_var + global_stats_offset, thread_c / ELEMENTS_PER_LDG);
        }
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            rmean[i] = (1.f - params.exp_avg_factor) * rmean[i] + params.exp_avg_factor * m1[i];
            rvar[i] = (1.f - params.exp_avg_factor) * rvar[i] + params.exp_avg_factor * m2[i];
        }
        if (is_valid_for_saving)
        {
            writeToGmem(params.gmem_running_mean + global_stats_offset, thread_c / ELEMENTS_PER_LDG, rmean);
            writeToGmem(params.gmem_running_var + global_stats_offset, thread_c / ELEMENTS_PER_LDG, rvar);
        }
#endif

// Update the scale with the stddev and eps.
#pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_LDG; ++i)
        {
            scale[i] *= svarinv[i];
        }

        // The base pointer to write to.
        Output_Data_Type* const gmem_dst
            = &reinterpret_cast<Output_Data_Type*>(params.gmem_dst)[global_thread_c_output + global_batch_offset];

// Store the elements in registers.
#pragma unroll 1
        for (int32_t loop_i = OUTER_LOOPS - 1; loop_i >= 0; --loop_i)
        {

            // The value for nhw.
            int32_t out_nhw = cta_nhw_regs + loop_i * pixels_per_iteration;

            // On CUDA-11.5 or above, full unrolling caused compiler to panic about register pressure and the perf
            // dropped significantly. Therefore, limit the extent of unrolling on CUDA-11.5 or above. The number "8" is
            // chosen based on experiments.
#if CUDA_VERSION >= 11050
#pragma unroll 8
#else
#pragma unroll
#endif
            // Normalize the elements and write to memory.
            for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
            {
                // Convert to float.
                float x_math[ELEMENTS_PER_LDG];
                toFloat<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage[i], int8_in_scale);

                // Normalize and apply activation function
                normalize(x_math, bias, scale, m1);
                if (params.use_relu)
                {
                    reluActivation(x_math, params.relu_alpha);
                }

                // Write back.
                int32_t const idx = out_nhw + thread_in_cta_nhw + i * PIXELS_PER_LDG;
                if ((unsigned) idx < params.nhw && is_valid_c)
                {
                    stgStream(&gmem_dst[idx * stride_c_output], x_math, int8_out_scale);
                }
            }

            // The next value of nhw.
            out_nhw -= pixels_per_iteration;

// Read the next elements from memory.
#pragma unroll
            for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i)
            {
                int32_t const idx = out_nhw + thread_in_cta_nhw + i * PIXELS_PER_LDG;
                if ((unsigned) idx < params.nhw && is_valid_c)
                {
                    ldgStream(x_storage[i], &gmem_src[idx * stride_c_output]);
                }
            }
        }

        // Normalize the elements from SMEM and write them out.
        if (pixels_in_smem > 0)
        {
            for (int32_t i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i)
            {
                // Read from SMEM.
                int32_t const offset = i * THREADS_PER_CTA * PACKED_ELEMENTS_PER_LDG;
                float x_math[ELEMENTS_PER_LDG];
                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG];
                readFromSmem(x_storage_local, &smem_storage[offset], threadIdx.x);
                toFloat<PACKED_ELEMENTS_PER_LDG, IS_INPUT_INT8>(x_math, x_storage_local, int8_in_scale);

                // Normalize and apply activation function
                normalize(x_math, bias, scale, m1);
                if (params.use_relu)
                {
                    reluActivation(x_math, params.relu_alpha);
                }

                // Write back.
                int32_t const idx = smem_nhw + thread_in_cta_nhw + i * PIXELS_PER_LDG;
                if ((unsigned) idx < params.nhw && is_valid_c)
                {
                    stgStream(&gmem_dst[idx * stride_c_output], x_math, int8_out_scale);
                }
            }
        }
        __syncthreads();
    } // blockIdx.y loop
}

template <typename Kernel_params>
dim3 estimateInGridDim(InstanceNormFwdParams const& params)
{
    dim3 grid_dim;
    grid_dim.x = divUp(params.nhw, Kernel_params::MIN_PIXELS_PER_CTA); // PIXELS_PER_CTA
    grid_dim.y = divUp(params.c, Kernel_params::C_ELEMENTS_PER_CTA) * params.n;
    grid_dim.z = 1; // params.n;

    return grid_dim;
}

template <typename Kernel_params>
void instanceNormBufferSizes(
    InstanceNormFwdParams const& params, size_t& size_sums, size_t& size_counts, size_t& size_retired_ctas)
{
    dim3 grid_dim = estimateInGridDim<Kernel_params>(params);

    size_sums = grid_dim.z * grid_dim.y * grid_dim.x * Kernel_params::THREADS_PER_PIXEL
        * Kernel_params::ELEMENTS_PER_LDG * 2 * sizeof(GMEM_SUMS_TYPE);
    size_counts = grid_dim.z * grid_dim.y * grid_dim.x * sizeof(int32_t);
    size_retired_ctas = grid_dim.z * grid_dim.y * sizeof(int32_t);

    size_sums = divUp(size_sums, 256) * 256;
    size_counts = divUp(size_counts, 256) * 256;
    size_retired_ctas = divUp(size_retired_ctas, 256) * 256;
}

template <typename Kernel_params>
int32_t instance_norm_fwd_launch(
    InstanceNormFwdContext const& context, InstanceNormFwdParams& params, cudaStream_t stream)
{

    size_t smem_size = Kernel_params::PIXELS_PER_THREAD_IN_SMEM * Kernel_params::THREADS_PER_CTA
        * Kernel_params::ELEMENTS_PER_LDG * sizeof(typename Kernel_params::StorageType);

    dim3 grid_dim = estimateInGridDim<Kernel_params>(params);

    params.c_blks = divUp(params.c, Kernel_params::C_ELEMENTS_PER_CTA);

    size_t size_retired_ctas = grid_dim.z * grid_dim.y * sizeof(int32_t);

#define KERNEL_RUN(OUTER_LOOPS, DESIRED_OCCUPANCY)                                                                     \
    {                                                                                                                  \
        PLUGIN_CHECK_CUDA(cudaMemsetAsync(params.gmem_retired_ctas, 0, size_retired_ctas, stream));                    \
        if (smem_size > 0)                                                                                             \
            PLUGIN_CHECK_CUDA(cudaFuncSetAttribute(                                                                    \
                instanceNormFwd<typename Kernel_params::StorageType, typename Kernel_params::Input_Data_Type,          \
                    typename Kernel_params::Output_Data_Type, Kernel_params::THREADS_PER_CTA,                          \
                    Kernel_params::THREADS_PER_PIXEL, Kernel_params::PIXELS_PER_THREAD_IN_REGISTERS,                   \
                    Kernel_params::PIXELS_PER_THREAD_IN_SMEM, Kernel_params::ELEMENTS_PER_LDG,                         \
                    Kernel_params::USE_ONLINE_APPROACH, OUTER_LOOPS, DESIRED_OCCUPANCY>,                               \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));                                              \
        instanceNormFwd<typename Kernel_params::StorageType, typename Kernel_params::Input_Data_Type,                  \
            typename Kernel_params::Output_Data_Type, Kernel_params::THREADS_PER_CTA,                                  \
            Kernel_params::THREADS_PER_PIXEL, Kernel_params::PIXELS_PER_THREAD_IN_REGISTERS,                           \
            Kernel_params::PIXELS_PER_THREAD_IN_SMEM, Kernel_params::ELEMENTS_PER_LDG,                                 \
            Kernel_params::USE_ONLINE_APPROACH, OUTER_LOOPS, DESIRED_OCCUPANCY>                                        \
            <<<grid_dim, Kernel_params::THREADS_PER_CTA, smem_size, stream>>>(params);                                 \
    }

    size_t total_smem_bytes
        = smem_size + Kernel_params::ELEMENTS_PER_LDG * Kernel_params::THREADS_PER_CTA * sizeof(float);
    int32_t smem_driven_fwd_occupancy = min(int32_t(context.sm_shared_size) / (int32_t) total_smem_bytes, (int32_t) 2);
    int32_t max_grid = context.sm_count * smem_driven_fwd_occupancy;
    if ((context.sm_version >= 700) && (context.sm_version < 800))
    {
        max_grid = max_grid - 4;
    }

    if (max_grid / int32_t(grid_dim.x) > 1)
    {
        grid_dim.y = max_grid / int32_t(grid_dim.x);
        grid_dim.y = int32_t(grid_dim.y) > params.c_blks * params.n ? params.c_blks * params.n : int32_t(grid_dim.y);
    }
    else
    {
        grid_dim.y = 1;
    }

    int32_t loop = 1;
    if (int32_t(grid_dim.x) <= max_grid)
    {
        if (smem_driven_fwd_occupancy >= 2)
        {
            KERNEL_RUN(1, 2);
        }
        else
        {
            KERNEL_RUN(1, 1);
        }
    }
    else
    {
        grid_dim.x = max_grid;
        int32_t nhw_in_regs
            = params.nhw - Kernel_params::PIXELS_PER_THREAD_IN_SMEM * Kernel_params::PIXELS_PER_LDG * grid_dim.x;
        int32_t pixels_per_iteration
            = Kernel_params::PIXELS_PER_THREAD_IN_REGISTERS * Kernel_params::PIXELS_PER_LDG * grid_dim.x;
        nhw_in_regs = (nhw_in_regs <= 0) ? pixels_per_iteration : nhw_in_regs;
        if (nhw_in_regs < 0)
        {
            nhw_in_regs = pixels_per_iteration;
            // make PIXELS_PER_THREAD_IN_SMEM <= PIXELS_PER_THREAD_IN_REGISTERS if the assert fails
            assert(pixels_per_iteration >= params.nhw);
        }

        loop = divUp(nhw_in_regs, pixels_per_iteration);
        params.outer_loops = loop;
        assert(loop >= 1);

        if (loop == 1)
        {
            if (smem_driven_fwd_occupancy >= 2)
            {
                KERNEL_RUN(1, 2);
            }
            else
            {
                KERNEL_RUN(1, 1);
            }
        }
        else
        {
            if (smem_driven_fwd_occupancy >= 2)
            {
                KERNEL_RUN(0, 2);
            }
            else
            {
                KERNEL_RUN(0, 1);
            }
        }
    }
    return loop;
}

static int32_t c_cond_g = 32;

void instanceNormBufferSizesDispatch(InstanceNormFwdContext const& context, InstanceNormFwdParams const& params,
    size_t& size_sums, size_t& size_counts, size_t& size_retired_ctas, int32_t input_data_type,
    int32_t output_data_type)
{
    if (input_data_type == 2 && output_data_type == 2)
    {
        switch (context.sm_version)
        {
        case 700:
            return instanceNormBufferSizes<kernel_params_32_int8_sm_700>(
                params, size_sums, size_counts, size_retired_ctas);
            break;
        case 720:
            return instanceNormBufferSizes<kernel_params_32_int8_sm_720>(
                params, size_sums, size_counts, size_retired_ctas);
            break;
        case 750:
            return instanceNormBufferSizes<kernel_params_32_int8_sm_750>(
                params, size_sums, size_counts, size_retired_ctas);
            break;
        case 800:
            return instanceNormBufferSizes<kernel_params_32_int8_sm_800>(
                params, size_sums, size_counts, size_retired_ctas);
            break;
        case 860:
            return instanceNormBufferSizes<kernel_params_32_int8_sm_860>(
                params, size_sums, size_counts, size_retired_ctas);
            break;
        case 870:
            return instanceNormBufferSizes<kernel_params_32_int8_sm_870>(
                params, size_sums, size_counts, size_retired_ctas);
            break;
        default:
            return instanceNormBufferSizes<kernel_params_32_int8>(params, size_sums, size_counts, size_retired_ctas);
            break;
        }
        return instanceNormBufferSizes<kernel_params_32_int8>(params, size_sums, size_counts, size_retired_ctas);
    }
    else if (input_data_type == 1 && output_data_type == 2)
    {
        return instanceNormBufferSizes<kernel_params_32_fp16_int8>(params, size_sums, size_counts, size_retired_ctas);
    }
    else if (input_data_type == 1 && output_data_type == 1)
    {
        if (params.c <= c_cond_g)
        {
            return instanceNormBufferSizes<kernel_params_32>(params, size_sums, size_counts, size_retired_ctas);
        }
        else
        {
            return instanceNormBufferSizes<kernel_params_64>(params, size_sums, size_counts, size_retired_ctas);
        }
    }
    else
    {
        fprintf(stderr, "Unsupported format combination by the instance norm kernel\n");
        assert(0);
    }
}

int32_t instanceNormFwdDispatch(InstanceNormFwdContext const& context, InstanceNormFwdParams& params,
    cudaStream_t stream, int32_t input_data_type, int32_t output_data_type)
{
    assert(context.sm_version >= 600);
    if (input_data_type == 2 && output_data_type == 2)
    {
        switch (context.sm_version)
        {
        case 700: return instance_norm_fwd_launch<kernel_params_32_int8_sm_700>(context, params, stream); break;
        case 720: return instance_norm_fwd_launch<kernel_params_32_int8_sm_720>(context, params, stream); break;
        case 750: return instance_norm_fwd_launch<kernel_params_32_int8_sm_750>(context, params, stream); break;
        case 800: return instance_norm_fwd_launch<kernel_params_32_int8_sm_800>(context, params, stream); break;
        case 860: return instance_norm_fwd_launch<kernel_params_32_int8_sm_860>(context, params, stream); break;
        case 870: return instance_norm_fwd_launch<kernel_params_32_int8_sm_870>(context, params, stream); break;
        default: return instance_norm_fwd_launch<kernel_params_32_int8>(context, params, stream); break;
        }
    }
    else if (input_data_type == 1 && output_data_type == 2)
    {
        return instance_norm_fwd_launch<kernel_params_32_fp16_int8>(context, params, stream);
    }
    else if (input_data_type == 1 && output_data_type == 1)
    {
        if (params.c <= c_cond_g)
        {
            return instance_norm_fwd_launch<kernel_params_32>(context, params, stream);
        }
        else
        {
            return instance_norm_fwd_launch<kernel_params_64>(context, params, stream);
        }
    }
    else
    {
        fprintf(stderr, "Unsupported format combination by the instance norm kernel\n");
        assert(0);
    }

    return 0;
}

} // namespace instance_norm_impl
