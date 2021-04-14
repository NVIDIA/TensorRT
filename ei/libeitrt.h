/* Edge Impulse inferencing library
 * Copyright (c) 2020 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <memory>

//forward declare the wrapper class side TensorRT so we don't bring in all the dependencies
class EiTrt;

namespace libeitrt
{

/**
 * @brief Creates and initializes an inference engine for TensorRT.
 * If the engine has already been created from the provided file path, then
 * the engine is loaded from disk.
 * 
 * The engine is then persisted via the EiTrt object until it is deleted, 
 * to provide for fastest inference with lowest overhead
 * 
 * WARNING: This function leaks..the handle can not be deleted b/c of forward declaration
 * The fix for this is to define an interface (virtual class) that has a virtual destructor
 * And also the infer function (although this way is more C friendly!)
 * My bad...should have done that from get go.
 * 
 * @param model_file_name Model file path.
 * Should have hash appended so that engines are regenerated when models change!
 * @return std::unique_ptr<EiTrt> EiTrt handle.  Contained ptr is NULL if error
 */
EiTrt* create_EiTrt(const char* model_file_name, bool debug);

/**
 * @brief Perform inference
 * 
 * @param ei_trt_handle Created handle to inference engine
 * @param[in] input Input features (buffer member of ei_matrix)
 * @param[out] output Buffer to write output to 
 * @param output_size Buffer size
 * @return int 0 on success, <0 otherwise
 */
int infer(EiTrt* ei_trt_handle, float* input, float* output, int output_size);
}