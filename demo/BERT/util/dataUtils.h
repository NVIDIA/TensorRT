/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRT_DATA_UTILS_H
#define TRT_DATA_UTILS_H

#include <NvInfer.h>
#include <bertUtils.h>
#include <map>
#include <string>
#include <vector>

namespace bert
{

//! \brief Loads a dictionary of weights
//! \details The function loads the weights of the BERT network from a weights file. The Weights in the dictionary own
//! the storage behind the Weights::values pointer. It is therefore the callers responsibility to free it. See also helpers/convert_weights.py
//!\param path path to inputs
//!\param weightMap map of weights that the function will populate
void loadWeights(const std::string& path, WeightMap& weightMap);

//! \brief Loads a batch of inputs
//! \details The function loads inputs for the network consisting batches of tokenized text, input masks and segement ids.
//! Each batch is represented as a matrix of int32 elements of size sequence length x batch size.
//! Each batch is assumed to have the same sequence length S, which the function outputs, so that the batch size of each batch can be computed.
//! If multiple batches are present, tokens, masks and segement ids are associated with each other in the order in which they are encountered.
//! See also helpers/generate_dbg.py
//!\param path Path to the input file
//!\param Bmax Ouputs the largest batch size encountered
//!\param S Outputs the sequence lenght of the inputs
//!\param inputIds Vector of input id batches
//!\param inputMasks  Vector of input mask batches
//!\param segmentIds Vector of segment id batches
void loadInputs(const std::string& path, int& Bmax, int& S, std::vector<nvinfer1::Weights>& inputIds,
    std::vector<nvinfer1::Weights>& inputMasks, std::vector<nvinfer1::Weights>& segmentIds,
    std::vector<nvinfer1::Dims> & inputDims);

//! \brief infers the characteristic sizes of the BERT network from loaded weights
//! \param weightMap Loaded weights. See loadWeights
//! \param hiddenSize Outputs hidden size, which is numHeads * headSize and equals the embeddings dimension
//! \param intermediateSize Outputs the intermediate size in the transformer which is typically 4 x hidden size
//! \param numHiddenLayers Outputs the number of transformer layers requested
void inferNetworkSizes(const WeightMap& weightMap, int& hiddenSize,
    int& intermediateSize, int& numHiddenLayers);

//! \brief Transposes logits from BxSx2 to 2xBxS
//! \details Due to a limitation of TensorRT, the network itself cannot transpose the output of the squad logits
//! into the desired shape. This function performs the transpose on the input in-place.
//! \param logits Vector of length B*S*2. Will contain the result as of the tranpose
//! \param B batch size
//! \param S sequence length
void transposeLogits(std::vector<float>& logits, const int B, const int S);

}

#endif // TRT_DATA_UTILS_H
