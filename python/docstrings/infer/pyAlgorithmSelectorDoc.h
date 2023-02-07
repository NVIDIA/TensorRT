/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file contains all AlgorithmSelector docstrings, since these are typically too long to keep in the binding code.
#pragma once

namespace tensorrt
{
namespace IAlgorithmIOInfoDOC
{
constexpr const char* descr = R"trtdoc(
    This class carries information about input or output of the algorithm.
    IAlgorithmIOInfo for all the input and output along with IAlgorithmVariant denotes the variation of algorithm
    and can be used to select or reproduce an algorithm using IAlgorithmSelector.select_algorithms().

    :ivar tensor_format: :class:`TensorFormat` TensorFormat of the input/output of algorithm.
    :ivar dtype: :class:`DataType`  DataType of the input/output of algorithm.
    :ivar strides: :class:`Dims` strides of the input/output tensor of algorithm.
)trtdoc";
} // namespace IAlgorithmIOInfoDOC

namespace IAlgorithmVariantDOC
{
constexpr const char* descr = R"trtdoc(
    provides a unique 128-bit identifier, which along with the input and output information
    denotes the variation of algorithm and can be used to select or reproduce an algorithm,
    using IAlgorithmSelector.select_algorithms()
    see IAlgorithmIOInfo, IAlgorithm, IAlgorithmSelector.select_algorithms()
    note A single implementation can have multiple tactics.

    :ivar implementation: :class:`int` implementation of the algorithm.
    :ivar tactic: :class:`int`  tactic of the algorithm.
)trtdoc";

} // namespace IAlgorithmVariantDOC

namespace IAlgorithmContextDoc
{
constexpr const char* descr = R"trtdoc(
    Describes the context and requirements, that could be fulfilled by one or
    more instances of IAlgorithm.
    see IAlgorithm

    :ivar name: :class:`str` name of the algorithm node.
    :ivar num_inputs: :class:`int`  number of inputs of the algorithm.
    :ivar num_outputs: :class:`int` number of outputs of the algorithm.
)trtdoc";

constexpr const char* get_shape = R"trtdoc(
    Get the minimum / optimum / maximum dimensions for a dynamic input tensor.

    :arg index: Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs and the outputs.

    :returns: A `List[Dims]` of length 3, containing the minimum, optimum, and maximum shapes, in that order. If the shapes have not been set yet, an empty list is returned.`
)trtdoc";
} // namespace IAlgorithmContextDoc

namespace IAlgorithmDoc
{
constexpr const char* descr = R"trtdoc(
        Application-implemented interface for selecting and reporting the tactic selection of a layer.
        Tactic Selection is a step performed by the builder for deciding best algorithms for a layer.

    :ivar algorithm_variant: :class:`IAlgorithmVariant&`  the algorithm variant.
    :ivar timing_msec: :class:`float` The time in milliseconds to execute the algorithm.
    :ivar workspace_size: :class:`int` The size of the GPU temporary memory in bytes which the algorithm uses at execution time.
)trtdoc";

constexpr const char* get_algorithm_io_info = R"trtdoc(
    A single call for both inputs and outputs. Incremental numbers assigned to indices of inputs and the outputs.

    :arg index: Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs and the outputs.

    :returns: A :class:`IAlgorithmIOInfo&`
)trtdoc";
} // namespace IAlgorithmDoc

namespace IAlgorithmSelectorDoc
{
constexpr const char* descr = R"trtdoc(
    Interface implemented by application for selecting and reporting algorithms of a layer provided by the
    builder.
    note A layer in context of algorithm selection may be different from ILayer in INetworkDefiniton.
    For example, an algorithm might be implementing a conglomeration of multiple ILayers in INetworkDefinition.

    To implement a custom algorithm selector, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyAlgoSelector(trt.IAlgorithmSelector):
            def __init__(self):
                trt.IAlgorithmSelector.__init__(self)

)trtdoc";

constexpr const char* select_algorithms = R"trtdoc(
    Select Algorithms for a layer from the given list of algorithm choices.

    Note: TRT uses its default algorithm selection to choose from the list returned by the user.
    If the returned list is empty, TRT’s default algorithm selection is used unless strict type constraints are set.
    The list of choices is valid only for this specific algorithm context.

    For example, the simplest implementation looks like this:
    ::

        def select_algorithms(self, context, choices):
            assert len(choices) > 0
            return list(range(len(choices)))

    :arg context: The context for which the algorithm choices are valid.
    :arg choices: The list of algorithm choices to select for implementation of this layer.

    :returns: A :class:`List[int]` indicating the indices from the choices vector that TensorRT should choose from.

)trtdoc";

constexpr const char* report_algorithms = R"trtdoc(
    Called by TensorRT to report choices it made.

    Note: For a given optimization profile, this call comes after all calls to select_algorithms.
    choices[i] is the choice that TensorRT made for algoContexts[i], for i in [0, num_algorithms-1]

    For example, a possible implementation may look like this:
    ::

        def report_algorithms(self, contexts, choices):
            # Prints the time of the chosen algorithm by TRT from the
            # selection list passed in by select_algorithms
            for choice in choices:
                print(choice.timing_msec)

    :arg contexts: The list of all algorithm contexts.
    :arg choices: The list of algorithm choices made by TensorRT corresponding to each context.
)trtdoc";
} // namespace IAlgorithmSelectorDoc
} // namespace tensorrt
