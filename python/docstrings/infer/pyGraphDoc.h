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

// This file contains all INetworkDefinition related docstrings, since these are typically too long to keep in the binding code.
#pragma once

namespace tensorrt
{
    namespace LayerTypeDoc
    {

        constexpr const char* descr = R"trtdoc(Type of Layer)trtdoc";
        constexpr const char* CONVOLUTION = R"trtdoc(Convolution layer)trtdoc";
        constexpr const char* FULLY_CONNECTED = R"trtdoc(Fully connected layer)trtdoc";
        constexpr const char* ACTIVATION = R"trtdoc(Activation layer)trtdoc";
        constexpr const char* POOLING = R"trtdoc(Pooling layer)trtdoc";
        constexpr const char* LRN = R"trtdoc(LRN layer)trtdoc";
        constexpr const char* SCALE = R"trtdoc(Scale layer)trtdoc";
        constexpr const char* SOFTMAX = R"trtdoc(Softmax layer)trtdoc";
        constexpr const char* DECONVOLUTION = R"trtdoc(Deconvolution layer)trtdoc";
        constexpr const char* CONCATENATION = R"trtdoc(Concatenation layer)trtdoc";
        constexpr const char* ELEMENTWISE = R"trtdoc(Elementwise layer)trtdoc";
        constexpr const char* PLUGIN = R"trtdoc(Plugin layer)trtdoc";
        constexpr const char* RNN = R"trtdoc(RNN layer)trtdoc";
        constexpr const char* UNARY = R"trtdoc(Unary layer)trtdoc";
        constexpr const char* PADDING = R"trtdoc(Padding layer)trtdoc";
        constexpr const char* SHUFFLE = R"trtdoc(Shuffle layer)trtdoc";
        constexpr const char* REDUCE = R"trtdoc(Reduce layer)trtdoc";
        constexpr const char* TOPK = R"trtdoc(TopK layer)trtdoc";
        constexpr const char* GATHER = R"trtdoc(Gather layer)trtdoc";
        constexpr const char* MATRIX_MULTIPLY = R"trtdoc(Matrix multiply layer)trtdoc";
        constexpr const char* RAGGED_SOFTMAX = R"trtdoc(Ragged softmax layer)trtdoc";
        constexpr const char* CONSTANT = R"trtdoc(Constant layer)trtdoc";
        constexpr const char* RNN_V2 = R"trtdoc(RNNv2 layer)trtdoc";
        constexpr const char* IDENTITY = R"trtdoc(Identity layer)trtdoc";
        constexpr const char* PLUGIN_V2 = R"trtdoc(PluginV2 layer)trtdoc";
        constexpr const char* SLICE = R"trtdoc(Slice layer)trtdoc";
        constexpr const char* SHAPE = R"trtdoc(Shape layer)trtdoc";
        constexpr const char* PARAMETRIC_RELU = R"trtdoc(Parametric ReLU layer)trtdoc";
        constexpr const char* RESIZE = R"trtdoc(Resize layer)trtdoc";
        constexpr const char* TRIP_LIMIT = R"trtdoc(Loop Trip limit layer)trtdoc";
        constexpr const char* RECURRENCE = R"trtdoc(Loop Recurrence layer)trtdoc";
        constexpr const char* ITERATOR = R"trtdoc(Loop Iterator layer)trtdoc";
        constexpr const char* LOOP_OUTPUT = R"trtdoc(Loop output layer)trtdoc";
        constexpr const char* SELECT = R"trtdoc(Select layer)trtdoc";
        constexpr const char* FILL = R"trtdoc(Fill layer)trtdoc";
    } // LayerTypeDoc

    namespace TensorLocationDoc
    {
        constexpr const char* descr = R"trtdoc(The physical location of the data.)trtdoc";

        constexpr const char* DEVICE = R"trtdoc(Data is stored on the device.)trtdoc";
        constexpr const char* HOST = R"trtdoc(Data is stored on the device.)trtdoc";
    } // TensorLocationDoc

    namespace TensorFormatDoc
    {
        constexpr const char* descr = R"trtdoc(
            Format of the input/output tensors.

            This enum is extended to be used by both plugins and reformat-free network I/O tensors.

            For more information about data formats, see the topic "Data Format Description" located in the
            TensorRT Developer Guide (https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html).
        )trtdoc";

        constexpr const char* LINEAR = R"trtdoc(
            Row major linear format.

            For a tensor with dimensions {N, C, H, W}, the W axis always has unit stride, and the stride of every other axis is at least the the product of of the next dimension times the next stride. the strides are the same as for a C array with dimensions [N][C][H][W].
        )trtdoc";

        constexpr const char* CHW2 = R"trtdoc(
            Two wide channel vectorized row major format.

            This format is bound to FP16. It is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/2][h][w][c%2].
        )trtdoc";

        constexpr const char* HWC8 = R"trtdoc(
            Eight channel format where C is padded to a multiple of 8.

            This format is bound to FP16. It is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to the array with dimensions [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][h][w][c].
        )trtdoc";

        constexpr const char* CHW4 = R"trtdoc(
            Four wide channel vectorized row major format.
            This format is bound to INT8. It is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/4][h][w][c%4].
        )trtdoc";

        constexpr const char* CHW16 = R"trtdoc(
            Sixteen wide channel vectorized row major format.

            This format is bound to FP16. It is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/16][h][w][c%16].
        )trtdoc";

        constexpr const char* CHW32 = R"trtdoc(
            Thirty-two wide channel vectorized row major format.

            This format is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/32][h][w][c%32].
        )trtdoc";

        constexpr const char* DHWC8 = R"trtdoc(
            Eight channel format where C is padded to a multiple of 8.

            This format is bound to FP16, and it is only available for dimensions >= 4.

            For a tensor with dimensions {N, C, D, H, W}, the memory layout is equivalent to an array with dimensions [N][D][H][W][(C+7)/8*8], with the tensor coordinates (n, c, d, h, w) mapping to array subscript [n][d][h][w][c].
        )trtdoc";

        constexpr const char* CDHW32 = R"trtdoc(
            Thirty-two wide channel vectorized row major format with 3 spatial dimensions.

            This format is bound to FP16 and INT8. It is only available for dimensions >= 4.

            For a tensor with dimensions {N, C, D, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+31)/32][D][H][W][32], with the tensor coordinates (n, d, c, h, w) mapping to array subscript [n][c/32][d][h][w][c%32].
        )trtdoc";
        
        constexpr const char* HWC = R"trtdoc(
            Non-vectorized channel-last format.
            This format is bound to FP32 and is only available for dimensions >= 3.
        )trtdoc";

        constexpr const char* DLA_LINEAR = R"trtdoc(
            DLA planar format. Row major format. The stride for stepping along the H axis is rounded up to 64 bytes.
            
            This format is bound to FP16/Int8 and is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][C][H][roundUp(W, 64/elementSize)] where elementSize is 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c][h][w].
        )trtdoc";

        constexpr const char* DLA_HWC4 = R"trtdoc(
            DLA image format. channel-last format. C can only be 1, 3, 4. If C == 3 it will be rounded to 4. The stride for stepping along the H axis is rounded up to 32 bytes.
            
            This format is bound to FP16/Int8 and is only available for dimensions >= 3.

            For a tensor with dimensions {N, C, H, W}, with C’ is 1, 4, 4 when C is 1, 3, 4 respectively, the memory layout is equivalent to a C array with dimensions [N][H][roundUp(W, 32/C'/elementSize)][C'] where elementSize is 2 for FP16 and 1 for Int8, C' is the rounded C. The tensor coordinates (n, c, h, w) maps to array subscript [n][h][w][c].
        )trtdoc";

    } // TensorFormatDoc


    namespace ITensorDoc
    {
        constexpr const char* descr = R"trtdoc(
            A tensor in an :class:`INetworkDefinition` .

            :ivar name: :class:`str` The tensor name. For a network input, the name is assigned by the application. For tensors which are layer outputs, a default name is assigned consisting of the layer name followed by the index of the output in brackets.

            :ivar shape: :class:`Dims` The shape of a tensor. For a network input the shape is assigned by the application. For a network output it is computed based on the layer parameters and the inputs to the layer. If a tensor size or a parameter is modified in the network, the shape of all dependent tensors will be recomputed. This call is only legal for network input tensors, since the shape of layer output tensors are inferred based on layer inputs and parameters.

            :ivar dtype: :class:`DataType` The data type of a tensor. The type is unchanged if the type is invalid for the given tensor. 

            :ivar broadcast_across_batch: :class:`bool` Whether to enable broadcast of tensor across the batch. When a tensor is broadcast across a batch, it has the same value for every member in the batch. Memory is only allocated once for the single member. This method is only valid for network input tensors, since the flags of layer output tensors are inferred based on layer inputs and parameters. If this state is modified for a tensor in the network, the states of all dependent tensors will be recomputed.

            :ivar location: :class:`TensorLocation` The storage location of a tensor.
            :ivar is_network_input: :class:`bool` Whether the tensor is a network input.
            :ivar is_network_output: :class:`bool` Whether the tensor is a network output.
            :ivar dynamic_range: :class:`Tuple[float, float]` A tuple containing the [minimum, maximum] of the dynamic range, or :class:`None` if the range was not set.
            :ivar is_shape: :class:`bool` Whether the tensor is a shape tensor.
            :ivar allowed_formats: :class:`int` The allowed set of TensorFormat candidates. This should be an integer consisting of one or more :class:`TensorFormat` s, combined via bitwise OR after bit shifting. For example, ``1 << int(TensorFormats.CHW4) | 1 << int(TensorFormat.CHW32)``.
        )trtdoc";

        constexpr const char* set_dynamic_range = R"trtdoc(
            Set dynamic range for the tensor.
            NOTE: It is suggested to use ``tensor.dynamic_range = (min, max)`` instead.

            :arg min: Minimum of the dynamic range.
            :arg max: Maximum of the dyanmic range.
            :returns: true if succeed in setting range. Otherwise false.
        )trtdoc";

        constexpr const char* get_dynamic_range = R"trtdoc(
            Get dynamic range for the tensor.
            NOTE: It is suggested to use ``tensor.dynamic_range`` instead, which is a tuple including both the minimum and maximum of the dynamic range.

            :returns: The absolute maximum of the dynamic range.
        )trtdoc";

        constexpr const char* reset_dynamic_range = R"trtdoc(
            Undo the effect of setting the dynamic range.
        )trtdoc";
    } // ITensorDoc

    namespace ILayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            Base class for all layer classes in an :class:`INetworkDefinition` .

            :ivar name: :class:`str` The name of the layer.
            :ivar type: :class:`LayerType` The type of the layer.
            :ivar num_inputs: :class:`int` The number of inputs of the layer.
            :ivar num_outputs: :class:`int` The number of outputs of the layer.
            :ivar precision: :class:`DataType` The computation precision.
            :ivar precision_is_set: :class:`bool` Whether the precision is set or not.
        )trtdoc";

        constexpr const char* set_input = R"trtdoc(
            Set the layer input corresponding to the given index.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        )trtdoc";


        constexpr const char* get_input = R"trtdoc(
            Get the layer input corresponding to the given index.

            :arg index: The index of the input tensor.

            :returns: The input tensor, or :class:`None` if the index is out of range.
        )trtdoc";

        constexpr const char* get_output = R"trtdoc(
            Get the layer output corresponding to the given index.

            :arg index: The index of the output tensor.

            :returns: The output tensor, or :class:`None` if the index is out of range.
        )trtdoc";

        constexpr const char* reset_precision = R"trtdoc(
            Reset the computation precision of the layer.
        )trtdoc";

        constexpr const char* set_output_type = R"trtdoc(
            Constraint layer to generate output data with given type.
            Note that this method cannot be used to set the data type
            of the second output tensor of the topK layer. The data
            type of the second output tensor of the topK layer is always Int32.

            :arg index: The index of the output tensor to set the type.
            :arg dtype: DataType of the output.
        )trtdoc";

        constexpr const char* get_output_type = R"trtdoc(
            Get the output type of the layer.

            :arg index: The index of the output tensor.

            :returns: The output precision. Default : DataType.FLOAT.
        )trtdoc";

        constexpr const char* output_type_is_set = R"trtdoc(
            Whether the output type has been set for this layer.

            :arg index: The index of the output.

            :returns: Whether the output type has been explicitly set.
        )trtdoc";

        constexpr const char* reset_output_type = R"trtdoc(
            Reset output type of this layer.

            :arg index: The index of the output.
        )trtdoc";

    } // ILayerDoc

    namespace PaddingModeDoc
    {
        constexpr const char* descr = R"trtdoc(
            Enumerates types of padding available in convolution, deconvolution and pooling layers.
            Padding mode takes precedence if both :attr:`padding_mode` and :attr:`pre_padding` are set.

            |  EXPLICIT* corresponds to explicit padding.
            |  SAME* implicitly calculates padding such that the output dimensions are the same as the input dimensions. For convolution and pooling,
                output dimensions are determined by ceil(input dimensions, stride).
            |  CAFFE* corresponds to symmetric padding.
        )trtdoc";

        constexpr const char* EXPLICIT_ROUND_DOWN = R"trtdoc(Use explicit padding, rounding the output size down)trtdoc";
        constexpr const char* EXPLICIT_ROUND_UP = R"trtdoc(Use explicit padding, rounding the output size up)trtdoc";
        constexpr const char* SAME_UPPER = R"trtdoc(Use SAME padding, with :attr:`pre_padding` <= :attr:`post_padding` )trtdoc";
        constexpr const char* SAME_LOWER = R"trtdoc(Use SAME padding, with :attr:`pre_padding` >= :attr:`post_padding` )trtdoc";
        constexpr const char* CAFFE_ROUND_DOWN = R"trtdoc(Use CAFFE padding, rounding the output size down)trtdoc";
        constexpr const char* CAFFE_ROUND_UP = R"trtdoc(Use CAFFE padding, rounding the output size up)trtdoc";

    } // PaddingModeDoc

    namespace IConvolutionLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A convolution layer in an :class:`INetworkDefinition` .

            This layer performs a correlation operation between 3-dimensional filter with a 4-dimensional tensor to produce another 4-dimensional tensor.

            An optional bias argument is supported, which adds a per-channel constant to each value in the output.

            :ivar kernel_size: :class:`DimsHW` The HW kernel size of the convolution.
            :ivar num_output_maps: :class:`int` The number of output maps for the convolution.
            :ivar stride: :class:`DimsHW` The stride of the convolution. Default: (1, 1)
            :ivar padding: :class:`DimsHW` The padding of the convolution. The input will be zero-padded by this number of elements in the height and width directions. If the padding is asymmetric, this value corresponds to the pre-padding. Default: (0, 0)
            :ivar pre_padding: :class:`DimsHW` The pre-padding. The start of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
            :ivar post_padding: :class:`DimsHW` The post-padding. The end of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
            :ivar padding_mode: :class:`PaddingMode` The padding mode. Padding mode takes precedence if both :attr:`IConvolutionLayer.padding_mode` and either :attr:`IConvolutionLayer.pre_padding` or :attr:`IConvolutionLayer.post_padding` are set.
            :ivar num_groups: :class:`int` The number of groups for a convolution. The input tensor channels are divided into this many groups, and a convolution is executed for each group, using a filter per group. The results of the group convolutions are concatenated to form the output. **Note** When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output. Default: 1.
            :ivar kernel: :class:`Weights` The kernel weights for the convolution. The weights are specified as a contiguous array in `GKCRS` order, where `G` is the number of groups, `K` the number of output feature maps, `C` the number of input channels, and `R` and `S` are the height and width of the filter.
            :ivar bias: :class:`Weights` The bias weights for the convolution. Bias is optional. To omit bias, set this to an empty :class:`Weights` object. The bias is applied per-channel, so the number of weights (if non-zero) must be equal to the number of output feature maps.
            :ivar dilation: :class:`DimsHW` The dilation for a convolution. Default: (1, 1)
            :ivar kernel_size_nd: :class:`Dims` The multi-dimension kernel size of the convolution.
            :ivar stride_nd: :class:`Dims` The multi-dimension stride of the convolution. Default: (1, ..., 1)
            :ivar padding_nd: :class:`Dims` The multi-dimension padding of the convolution. The input will be zero-padded by this number of elements in each dimension. If the padding is asymmetric, this value corresponds to the pre-padding. Default: (0, ..., 0)
            :ivar dilation_nd: :class:`Dims` The multi-dimension dilation for the convolution. Default: (1, ..., 1)
        )trtdoc";
    } // IConvolutionLayerDoc

    namespace IFullyConnectedLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A fully connected layer in an :class:`INetworkDefinition` .

            This layer expects an input tensor of three or more non-batch dimensions.  The input is automatically reshaped into an `MxV` tensor `X`, where `V` is a product of the last three dimensions and `M` is a product of the remaining dimensions (where the product over 0 dimensions is defined as 1).  For example:

            - If the input tensor has shape `{C, H, W}`, then the tensor is reshaped into `{1, C*H*W}` .
            - If the input tensor has shape `{P, C, H, W}`, then the tensor is reshaped into `{P, C*H*W}` .

            The layer then performs:

            :math:`Y := matmul(X, W^T) + bias`

            Where `X` is the `MxV` tensor defined above, `W` is the `KxV` weight tensor of the layer, and `bias` is a row vector size `K` that is broadcasted to `MxK` .  `K` is the number of output channels, and configurable via :attr:`IFullyConnectedLayer.num_output_channels` .  If `bias` is not specified, it is implicitly `0` .

            The `MxK` result `Y` is then reshaped such that the last three dimensions are `{K, 1, 1}` and the remaining dimensions match the dimensions of the input tensor. For example:

            - If the input tensor has shape `{C, H, W}`, then the output tensor will have shape `{K, 1, 1}` .
            - If the input tensor has shape `{P, C, H, W}`, then the output tensor will have shape `{P, K, 1, 1}` .

            :ivar num_output_channels: :class:`int` The number of output channels `K` from the fully connected layer.
            :ivar kernel: :class:`Weights` The kernel weights, given as a `KxC` matrix in row-major order.
            :ivar bias: :class:`Weights` The bias weights. Bias is optional. To omit bias, set this to an empty :class:`Weights` object.
        )trtdoc";
    } // IFullyConnectedLayerDoc

    namespace ActivationTypeDoc
    {
        constexpr const char* descr = R"trtdoc(The type of activation to perform.)trtdoc";

        constexpr const char* RELU = R"trtdoc(Rectified Linear activation)trtdoc";
        constexpr const char* SIGMOID = R"trtdoc(Sigmoid activation)trtdoc";
        constexpr const char* TANH = R"trtdoc(Hyperbolic Tangent activation)trtdoc";
        constexpr const char* LEAKY_RELU = R"trtdoc(Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0)trtdoc";
        constexpr const char* ELU = R"trtdoc(Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0)trtdoc";
        constexpr const char* SELU = R"trtdoc(Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0)trtdoc";
        constexpr const char* SOFTSIGN = R"trtdoc(Softsign activation: f(x) = x / (1 + abs(x)))trtdoc";
        constexpr const char* SOFTPLUS = R"trtdoc(Softplus activation: f(x) = alpha * log(exp(beta * x) + 1))trtdoc";
        constexpr const char* CLIP = R"trtdoc(Clip activation: f(x) = max(alpha, min(beta, x)))trtdoc";
        constexpr const char* HARD_SIGMOID = R"trtdoc(Hard sigmoid activation: f(x) = max(0, min(1, alpha * x + beta)))trtdoc";
        constexpr const char* SCALED_TANH = R"trtdoc(Scaled Tanh activation: f(x) = alpha * tanh(beta * x))trtdoc";
        constexpr const char* THRESHOLDED_RELU = R"trtdoc(Thresholded Relu activation: f(x) = x if x > alpha, f(x) = 0 if x <= alpha)trtdoc";

    } // ActivationTypeDoc

    namespace IActivationLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            An Activation layer in an :class:`INetworkDefinition` . This layer applies a per-element activation function to its input. The output has the same shape as the input.

            :ivar type: :class:`ActivationType` The type of activation to be performed.
            :ivar alpha: :class:`float` The alpha parameter that is used by some parametric activations (LEAKY_RELU, ELU, SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH). Other activations ignore this parameter.
            :ivar beta: :class:`float` The beta parameter that is used by some parametric activations (SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH). Other activations ignore this parameter.
        )trtdoc";
    } // IActivationLayerDoc

    namespace PoolingTypeDoc
    {
        constexpr const char* descr = R"trtdoc(The type of pooling to perform in a pooling layer.)trtdoc";

        constexpr const char* MAX = R"trtdoc(Maximum over elements)trtdoc";
        constexpr const char* AVERAGE = R"trtdoc(Average over elements. If the tensor is padded, the count includes the padding)trtdoc";
        constexpr const char* MAX_AVERAGE_BLEND = R"trtdoc(Blending between the max pooling and average pooling: `(1-blendFactor)*maxPool + blendFactor*avgPool`)trtdoc";
    } // PoolingTypeDoc


    namespace IPoolingLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A Pooling layer in an :class:`INetworkDefinition` . The layer applies a reduction operation within a window over the input.

            :ivar type: :class:`PoolingType` The type of pooling to be performed.
            :ivar window_size: :class:`DimsHW` The window size for pooling.
            :ivar stride: :class:`DimsHW` The stride for pooling. Default: (1, 1)
            :ivar padding: :class:`DimsHW` The padding for pooling. Default: (0, 0)
            :ivar pre_padding: :class:`DimsHW` The pre-padding. The start of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
            :ivar post_padding: :class:`DimsHW` The post-padding. The end of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
            :ivar padding_mode: :class:`PaddingMode` The padding mode. Padding mode takes precedence if both :attr:`IPoolingLayer.padding_mode` and either :attr:`IPoolingLayer.pre_padding` or :attr:`IPoolingLayer.post_padding` are set.
            :ivar blend_factor: :class:`float` The blending factor for the max_average_blend mode: :math:`max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool` . ``blend_factor`` is a user value in [0,1] with the default value of 0.0. This value only applies for the :const:`PoolingType.MAX_AVERAGE_BLEND` mode.
            :ivar average_count_excludes_padding: :class:`bool` Whether average pooling uses as a denominator the overlap area between the window and the unpadded input. If this is not set, the denominator is the overlap between the pooling window and the padded input. Default: True
            :ivar window_size_nd: :class:`Dims` The multi-dimension window size for pooling.
            :ivar stride_nd: :class:`Dims` The multi-dimension stride for pooling. Default: (1, ..., 1)
            :ivar padding_nd: :class:`Dims` The multi-dimension padding for pooling. Default: (0, ..., 0)
        )trtdoc";
    } // IPoolingLayerDoc

    namespace ILRNLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A LRN layer in an :class:`INetworkDefinition` . The output size is the same as the input size.

            :ivar window_size: :class:`int` The LRN window size. The window size must be odd and in the range of [1, 15].
            :ivar alpha: :class:`float` The LRN alpha value. The valid range is [-1e20, 1e20].
            :ivar beta: :class:`float` The LRN beta value. The valid range is [0.01, 1e5f].
            :ivar k: :class:`float` The LRN K value. The valid range is [1e-5, 1e10].
        )trtdoc";
    } // ILRNLayerDoc

    namespace ScaleModeDoc
    {
        constexpr const char* descr = R"trtdoc(Controls how scale is applied in a Scale layer.)trtdoc";

        constexpr const char* UNIFORM = R"trtdoc(Identical coefficients across all elements of the tensor.)trtdoc";
        constexpr const char* CHANNEL = R"trtdoc(Per-channel coefficients. The channel dimension is assumed to be the third to last dimension.)trtdoc";
        constexpr const char* ELEMENTWISE = R"trtdoc(Elementwise coefficients.)trtdoc";
    } // ScaleModeDoc


    namespace IScaleLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A Scale layer in an :class:`INetworkDefinition` .

            This layer applies a per-element computation to its input:

            :math:`output = (input * scale + shift) ^ power`

            The coefficients can be applied on a per-tensor, per-channel, or per-element basis.

            **Note**
            If the number of weights is 0, then a default value is used for shift, power, and scale. The default shift is 0, the default power is 1, and the default scale is 1.

            The output size is the same as the input size.

            **Note**
            The input tensor for this layer is required to have a minimum of 3 dimensions.

            :ivar mode: :class:`ScaleMode` The scale mode.
            :ivar shift: :class:`Weights` The shift value.
            :ivar scale: :class:`Weights` The scale value.
            :ivar power: :class:`Weights` The power value.
            :ivar channel_axis: :class:`int` The channel axis.
        )trtdoc";
    } // IScaleLayerDoc


    namespace ISoftMaxLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A Softmax layer in an :class:`INetworkDefinition` .

            This layer applies a per-channel softmax to its input.

            The output size is the same as the input size.

            :ivar axes: :class:`int` The axes along which softmax is computed. Currently, only one axis can be set. The axis is specified by setting the bit corresponding to the axis, after excluding the batch dimension, to 1. Let's say we have an NCHW tensor as input (three non-batch dimensions). Bit 0 corresponds to the C dimension boolean. Bit 1 corresponds to the H dimension boolean. Bit 2 corresponds to the W dimension boolean. For example, to perform softmax on axis R of a NPQRCHW input, set bit 2. By default, softmax is performed on the axis which is the number of non-batch axes minus three. It is 0 if there are fewer than 3 non-batch axes. For example, if the input is NCHW, the default axis is C. If the input is NHW, then the default axis is H.
        )trtdoc";
    } // ISoftMaxLayerDoc


    namespace IConcatenationLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A concatenation layer in an :class:`INetworkDefinition` .

            The output channel size is the sum of the channel sizes of the inputs.
            The other output sizes are the same as the other input sizes, which must all match.

            :ivar axis: :class:`int` The axis along which concatenation occurs. 0 is the major axis (excluding the batch dimension). The default is the number of non-batch axes in the tensor minus three (e.g. for an NCHW input it would be 0), or 0 if there are fewer than 3 non-batch axes.
        )trtdoc";
    } // IConcatenationLayerDoc

    namespace IDeconvolutionLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A deconvolution layer in an :class:`INetworkDefinition` .

            :ivar kernel_size: :class:`DimsHW` The HW kernel size of the convolution.
            :ivar num_output_maps: :class:`int` The number of output feature maps for the deconvolution.
            :ivar stride: :class:`DimsHW` The stride of the deconvolution. Default: (1, 1)
            :ivar padding: :class:`DimsHW` The padding of the deconvolution. The input will be zero-padded by this number of elements in the height and width directions. Padding is symmetric. Default: (0, 0)
            :ivar pre_padding: :class:`DimsHW` The pre-padding. The start of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
            :ivar post_padding: :class:`DimsHW` The post-padding. The end of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
            :ivar padding_mode: :class:`PaddingMode` The padding mode. Padding mode takes precedence if both :attr:`IDeconvolutionLayer.padding_mode` and either :attr:`IDeconvolutionLayer.pre_padding` or :attr:`IDeconvolutionLayer.post_padding` are set.
            :ivar num_groups: :class:`int` The number of groups for a deconvolution. The input tensor channels are divided into this many groups, and a deconvolution is executed for each group, using a filter per group. The results of the group convolutions are concatenated to form the output. **Note** When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output. Default: 1
            :ivar kernel: :class:`Weights` The kernel weights for the deconvolution. The weights are specified as a contiguous array in `CKRS` order, where `C` the number of input channels, `K` the number of output feature maps, and `R` and `S` are the height and width of the filter.
            :ivar bias: :class:`Weights` The bias weights for the deconvolution. Bias is optional. To omit bias, set this to an empty :class:`Weights` object. The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of output feature maps.
            :ivar kernel_size_nd: :class:`Dims` The multi-dimension kernel size of the convolution.
            :ivar stride_nd: :class:`Dims` The multi-dimension stride of the deconvolution. Default: (1, ..., 1)
            :ivar padding_nd: :class:`Dims` The multi-dimension padding of the deconvolution. The input will be zero-padded by this number of elements in each dimension. Padding is symmetric. Default: (0, ..., 0)
        )trtdoc";
    } // IDeconvolutionLayerDoc


    namespace ElementWiseOperationDoc
    {
        constexpr const char* descr = R"trtdoc(The binary operations that may be performed by an ElementWise layer.)trtdoc";

        constexpr const char* SUM = R"trtdoc(Sum of the two elements)trtdoc";
        constexpr const char* PROD = R"trtdoc(Product of the two elements)trtdoc";
        constexpr const char* MAX = R"trtdoc(Max of the two elements)trtdoc";
        constexpr const char* MIN = R"trtdoc(Min of the two elements)trtdoc";
        constexpr const char* SUB = R"trtdoc(Subtract the second element from the first)trtdoc";
        constexpr const char* DIV = R"trtdoc(Divide the first element by the second)trtdoc";
        constexpr const char* POW = R"trtdoc(The first element to the power of the second element)trtdoc";
        constexpr const char* FLOOR_DIV = R"trtdoc(Floor division of the first element by the second)trtdoc";
        constexpr const char* AND = R"trtdoc(Logical AND of two elements)trtdoc";
        constexpr const char* OR = R"trtdoc(Logical OR of two elements)trtdoc";
        constexpr const char* XOR = R"trtdoc(Logical XOR of two elements)trtdoc";
        constexpr const char* EQUAL = R"trtdoc(Check if two elements are equal)trtdoc";
        constexpr const char* GREATER = R"trtdoc(Check if element in first tensor is greater than corresponding element in second tensor)trtdoc";
        constexpr const char* LESS = R"trtdoc(Check if element in first tensor is less than corresponding element in second tensor)trtdoc";
    } // ElementWiseOperationDoc


    namespace IElementWiseLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A elementwise layer in an :class:`INetworkDefinition` .

            This layer applies a per-element binary operation between corresponding elements of two tensors.

            The input dimensions of the two input tensors must be equal, and the output tensor is the same size as each input.

            :ivar op: :class:`ElementWiseOperation` The binary operation for the layer.
        )trtdoc";
    } // IElementWiseLayerDoc

    namespace IGatherLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A gather layer in an :class:`INetworkDefinition` .

            :ivar axis: :class:`int` The non-batch dimension axis to gather on. The axis must be less than the number of non-batch dimensions in the data input.
            :ivar num_elementwise_dims: :class:`int` The number of leading dimensions of indices tensor to be handled elementwise. Must be 0 if there is an implicit batch dimension. It can be 0 or 1 if there is not an implicit batch dimension.
        )trtdoc";
    } // IGatherLayerDoc

    namespace RNNOperationDoc
    {
        constexpr const char* descr = R"trtdoc(
            The RNN operations that may be performed by an RNN layer.

            **Equation definitions**

            In the equations below, we use the following naming convention:

            |  `t` := current time step
            |  `i` := input gate
            |  `o` := output gate
            |  `f` := forget gate
            |  `z` := update gate
            |  `r` := reset gate
            |  `c` := cell gate
            |  `h` := hidden gate

            |  `g[t]` denotes the output of gate g at timestep `t`, e.g.`f[t]` is the output of the forget gate `f` .
            |  `X[t]` := input tensor for timestep `t`
            |  `C[t]` := cell state for timestep `t`
            |  `H[t]` := hidden state for timestep `t`

            |  `W[g]` := `W` (input) parameter weight matrix for gate `g`
            |  `R[g]` := `U` (recurrent) parameter weight matrix for gate `g`
            |  `Wb[g]` := `W` (input) parameter bias vector for gate `g`
            |  `Rb[g]` := `U` (recurrent) parameter bias vector for gate `g`

            Unless otherwise specified, all operations apply pointwise to elements of each operand tensor.

            |  `ReLU(X)` := `max(X, 0)`
            |  `tanh(X)` := hyperbolic tangent of `X`
            |  `sigmoid(X)` := `1 / (1 + exp(-X))`
            |  `exp(X)` := `e^X`
            |  `A.B` denotes matrix multiplication of `A` and `B` .
            |  `A*B` denotes pointwise multiplication of `A` and `B` .

            **Equations**

            Depending on the value of RNNOperation chosen, each sub-layer of the RNN layer will perform one of the following operations:

            **RELU**

            :math:`H[t] := ReLU(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])`

            **TANH**

            :math:`H[t] := tanh(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])`

            **LSTM**

            |  :math:`i[t] := sigmoid(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])`
            |  :math:`f[t] := sigmoid(W[f].X[t] + R[f].H[t-1] + Wb[f] + Rb[f])`
            |  :math:`o[t] := sigmoid(W[o].X[t] + R[o].H[t-1] + Wb[o] + Rb[o])`
            |  :math:`c[t] :=    tanh(W[c].X[t] + R[c].H[t-1] + Wb[c] + Rb[c])`


            |  :math:`C[t] := f[t]*C[t-1] + i[t]*c[t]`
            |  :math:`H[t] := o[t]*tanh(C[t])`

            **GRU**

            |  :math:`z[t] := sigmoid(W[z].X[t] + R[z].H[t-1] + Wb[z] + Rb[z])`
            |  :math:`r[t] := sigmoid(W[r].X[t] + R[r].H[t-1] + Wb[r] + Rb[r])`
            |  :math:`h[t] := tanh(W[h].X[t] + r[t]*(R[h].H[t-1] + Rb[h]) + Wb[h])`
            |  :math:`H[t] := (1 - z[t])*h[t] + z[t]*H[t-1]`
        )trtdoc";

        constexpr const char* RELU = R"trtdoc(Single gate RNN w/ ReLU activation)trtdoc";
        constexpr const char* TANH = R"trtdoc(Single gate RNN w/ TANH activation)trtdoc";
        constexpr const char* LSTM = R"trtdoc(Four-gate LSTM network w/o peephole connections)trtdoc";
        constexpr const char* GRU = R"trtdoc(Three-gate network consisting of Gated Recurrent Units)trtdoc";

    } // RNNOperationDoc

    namespace RNNDirectionDoc
    {
        constexpr const char* descr = R"trtdoc(The RNN direction that may be performed by an RNN layer.)trtdoc";

        constexpr const char* UNIDIRECTION = R"trtdoc(Network iterates from first input to last input)trtdoc";
        constexpr const char* BIDIRECTION = R"trtdoc(Network iterates from first to last (and vice versa) and outputs concatenated)trtdoc";
    } // RNNDirectionDoc

    namespace RNNInputModeDoc
    {
        constexpr const char* descr = R"trtdoc(
            The RNN input modes that may occur with an RNN layer.

             If the RNN is configured with :const:`RNNInputMode.LINEAR` , then for each gate `g` in the first layer of the RNN,
             the input vector `X[t]` (length `E`) is left-multiplied by the gate's corresponding weight matrix `W[g]`
             (dimensions `HxE`) as usual, before being used to compute the gate output as described by :class:`RNNOperation` .

             If the RNN is configured with :const:`RNNInputMode.SKIP` , then this initial matrix multiplication is "skipped"
             and `W[g]` is conceptually an identity matrix. In this case, the input vector `X[t]` must have length `H`
             (the size of the hidden state).
        )trtdoc";

        constexpr const char* LINEAR = R"trtdoc(Perform the normal matrix multiplication in the first recurrent layer)trtdoc";
        constexpr const char* SKIP = R"trtdoc(No operation is performed on the first recurrent layer)trtdoc";
    } // RNNInputModeDoc

    namespace IRNNLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            An RNN layer in an :class:`INetworkDefinition` .

            This layer applies an RNN operation on the inputs.

            **Deprecated** This interface is superseded by IRNNv2Layer.

            :ivar num_layers: :class:`int` The number of layers in the RNN.
            :ivar hidden_size: :class:`int` The size of the hidden layers.
            :ivar max_seq_length: :class:`int` The sequence length. This is the maximum number of input tensors that the RNN can process at once.
            :ivar op: :class:`RNNOperation` The operation of the RNN layer.
            :ivar input_mode: :class:`RNNInputMode` The input mode of the RNN layer.
            :ivar direction: :class:`RNNDirection` the direction of the RNN layer. The direction determines if the RNN is run as a unidirectional(left to right) or bidirectional(left to right and right to left). In the :const:`RNNDirection.BIDIRECTION` case the output is concatenated together, resulting in output size of 2x :attr:`hidden_size` .
            :ivar weights: :class:`Weights` The weight parameters for the RNN. For more information, see `IRNNLayer::setWeights() <https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_r_n_n_layer.html#a9333d560e68d49cbf60ba34d8872abf1>`_.
            :ivar bias: :class:`Weights` The bias parameter vector for the RNN layer. For more information see `IRNNLayer::setBias() <https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_r_n_n_layer.html#ae1a1d9599294c990d281a80ca684f1f0>`_.
            :ivar data_length: :class:`int` The length of the data being processed by the RNN for use in computing other values.
            :ivar hidden_state: :class:`ITensor` the initial hidden state of the RNN with the provided hidden ITensor.
                The layout for hidden is a linear layout of a 3D matrix:
                C - The number of layers in the RNN, it must match :attr:`num_layers` .
                H - The number of mini-batches for each time sequence.
                W - The size of the per layer hidden states, it must match :attr:`hidden_size` .
                The amount of space required is doubled if :attr:`direction` is :const:`RNNDirection.BIDIRECTION` with the bidirectional states coming after the unidirectional states.
                If not specified, then the initial hidden state is set to zero.

            :ivar cell_state: :class:`ITensor` the initial cell state of the RNN with the provided cell ITensor.
                The layout for cell is a linear layout of a 3D matrix:
                C - The number of layers in the RNN, it must match :attr:`num_layers` .
                H - The number of mini-batches for each time sequence.
                W - The size of the per layer hidden states, it must match :attr:`hidden_size` .
                The amount of space required is doubled if :attr:`direction` is :const:`RNNDirection.BIDIRECTION` with the bidirectional states coming after the unidirectional states.
                If not specified, then the initial cell state is set to zero.
                The cell state only affects LSTM RNN's.
        )trtdoc";
    } // IRNNLayerDoc

    namespace RNNGateTypeDoc
    {
        constexpr const char* descr = R"trtdoc(
            The RNN input modes that may occur with an RNN layer.

             If the RNN is configured with :const:`RNNInputMode.LINEAR` , then for each gate `g` in the first layer of the RNN,
             the input vector `X[t]` (length `E`) is left-multiplied by the gate's corresponding weight matrix `W[g]`
             (dimensions `HxE`) as usual, before being used to compute the gate output as described by :class:`RNNOperation` .

             If the RNN is configured with :const:`RNNInputMode.SKIP` , then this initial matrix multiplication is "skipped"
             and `W[g]` is conceptually an identity matrix. In this case, the input vector `X[t]` must have length `H`
             (the size of the hidden state).
        )trtdoc";

        constexpr const char* INPUT = R"trtdoc(Input Gate)trtdoc";
        constexpr const char* OUTPUT = R"trtdoc(Output Gate)trtdoc";
        constexpr const char* FORGET = R"trtdoc(Forget Gate)trtdoc";
        constexpr const char* UPDATE = R"trtdoc(Update Gate)trtdoc";
        constexpr const char* RESET = R"trtdoc(Reset Gate)trtdoc";
        constexpr const char* CELL = R"trtdoc(Cell Gate)trtdoc";
        constexpr const char* HIDDEN = R"trtdoc(Hidden Gate)trtdoc";
    } // RNNGateTypeDoc

    namespace IRNNv2LayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            An RNN layer in an :class:`INetworkDefinition` , version 2

            :ivar num_layers: :class:`int` The layer count of the RNN.
            :ivar hidden_size: :class:`int` The hidden size of the RNN.
            :ivar max_seq_length: :class:`int` The maximum sequence length of the RNN
            :ivar data_length: :class:`int` The length of the data being processed by the RNN for use in computing other values.

            :ivar seq_lengths: :class:`ITensor` Individual sequence lengths in the batch with the :class:`ITensor` provided.
                The :attr:`seq_lengths` :class:`ITensor` should be a {N1, ..., Np} tensor, where N1..Np are the index dimensions
                of the input tensor to the RNN.
                If :attr:`seq_lengths` is not specified, then the RNN layer assumes all sequences are size :attr:`max_seq_length` .
                All sequence lengths in :attr:`seq_lengths` should be in the range [1, :attr:`max_seq_length` ]. Zero-length sequences are not supported.
                This tensor must be of type int32.
            :ivar op: :class:`RNNOperation` The operation of the RNN layer.
            :ivar input_mode: :class:`int` The input mode of the RNN layer.
            :ivar direction: :class:`int` The direction of the RNN layer.

            :ivar hidden_state: :class:`ITensor` the initial hidden state of the RNN with the provided :attr:`hidden_state` :class:`ITensor` .
                The :attr:`hidden_state` :class:`ITensor` should have the dimensions `{N1, ..., Np, L, H}`, where:
                `N1..Np` are the index dimensions specified by the input tensor
                `L` is the number of layers in the RNN, equal to :attr:`num_layers`
                `H` is the hidden state for each layer, equal to :attr:`hidden_size` if :attr:`direction` is :const:`RNNDirection.UNIDIRECTION` , and 2x :attr:`hidden_size` otherwise.
            :ivar cell_state: :class:`ITensor` The initial cell state of the LSTM with the provided :attr:`cell_state` :class:`ITensor` .
                The :attr:`cell_state` :class:`ITensor` should have the dimensions `{N1, ..., Np, L, H}`, where:
                `N1..Np` are the index dimensions specified by the input tensor
                `L` is the number of layers in the RNN, equal to :attr:`num_layers`
                `H` is the hidden state for each layer, equal to :attr:`hidden_size` if :attr:`direction` is :const:`RNNDirection.UNIDIRECTION`, and 2x :attr:`hidden_size` otherwise.
                It is an error to set this on an RNN layer that is not configured with :const:`RNNOperation.LSTM` .
        )trtdoc";

        constexpr const char* set_weights_for_gate = R"trtdoc(
            Set the weight parameters for an individual gate in the RNN.

            :arg layer_index: The index of the layer that contains this gate. Refer to :attr:`IRNNLayer.weights` for a description of the layer index.
            :arg gate: The name of the gate within the RNN layer. The gate name must correspond to one of the gates used by this layer's :class:`RNNOperation` .
            :arg is_w: True if the weight parameters are for the input matrix W[g] and false if they are for the recurrent input matrix R[g]. See :class:`RNNOperation` for equations showing how these matrices are used in the RNN gate.
            :arg weights: The weight structure holding the weight parameters, which are stored as a row-major 2D matrix. Refer to :attr:`IRNNLayer.weights` for documentation on the expected dimensions of this matrix.
        )trtdoc";
        constexpr const char* get_weights_for_gate = R"trtdoc(
            Get the weight parameters for an individual gate in the RNN.

            :arg layer_index: The index of the layer that contains this gate.
            :arg gate: The name of the gate within the RNN layer.
            :arg is_w: True if the weight parameters are for the input matrix W[g] and false if they are for the recurrent input matrix R[g].

            :returns: The weight parameters.
        )trtdoc";
        constexpr const char* set_bias_for_gate = R"trtdoc(
            Set the bias parameters for an individual gate in the RNN.

            :arg layer_index: The index of the layer that contains this gate. Refer to :attr:`IRNNLayer.weights` for a description of the layer index.
            :arg gate: The name of the gate within the RNN layer. The gate name must correspond to one of the gates used by this layer's :class:`RNNOperation` .
            :arg is_w: True if the bias parameters are for the input bias Wb[g] and false if they are for the recurrent input bias Rb[g]. See
                    :class:`RNNOperation` for equations showing how these bias vectors are used in the RNN gate.
            :arg bias: The weight structure holding the bias parameters, which should be an array of size :attr:`hidden_size` .
        )trtdoc";
        constexpr const char* get_bias_for_gate = R"trtdoc(
            Get the bias parameters for an individual gate in the RNN.

            :arg layer_index: The index of the layer that contains this gate.
            :arg gate: The name of the gate within the RNN layer.
            :arg is_w: True if the bias parameters are for the input bias Wb[g] and false if they are for the recurrent input bias Rb[g].

            :returns: The bias parameters.
        )trtdoc";
    } // IRNNv2LayerDoc

    namespace IOutputDimensionsFormulaDoc
    {
        constexpr const char* descr = R"trtdoc(
            Application-implemented interface to compute layer output sizes.
        )trtdoc";

        constexpr const char* compute = R"trtdoc(
            Application-implemented interface to compute the HW output dimensions of a layer from the layer input and parameters.

            :arg input_shape: The input shape of the layer.
            :arg kernel_shape: The kernel shape (or window size, for a pooling layer) parameter of the layer operation.
            :arg stride: The stride parameter for the layer.
            :arg padding: The padding parameter of the layer.
            :arg dilation: The dilation parameter of the layer (only applicable to convolutions).
            :arg layer_name: The name of the layer.

            :returns: The output size of the layer
        )trtdoc";
    } // IOutputDimensionsFormulaDoc

    namespace IPluginLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
             A plugin layer in an :class:`INetworkDefinition` .

             :ivar plugin: :class:`IPlugin` The plugin for the layer.
        )trtdoc";
    } // IPluginLayerDoc

    namespace IPluginV2LayerDoc
    {
        constexpr const char* descr = R"trtdoc(
             A plugin layer in an :class:`INetworkDefinition` .

             :ivar plugin: :class:`IPluginV2` The plugin for the layer.
        )trtdoc";
    } // IPluginV2LayerDoc

    namespace UnaryOperationDoc
    {
        constexpr const char* descr = R"trtdoc(The unary operations that may be performed by a Unary layer.)trtdoc";

        constexpr const char* EXP = R"trtdoc(Exponentiation)trtdoc";
        constexpr const char* LOG = R"trtdoc(Log (base e))trtdoc";
        constexpr const char* SQRT = R"trtdoc(Square root)trtdoc";
        constexpr const char* RECIP = R"trtdoc(Reciprocal)trtdoc";
        constexpr const char* ABS = R"trtdoc(Absolute value)trtdoc";
        constexpr const char* NEG = R"trtdoc(Negation)trtdoc";
        constexpr const char* SIN = R"trtdoc(Sine)trtdoc";
        constexpr const char* COS = R"trtdoc(Cosine)trtdoc";
        constexpr const char* TAN = R"trtdoc(Tangent)trtdoc";
        constexpr const char* SINH = R"trtdoc(Hyperbolic sine)trtdoc";
        constexpr const char* COSH = R"trtdoc(Hyperbolic cosine)trtdoc";
        constexpr const char* ASIN = R"trtdoc(Inverse sine)trtdoc";
        constexpr const char* ACOS = R"trtdoc(Inverse cosine)trtdoc";
        constexpr const char* ATAN = R"trtdoc(Inverse tangent)trtdoc";
        constexpr const char* ASINH = R"trtdoc(Inverse hyperbolic sine)trtdoc";
        constexpr const char* ACOSH = R"trtdoc(Inverse hyperbolic cosine)trtdoc";
        constexpr const char* ATANH = R"trtdoc(Inverse hyperbolic tangent)trtdoc";
        constexpr const char* CEIL = R"trtdoc(Ceiling)trtdoc";
        constexpr const char* FLOOR = R"trtdoc(Floor)trtdoc";
        constexpr const char* ERF = R"trtdoc(Gauss error function)trtdoc";
        constexpr const char* NOT = R"trtdoc(Not)trtdoc";
    } // UnaryOperationDoc

    namespace IUnaryLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A unary layer in an :class:`INetworkDefinition` .

            :ivar op: :class:`UnaryOperation` The unary operation for the layer.
        )trtdoc";
    } // IUnaryLayerDoc

    namespace ReduceOperationDoc
    {
        constexpr const char* descr = R"trtdoc(The reduce operations that may be performed by a Reduce layer)trtdoc";

        constexpr const char* SUM = R"trtdoc()trtdoc";
        constexpr const char* PROD = R"trtdoc()trtdoc";
        constexpr const char* MAX = R"trtdoc()trtdoc";
        constexpr const char* MIN = R"trtdoc()trtdoc";
        constexpr const char* AVG = R"trtdoc()trtdoc";
    } // ReduceOperationDoc

    namespace IReduceLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A reduce layer in an :class:`INetworkDefinition` .

            :ivar op: :class:`ReduceOperation` The reduce operation for the layer.
            :ivar axes: :class:`int` The axes over which to reduce.
            :ivar keep_dims: :class:`bool` Specifies whether or not to keep the reduced dimensions for the layer.
        )trtdoc";
    } // IReduceLayerDoc

    namespace IPaddingLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A padding layer in an :class:`INetworkDefinition` .

            :ivar pre_padding: :class:`DimsHW` The padding that is applied at the start of the tensor. Negative padding results in trimming the edge by the specified amount.
            :ivar post_padding: :class:`DimsHW` The padding that is applied at the end of the tensor. Negative padding results in trimming the edge by the specified amount
            :ivar pre_padding_nd: :class:`Dims` The padding that is applied at the start of the tensor. Negative padding results in trimming the edge by the specified amount. Only 2 dimensions currently supported.
            :ivar post_padding_nd: :class:`Dims` The padding that is applied at the end of the tensor. Negative padding results in trimming the edge by the specified amount. Only 2 dimensions currently supported.
        )trtdoc";
    } // IPaddingLayerDoc

    namespace PermutationDoc
    {
        constexpr const char* descr = R"trtdoc(
            The elements of the permutation. The permutation is applied as outputDimensionIndex = permutation[inputDimensionIndex], so to permute from CHW order to HWC order, the required permutation is [1, 2, 0], and to permute from HWC to CHW, the required permutation is [2, 0, 1].

            It supports iteration and indexing and is implicitly convertible to/from Python iterables (like :class:`tuple` or :class:`list` ). Therefore, you can use those classes in place of :class:`Permutation` .
        )trtdoc";
    } // PermutationDoc

    namespace IShuffleLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A shuffle layer in an :class:`INetworkDefinition` .

            This class shuffles data by applying in sequence: a transpose operation, a reshape operation and a second transpose operation. The dimension types of the output are those of the reshape dimension.

            :ivar first_transpose: :class:`Permutation` The permutation applied by the first transpose operation. Default: Identity Permutation
            :ivar reshape_dims: :class:`Dims` The reshaped dimensions.
                Two special values can be used as dimensions.
                Value 0 copies the corresponding dimension from input. This special value can be used more than once in the dimensions. If number of reshape dimensions is less than input, 0s are resolved by aligning the most significant dimensions of input.
                Value -1 infers that particular dimension by looking at input and rest of the reshape dimensions. Note that only a maximum of one dimension is permitted to be specified as -1.
                The product of the new dimensions must be equal to the product of the old.
            :ivar second_transpose: :class:`Permutation` The permutation applied by the second transpose operation. Default: Identity Permutation
            :ivar zero_is_placeholder: :class:`bool` The meaning of 0 in reshape dimensions.
                If true, then a 0 in the reshape dimensions denotes copying the corresponding
                dimension from the first input tensor.  If false, then a 0 in the reshape
                dimensions denotes a zero-length dimension.
        )trtdoc";

        constexpr const char* set_input = R"trtdoc(
            Sets the input tensor for the given index. The index must be 0 for a static shuffle layer.
            A static shuffle layer is converted to a dynamic shuffle layer by calling :func:`set_input` with an index 1.
            A dynamic shuffle layer cannot be converted back to a static shuffle layer.

            For a dynamic shuffle layer, the values 0 and 1 are valid.
            The indices in the dynamic case are as follows:

            | Index | Description
            |   0   | Data or Shape tensor to be shuffled.
            |   1   | The dimensions for the reshape operation, as a 1D Int32 shape tensor.

            If this function is called with a value 1, then :attr:`num_inputs` changes
            from 1 to 2.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        )trtdoc";

    } // IShuffleLayerDoc

    namespace ISliceLayerDoc
    {
    constexpr const char* descr = R"trtdoc(
            A slice layer in an :class:`INetworkDefinition` .

            :ivar start: :class:`Dims` The start offset.
            :ivar shape: :class:`Dims` The output dimensions.
            :ivar stride: :class:`Dims` The slicing stride.
            :ivar mode: :class:`SliceMode` Controls how ISliceLayer handles out of bounds coordinates.
        )trtdoc";

    constexpr const char* set_input = R"trtdoc(
            Sets the input tensor for the given index. The index must be 0 for a static slice layer.
            A static slice layer is converted to a dynamic slice layer by calling :func:`set_input` with an index > 0.
            A dynamic slice layer cannot be converted back to a static slice layer.

            For a dynamic slice layer, the values 0-3 are valid. If an index > 0 is specified, all values between
            index 0 and that index must be dynamic tensors. The values larger than index can use static dimensions.
            For example, if an index of two is specified, the stride tensor can be set via setStride, but the start tensor
            must be specified via :func:`set_input` as both size and start are converted to dynamic tensors.
            The indices in the dynamic case are as follows:

            | Index | Description
            |   0   | Data or Shape tensor to be sliced.
            |   1   | The start tensor to begin slicing, N-dimensional for Data, and 1-D for Shape.
            |   2   | The size tensor of the resulting slice, N-dimensional for Data, and 1-D for Shape.
            |   3   | The stride of the slicing operation, N-dimensional for Data, and 1-D for Shape.

            If this function is called with a value greater than 0, then :attr:`num_inputs` changes
            from 1 to index + 1. When converting from static to dynamic slice layer,
            all unset tensors, between 1 and index + 1, are initialized to nullptr. It is an error to attempt to build
            a network that has any nullptr inputs.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        )trtdoc";


    } // ISliceLayerDoc

    namespace SliceModeDoc
    {
    constexpr const char* descr = R"trtdoc(Controls how ISliceLayer handles out of bounds coordinates)trtdoc";

    constexpr const char* DEFAULT
        = R"trtdoc(Fail with error when the coordinates are out of bounds. This is the default.)trtdoc";
    constexpr const char* WRAP = R"trtdoc(Coordinates wrap around periodically)trtdoc";
    } // SliceModeDoc

    namespace IShapeLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A shape layer in an :class:`INetworkDefinition` . Used for getting the shape of a tensor.
            This class sets the output to a one-dimensional tensor with the dimensions of the input tensor.

            For example, if the input is a four-dimensional tensor (of any type) with
            dimensions [2,3,5,7], the output tensor is a one-dimensional Int32 tensor
            of length 4 containing the sequence 2, 3, 5, 7.
        )trtdoc";

    } // IShapeLayerDoc

    namespace TopKOperationDoc
    {
        constexpr const char* descr = R"trtdoc(The operations that may be performed by a TopK layer)trtdoc";

        constexpr const char* MAX = R"trtdoc(Maximum of the elements)trtdoc";
        constexpr const char* MIN = R"trtdoc(Minimum of the elements)trtdoc";
    } // TopKOperationDoc

    namespace ITopKLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A TopK layer in an :class:`INetworkDefinition` .

            :ivar op: :class:`TopKOperation` The operation for the layer.
            :ivar k: :class:`TopKOperation` the k value for the layer. Currently only values up to 25 are supported.
            :ivar axes: :class:`TopKOperation` The axes along which to reduce.
        )trtdoc";
    } // ITopKLayerDoc

    namespace MatrixOperationDoc
    {
        constexpr const char* descr = R"trtdoc(The matrix operations that may be performed by a Matrix layer)trtdoc";

        constexpr const char* NONE = R"trtdoc()trtdoc";
        constexpr const char* TRANSPOSE = R"trtdoc(Transpose each matrix)trtdoc";
        constexpr const char* VECTOR = R"trtdoc(Treat operand as collection of vectors)trtdoc";
    } // MatrixOperationDoc

    namespace IMatrixMultiplyLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A matrix multiply layer in an :class:`INetworkDefinition` .

            Let A be op(getInput(0)) and B be op(getInput(1)) where
            op(x) denotes the corresponding MatrixOperation.

            When A and B are matrices or vectors, computes the inner product A * B:

            |   matrix * matrix -> matrix
            |   matrix * vector -> vector
            |   vector * matrix -> vector
            |   vector * vector -> scalar

            Inputs of higher rank are treated as collections of matrices or vectors.
            The output will be a corresponding collection of matrices, vectors, or scalars.

            :ivar op0: :class:`MatrixOperation` How to treat the first input.
            :ivar op1: :class:`MatrixOperation` How to treat the second input.
        )trtdoc";
    } // IMatrixMultiplyLayerDoc

    namespace IRaggedSoftMaxLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A ragged softmax layer in an :class:`INetworkDefinition` .

            This layer takes a ZxS input tensor and an additional Zx1 bounds tensor holding the lengths of the Z sequences.

            This layer computes a softmax across each of the Z sequences.

            The output tensor is of the same size as the input tensor.
        )trtdoc";
    } // IRaggedSoftMaxLayerDoc

    namespace IIdentityLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A layer that represents the identity function.

            If tensor precision is explicitly specified, it can be used to transform from one precision to another.
        )trtdoc";
    } // IIdentityLayerDoc

    namespace IConstantLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A constant layer in an :class:`INetworkDefinition` .

            Note: This layer does not support boolean types.

            :ivar weights: :class:`Weights` The weights for the layer.
            :ivar shape: :class:`Dims` The shape of the layer.
        )trtdoc";
    } // IConstantLayerDoc

    namespace IParametricReLULayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A parametric ReLU layer in an :class:`INetworkDefinition` .

            This layer applies a parametric ReLU activation to an input tensor (first input), with slopes taken from a
            slopes tensor (second input). This can be viewed as a leaky ReLU operation where the negative slope differs
            from element to element (and can in fact be learned).

            The slopes tensor must be unidirectional broadcastable to the input tensor: the rank of the two tensors must
            be the same, and all dimensions of the slopes tensor must either equal the input tensor or be 1.
            The output tensor has the same shape as the input tensor.
        )trtdoc";
    } // IParametricReLULayerDoc

    namespace ResizeModeDoc
    {
        constexpr const char* descr = R"trtdoc(Various modes of resize in the resize layer.)trtdoc";

        constexpr const char* NEAREST = R"trtdoc(1D, 2D, and 3D nearest neighbor resizing.)trtdoc";
        constexpr const char* LINEAR = R"trtdoc(Can handle linear, bilinear, trilinear resizing.)trtdoc";
    } /* ResizeModeDoc */

    namespace IResizeLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A resize layer in an :class:`INetworkDefinition` .

            Resize layer can be used for resizing a N-D tensor.

            Resize layer currently supports the following configurations:

            * ResizeMode.NEAREST - resizes innermost `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0.
            * ResizeMode.LINEAR - resizes innermost `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0.

            Default resize mode is ResizeMode.NEAREST.

            Resize layer provides two ways to resize tensor dimensions:

            * Set output dimensions directly. It can be done for static as well as dynamic resize layer.
              Static resize layer requires output dimensions to be known at build-time.
              Dynamic resize layer requires output dimensions to be set as one of the input tensors.

            * Set scales for resize. Each output dimension is calculated as floor(input dimension * scale).
              Only static resize layer allows setting scales where the scales are known at build-time.

            :ivar shape: :class:`Dims` The output dimensions. Must to equal to input dimensions size.
            :ivar scales: :class:`List[float]` List of resize scales.
            :ivar resize_mode: :class:`ResizeMode` Resize mode can be Linear or Nearest.
            :ivar align_corners: :class:`bool` If True, the centers of the 4 corner pixels of both input and output tensors are aligned. Default: False.
        )trtdoc";

        constexpr const char* set_input = R"trtdoc(
            Sets the input tensor for the given index.

            If index == 1 and num_inputs == 1, and there is no implicit batch dimension,
            in which case num_inputs changes to 2.
            Once such additional input is set, resize layer works in dynamic mode.
            When index == 1 and num_inputs == 1, the output dimensions are used from
            the input tensor, overriding the dimensions supplied by `shape`.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        )trtdoc";
    } /* IResizeLayerDoc */


    namespace LoopOutputDoc {
        constexpr const char* descr = R"trtdoc(Describes kinds of loop outputs.)trtdoc";

        constexpr const char* LAST_VALUE = R"trtdoc(Output value is value of tensor for last iteration.)trtdoc";
        constexpr const char* CONCATENATE = R"trtdoc(Output value is concatenation of values of tensor for each iteration, in forward order.)trtdoc";
        constexpr const char* REVERSE = R"trtdoc(Output value is concatenation of values of tensor for each iteration, in reverse order.)trtdoc";
    } /* LoopOutputDoc */


    namespace TripLimitDoc {
        constexpr const char* descr = R"trtdoc(Describes kinds of trip limits.)trtdoc";

        constexpr const char* COUNT = R"trtdoc(Tensor is scalar of type kINT32 that contains the trip count.)trtdoc";
        constexpr const char* WHILE = R"trtdoc(Tensor is a scalar of type BOOL. Loop terminates when value is false.)trtdoc";

    } /* TripLimitDoc */


    namespace ILoopBoundaryLayerDoc {
        constexpr const char* descr = R"trtdoc(
            :ivar loop: :class:`ILoop` associated with this boundary layer.
        )trtdoc";

    } /* ILoopBoundaryLayerDoc */


    namespace IRecurrenceLayerDoc {
        constexpr const char* descr = R"trtdoc()trtdoc";
        constexpr const char* set_input = R"trtdoc(
            Set the first or second input.
            If index==1 and the number of inputs is one, the input is appended.
            The first input specifies the initial output value, and must come from outside the loop.
            The second input specifies the next output value, and must come from inside the loop.
            The two inputs must have the same dimensions.

            :param index: The index of the input to set.
            :param tensor: The input tensor.
        )trtdoc";
    } /* IRecurrenceLayerDoc */


    namespace ILoopOutputLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            An :class:`ILoopOutputLayer` is the sole way to get output from a loop.

            The first input tensor must be defined inside the loop; the output tensor is outside the loop.
            The second input tensor, if present, must be defined outside the loop.

            If :attr:`kind` is ``LAST_VALUE``, a single input must be provided.

            If :attr:`kind` is ``CONCATENATE`` or ``REVERSE``, a second input must be provided.
            The second input must be a scalar “shape tensor”, defined before the loop commences,
            that specifies the concatenation length of the output.

            The output tensor has j more dimensions than the input tensor, where
            j == 0 if :attr:`kind` is ``LAST_VALUE``
            j == 1 if :attr:`kind` is ``CONCATENATE`` or ``REVERSE``.

            :ivar axis: The contenation axis. Ignored if :attr:`kind` is ``LAST_VALUE``.
                For example, if the input tensor has dimensions [b,c,d],
                and :attr:`kind` is  ``CONCATENATE``, the output has four dimensions.
                Let a be the value of the second input.
                axis=0 causes the output to have dimensions [a,b,c,d].
                axis=1 causes the output to have dimensions [b,a,c,d].
                axis=2 causes the output to have dimensions [b,c,a,d].
                axis=3 causes the output to have dimensions [b,c,d,a].
                Default is axis is 0.
            :ivar kind: The kind of loop output. See :class:`LoopOutput`
        )trtdoc";

        constexpr const char* set_input = R"trtdoc(
            Like :func:`ILayer.set_input`, but additionally works if index==1, :attr:`num_inputs`==1, in which case :attr:`num_inputs` changes to 2.
        )trtdoc";


    } /* ILoopOutputLayerDoc */


    namespace ITripLimitLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            :ivar kind: The kind of trip limit. See :class:`TripLimit`
        )trtdoc";
    } /* ITripLimitLayerDoc */


    namespace IIteratorLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            :ivar axis: The axis to iterate over
            :ivar reverse: For reverse=false, the layer is equivalent to add_gather(tensor, I, 0) where I is a
                scalar tensor containing the loop iteration number.
                For reverse=true, the layer is equivalent to add_gather(tensor, M-1-I, 0) where M is the trip count
                computed from TripLimits of kind ``COUNT``.
                The default is reverse=false.
        )trtdoc";
    } /* IIteratorLayerDoc */


    namespace ILoopDoc
    {
        constexpr const char* descr = R"trtdoc(
            Helper for creating a recurrent subgraph.

            :ivar name: The name of the loop. The name is used in error diagnostics.
        )trtdoc";

        constexpr const char* add_recurrence = R"trtdoc(
            Create a recurrence layer for this loop with initial_value as its first input.

            :param initial_value: The initial value of the recurrence layer.

            :returns: The added :class:`IRecurrenceLayer` , or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_trip_limit = R"trtdoc(
            Add a trip-count limiter, based on the given tensor.

            There may be at most one ``COUNT`` and one ``WHILE`` limiter for a loop.
            When both trip limits exist, the loop exits when the
            count is reached or condition is falsified.
            It is an error to not add at least one trip limiter.

            For ``WHILE``, the input tensor must be the output of a subgraph that contains
            only layers that are not :class:`ITripLimitLayer` , :class:`IIteratorLayer` or :class:`ILoopOutputLayer` .
            Any :class:`IRecurrenceLayer` s in the subgraph must belong to the same loop as the
            :class:`ITripLimitLayer` . A trivial example of this rule is that the input to the ``WHILE``
            is the output of an :class:`IRecurrenceLayer` for the same loop.


            :param tensor: The input tensor. Must be available before the loop starts.
            :param kind: The kind of trip limit. See :class:`TripLimit`

            :returns: The added :class:`ITripLimitLayer` , or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_iterator = R"trtdoc(
            Return layer that subscripts tensor by loop iteration.

            For reverse=false, this is equivalent to add_gather(tensor, I, 0) where I is a
            scalar tensor containing the loop iteration number.
            For reverse=true, this is equivalent to add_gather(tensor, M-1-I, 0) where M is the trip count
            computed from TripLimits of kind ``COUNT``.

            :param tensor: The tensor to iterate over.
            :param axis: The axis along which to iterate.
            :param reverse: Whether to iterate in the reverse direction.

            :returns: The :class:`IIteratorLayer` , or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_loop_output = R"trtdoc(
            Make an output for this loop, based on the given tensor.

            If ``kind`` is ``CONCATENATE`` or ``REVERSE``, a second input specifying the
            concatenation dimension must be added via method :func:`ILoopOutputLayer.set_input` .

            :param kind: The kind of loop output. See :class:`LoopOutput`
            :param axis: The axis for concatenation (if using ``kind`` of ``CONCATENATE`` or ``REVERSE``).

            :returns: The added :class:`ILoopOutputLayer` , or :class:`None` if it could not be created.
        )trtdoc";

    } /* ILoopDoc */


    namespace ISelectLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A select layer in an :class:`INetworkDefinition` .

            This layer implements an element-wise ternary conditional operation. Wherever ``condition`` is ``True``, elements are taken from the first input, and wherever ``condition`` is ``False``, elements are taken from the second input.
        )trtdoc";
    } /* ISelectLayerDoc */

    namespace FillOperationDoc
    {
        constexpr const char* descr = R"trtdoc(The tensor fill operations that may performed by an Fill layer.)trtdoc";

        constexpr const char* LINSPACE = R"trtdoc(Generate evenly spaced numbers over a specified interval)trtdoc";
        constexpr const char* RANDOM_UNIFORM = R"trtdoc(Generate a tensor with random values drawn from a uniform distribution)trtdoc";
    } /* FillOperationDoc */

    namespace IFillLayerDoc
    {
        constexpr const char* descr = R"trtdoc(
            A fill layer in an :class:`INetworkDefinition` .
        )trtdoc";

        constexpr const char* set_dimensions = R"trtdoc(
            set the output tensor's dimensions.

            :arg dims: the output tensor's dimensions.
        )trtdoc";

        constexpr const char* get_dimensions = R"trtdoc(
            get the output tensor's dimensions.
        )trtdoc";

        constexpr const char* set_operation = R"trtdoc(
            set the fill operation for the layer.

            :arg operation: the fill operation for the layer.
        )trtdoc";

        constexpr const char* get_operation = R"trtdoc(
            get the fill operation for the layer.
        )trtdoc";

        constexpr const char* set_alpha = R"trtdoc(
            set the alpha parameter (must be finite).

            Operation          | Usage
            kLINSPACE          | the start value;
            kRANDOMUNIFORM     | the minimum value;

            :arg alpha: has different meanings for each operators.
        )trtdoc";

        constexpr const char* get_alpha = R"trtdoc(
            get the alpha parameter.
            see :meth:`IFillLayer.set_alpha()` for details
        )trtdoc";

        constexpr const char* set_beta = R"trtdoc(
            set the beta parameter (must be finite).

            Operation          | Usage
            kLINSPACE          | the delta value;
            kRANDOMUNIFORM     | the maximal value;

            :arg beta: has different meanings for each operators.
        )trtdoc";

        constexpr const char* get_beta = R"trtdoc(
            get the beta parameter.
            see :meth:`IFillLayer.set_beta()` for details
        )trtdoc";

        constexpr const char* set_input = R"trtdoc(
            replace an input of this layer with a specific tensor.

            Index | Description for kLINSPACE
              0   | Shape tensor, represents the output tensor's dimensions.
              1   | Start, a scalar, represents the start value.
              2   | Delta, a 1D tensor, length equals to shape tensor's nbDims, represents the delta value for each dimension.

            Index | Description for kRANDOM_UNIFORM
              0   | Shape tensor, represents the output tensor's dimensions.
              1   | Minimum, a scalar, represents the minimum random value.
              2   | Maximum, a scalar, represents the maximal random value.

            :arg index: the index of the input to modify.
            :arg tensor: the input tensor.
        )trtdoc";
    } /* IFillLayerDoc */

    namespace INetworkDefinitionDoc
    {
        constexpr const char* descr = R"trtdoc(
            Represents a TensorRT Network from which the Builder can build an Engine

            :ivar pooling_output_dimensions_formula: :class:`IOutputDimensionsFormula` The formula from computing the pooling output dimensions. If set to :class:`None` , the default formula is used. The default formula in each dimension is :math:`(inputDim + padding * 2 - kernelSize) / stride + 1` .

            :ivar convolution_output_dimensions_formula: :class:`IOutputDimensionsFormula` **Deprecated** Does not currently work reliably and will be removed in a future release. The formula from computing the convolution output dimensions. If set to :class:`None` , the default formula is used. The default formula in each dimension is :math:`(inputDim + padding * 2 - kernelSize) / stride + 1` .

            :ivar deconvolution_output_dimensions_formula: :class:`IOutputDimensionsFormula` **Deprecated** Does not currently work reliably and will be removed in a future release. The formula from computing the deconvolution output dimensions. If :class:`None` is passed, the default formula is used. The default formula in each dimension is :math:`(inputDim - 1) * stride + kernelSize - 2 * padding` .

            :ivar num_layers: :class:`int` The number of layers in the network.
            :ivar num_inputs: :class:`int` The number of inputs of the network.
            :ivar num_outputs: :class:`int` The number of outputs of the network.
            :ivar name: :class:`str` The name of the network. This is used so that it can be associated with a built engine. The name must be at most 128 characters in length. TensorRT makes no use of this string except storing it as part of the engine so that it may be retrieved at runtime. A name unique to the builder will be generated by default.
            :ivar has_implicit_batch_dimension: :class:`bool` Whether the network was created with an implicit batch dimension. This is a network-wide property. Either all tensors in the network have an implicit batch dimension or none of them do. This is True when the INetworkDefinition is created with default flags: ``create_network()``. To specify explicit batch, set the flag: ``create_network(flags=1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))``.
            :ivar has_explicit_precision: :class:`bool` True if and only if this :class:`INetworkDefinition` was created with ``NetworkDefinitionCreationFlag.EXPLICIT_PRECISION`` set: ``create_network(flags=(1 << int(NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)))``.
        )trtdoc";

        constexpr const char* add_input = R"trtdoc(
            Adds an input to the network.

            :arg name: The name of the tensor.
            :arg dtype: The data type of the tensor. Currently, trt.int8 is not supported for inputs.
            :arg shape: The dimensions of the tensor. The total volume must be less than 2^30 elements.

            :returns: The newly added Tensor.
        )trtdoc";

        constexpr const char* mark_output = R"trtdoc(
            Mark a tensor as an output.

            :arg tensor: The tensor to mark.
        )trtdoc";

        constexpr const char* add_convolution = R"trtdoc(
            Add a 2D convolution layer to the network.
            See :class:`IConvolutionLayer` for more information.

            :arg input: The input tensor to the convolution.
            :arg num_output_maps: The number of output feature maps for the convolution.
            :arg kernel_shape: The dimensions of the convolution kernel.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new convolution layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_convolution_nd = R"trtdoc(
            Add a multi-dimension convolution layer to the network.
            See :class:`IConvolutionLayer` for more information.

            :arg input: The input tensor to the convolution.
            :arg num_output_maps: The number of output feature maps for the convolution.
            :arg kernel_shape: The dimensions of the convolution kernel.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new convolution layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_fully_connected = R"trtdoc(
            Add a fully connected layer to the network.
            See :class:`IFullyConnectedLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg num_outputs: The number of outputs of the layer.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new fully connected layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_activation = R"trtdoc(
            Add an activation layer to the network.
            See :class:`IActivationLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg type: The type of activation function to apply.

            :returns: The new activation layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_pooling = R"trtdoc(
            Add a 2D pooling layer to the network.
            See :class:`IPoolingLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg type: The type of pooling to apply.
            :arg window_size: The size of the pooling window.

            :returns: The new pooling layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_pooling_nd = R"trtdoc(
            Add a multi-dimension pooling layer to the network.
            See :class:`IPoolingLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg type: The type of pooling to apply.
            :arg window_size: The size of the pooling window.

            :returns: The new pooling layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_lrn = R"trtdoc(
            Add a LRN layer to the network.
            See :class:`ILRNLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg window: The size of the window.
            :arg alpha: The alpha value for the LRN computation.
            :arg beta: The beta value for the LRN computation.
            :arg k: The k value for the LRN computation.

            :returns: The new LRN layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_scale = R"trtdoc(
            Add a scale layer to the network.
            See :class:`IScaleLayer` for more information.

            :arg input: The input tensor to the layer. This tensor is required to have a minimum of 3 dimensions.
            :arg mode: The scaling mode.
            :arg shift: The shift value.
            :arg scale: The scale value.
            :arg power: The power value.

            If the weights are available, then the size of weights are dependent on the ScaleMode.
            For UNIFORM, the number of weights is equal to 1.
            For CHANNEL, the number of weights is equal to the channel dimension.
            For ELEMENTWISE, the number of weights is equal to the volume of the input.

            :returns: The new scale layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_scale_nd = R"trtdoc(
            Add a multi-dimension scale layer to the network.
            See :class:`IScaleLayer` for more information.

            :arg input: The input tensor to the layer. This tensor is required to have a minimum of 3 dimensions.
            :arg mode: The scaling mode.
            :arg shift: The shift value.
            :arg scale: The scale value.
            :arg power: The power value.
            :arg channel_axis: The channel dimension axis.

            If the weights are available, then the size of weights are dependent on the ScaleMode.
            For UNIFORM, the number of weights is equal to 1.
            For CHANNEL, the number of weights is equal to the channel dimension.
            For ELEMENTWISE, the number of weights is equal to the volume of the input.

            :returns: The new scale layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_softmax = R"trtdoc(
            Add a softmax layer to the network.
            See :class:`ISoftMaxLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new softmax layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_concatenation = R"trtdoc(
            Add a concatenation layer to the network. Note that all tensors must have the same dimension except for the Channel dimension.
            See :class:`IConcatenationLayer` for more information.

            :arg inputs: The input tensors to the layer.

            :returns: The new concatenation layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_deconvolution = R"trtdoc(
            Add a 2D deconvolution layer to the network.
            See :class:`IDeconvolutionLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg num_output_maps: The number of output feature maps.
            :arg kernel_shape: The dimensions of the convolution kernel.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new deconvolution layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_deconvolution_nd = R"trtdoc(
            Add a multi-dimension deconvolution layer to the network.
            See :class:`IDeconvolutionLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg num_output_maps: The number of output feature maps.
            :arg kernel_shape: The dimensions of the convolution kernel.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new deconvolution layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_elementwise = R"trtdoc(
            Add an elementwise layer to the network.
            See :class:`IElementWiseLayer` for more information.

            :arg input1: The first input tensor to the layer.
            :arg input2: The second input tensor to the layer.
            :arg op: The binary operation that the layer applies.

            The input tensors must have the same number of dimensions.
            For each dimension, their lengths must match, or one of them must be one.
            In the latter case, the tensor is broadcast along that axis.

            The output tensor has the same number of dimensions as the inputs.
            For each dimension, its length is the maximum of the lengths of the
            corresponding input dimension.

            :returns: The new element-wise layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_rnn = R"trtdoc(
            Add a ``layer_count`` deep RNN layer to the network with a sequence length of ``max_seq_length`` and ``hidden_size`` internal state per layer.
            See :class:`IRNNLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg layer_count: The number of layers in the RNN.
            :arg hidden_size: The size of the internal hidden state for each layer.
            :arg max_seq_length: The maximum length of the time sequence.
            :arg op: The type of RNN to execute.
            :arg mode: The input mode for the RNN.
            :arg direction: The direction to run the RNN.
            :arg weights: The weights for the weight matrix parameters of the RNN.
            :arg bias: The weights for the bias vectors parameters of the RNN.

            The input tensors must be of the type :const:`float32` or :const:`float16` .

            See :class:`IRNNLayer` for details on the required input format for ``weights`` and ``bias`` .

            The layout for the ``input`` tensor should be `{1, S_max, N, E}`, where:

            |  `S_max` is the maximum allowed sequence length (number of RNN iterations)
            |  `N` is the batch size
            |  `E` specifies the embedding length (unless :const:`RNNInputMode.SKIP` is set, in which case it should match :attr:`hidden_size` ).

            The first output tensor is the output of the final RNN layer across all timesteps, with dimensions `{S_max, N, H}`:

            |  `S_max` is the maximum allowed sequence length (number of RNN iterations)
            |  `N` is the batch size
            |  `H` is an output hidden state (equal to :attr:`hidden_size` or 2x :attr:`hidden_size` )

            The second tensor is the final hidden state of the RNN across all layers, and if the RNN is an LSTM (i.e. :attr:`op` is :const:`RNNOperation.LSTM` ), then the third tensor is the final cell state of the RNN across all layers.  Both the second and third output tensors have dimensions `{L, N, H}`:

            |  `L` is equal to  :attr:`num_layers` if getDirection is :const:`RNNDirection.UNIDIRECTION` , and 2* :attr:`num_layers` if getDirection is :const:`RNNDirection.BIDIRECTION` .  In the bi-directional case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and final backward hidden state is stored in `L = 2*l + 1` .
            |  `N` is the batch size
            |  `H` is  :attr:`hidden_size` .

            Note that in bidirectional RNNs, the full "hidden state" for a layer `l` is the concatenation of its forward hidden state and its backward hidden state, and its size is 2*H.

            **Deprecated** IRNNLayer is superseded by IRNNv2Layer. Use add_rnn_v2() instead.

            :returns: The new RNN layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_plugin = R"trtdoc(
            Add a plugin layer to the network.
            See :class:`IPlugin` for more information.

            :arg inputs: The input tensors to the layer.
            :arg plugin: The layer plugin.

            :returns: The new plugin layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_unary = R"trtdoc(
            Add a unary layer to the network.
            See :class:`IUnaryLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg op: The operation to apply.

            :returns: The new unary layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_padding = R"trtdoc(
            Add a 2D padding layer to the network.
            See :class:`IPaddingLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg pre_padding: The padding to apply to the start of the tensor.
            :arg post_padding: The padding to apply to the end of the tensor.

            :returns: The new padding layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_padding_nd = R"trtdoc(
            Add a multi-dimensional padding layer to the network.
            See :class:`IPaddingLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg pre_padding: The padding to apply to the start of the tensor.
            :arg post_padding: The padding to apply to the end of the tensor.

            :returns: The new padding layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_shuffle = R"trtdoc(
            Add a shuffle layer to the network.
            See :class:`IShuffleLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new shuffle layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_slice = R"trtdoc(
            Add a slice layer to the network.
            See :class:`ISliceLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg start: The start offset.
            :arg shape: The output shape.
            :arg stride: The slicing stride. Positive, negative, zero stride values, and combinations of them in different dimensions are allowed.

            :returns: The new slice layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_reduce = R"trtdoc(
            Add a reduce layer to the network.
            See :class:`IReduceLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg op: The reduction operation to perform.
            :arg axes: The reduction dimensions.

                |  Bit 0 of the uint32_t type corresponds to the non-batch dimension 0 boolean and so on.
                |  If a bit is set, then the corresponding dimension will be reduced.
                |  Let's say we have an NCHW tensor as input (three non-batch dimensions).
                |  Bit 0 corresponds to the C dimension boolean.
                |  Bit 1 corresponds to the H dimension boolean.
                |  Bit 2 corresponds to the W dimension boolean.
                |  Note that reduction is not permitted over the batch size dimension.
            :arg keep_dims: The boolean that specifies whether or not to keep the reduced dimensions in the output of the layer.

            :returns: The new reduce layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_topk = R"trtdoc(
            Add a TopK layer to the network.
            See :class:`ITopKLayer` for more information.

            The TopK layer has two outputs of the same dimensions. The first contains data values, the second contains index positions for the values. Output values are sorted, largest first for operation :const:`TopKOperation.MAX` and smallest first for operation :const:`TopKOperation.MIN` .

            Currently only values of K up to 1024 are supported.

            :arg input: The input tensor to the layer.
            :arg op: Operation to perform.
            :arg k: Number of elements to keep.

            :arg axes: The reduction dimensions.
               Bit 0 of the uint32_t type corresponds to the non-batch dimension 0 boolean and so on.
               If a bit is set, then the corresponding dimension will be reduced.
               Let's say we have an NCHW tensor as input (three non-batch dimensions).
               Bit 0 corresponds to the C dimension boolean.
               Bit 1 corresponds to the H dimension boolean.
               Bit 2 corresponds to the W dimension boolean.
               Note that TopK reduction is currently only permitted over one dimension.

            :returns: The new TopK layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_gather = R"trtdoc(
            Add a pooling layer to the network.
            See :class:`IGatherLayer` for more information.

            :arg input: The tensor to gather values from.
            :arg indices: The tensor to get indices from to populate the output tensor.
            :arg axis: The non-batch dimension axis in the data tensor to gather on.

            :returns: The new pooling layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_ragged_softmax = R"trtdoc(
            Add a ragged softmax layer to the network.
            See :class:`IRaggedSoftMaxLayer` for more information.

            :arg input: The ZxS input tensor.
            :arg bounds: The Zx1 bounds tensor.

            :returns: The new ragged softmax layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_matrix_multiply = R"trtdoc(
            Add a matrix multiply layer to the network.
            See :class:`IMatrixMultiplyLayer` for more information.

            :arg input0: The first input tensor (commonly A).
            :arg op0: Whether to treat input0 as matrices, transposed matrices, or vectors.
            :arg input1: The second input tensor (commonly B).
            :arg op1:  Whether to treat input1 as matrices, transposed matrices, or vectors.

            :returns: The new matrix multiply layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_matrix_multiply_deprecated = R"trtdoc(
            Add a matrix multiply layer to the network.
            See :class:`IMatrixMultiplyLayer` for more information.

            :arg input0: The first input tensor (commonly A).
            :arg transpose0: If true, op(input0)=transpose(input0), else op(input0)=input0.
            :arg input1: The second input tensor (commonly B).
            :arg transpose1: If true, op(input1)=transpose(input1), else op(input1)=input1.

            :returns: The new matrix multiply layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_constant = R"trtdoc(
            Add a constant layer to the network.
            See :class:`IConstantLayer` for more information.

            :arg shape: The shape of the constant.
            :arg weights: The constant value, represented as weights.

            :returns: The new constant layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_rnn_v2 = R"trtdoc(
            Add an RNNv2 layer to the network.
            See :class:`IRNNv2Layer` for more information.

            Add an ``layer_count`` deep RNN layer to the network with ``hidden_size`` internal states that can take a batch with fixed or variable sequence lengths.

            :arg input: The input tensor to the layer (see below).
            :arg layer_count: The number of layers in the RNN.
            :arg hidden_size: Size of the internal hidden state for each layer.
            :arg max_seq_length: Maximum sequence length for the input.
            :arg op: The type of RNN to execute.

            By default, the layer is configured with :const:`RNNDirection.UNIDIRECTION` and :const:`RNNInputMode.LINEAR` . To change these settings, set :attr:`IRNNv2Layer.direction` and :attr:`IRNNv2Layer.input_mode` .

            Weights and biases for the added layer should be set using :meth:`IRNNv2Layer.set_weights_for_gate()` and :meth:`IRNNv2Layer.set_bias_for_gate()` prior to building an engine using this network.

            The input tensors must be of the type :const:`float32` or :const:`float16` .
            The layout of the weights is row major and must be the same datatype as the input tensor.
            ``weights`` contain 8 matrices and ``bias`` contains 8 vectors.

            See :meth:`IRNNv2Layer.set_weights_for_gate()` and :meth:`IRNNv2Layer.set_bias_for_gate()` for details on the required input format for ``weights`` and ``bias`` .

            The ``input`` ITensor should contain zero or more index dimensions `{N1, ..., Np}`, followed by two dimensions, defined as follows:

            |  `S_max` is the maximum allowed sequence length (number of RNN iterations)
            |  `E` specifies the embedding length (unless :const:`RNNInputMode.SKIP` is set, in which case it should match :attr:`IRNNv2Layer.hidden_size` ).

            By default, all sequences in the input are assumed to be size ``max_seq_length`` .  To provide explicit sequence lengths for each input sequence in the batch, set :attr:`IRNNv2Layer.seq_lengths` .

            The RNN layer outputs up to three tensors.

            The first output tensor is the output of the final RNN layer across all timesteps, with dimensions `{N1, ..., Np, S_max, H}`:

            |  `N1..Np` are the index dimensions specified by the input tensor
            |  `S_max` is the maximum allowed sequence length (number of RNN iterations)
            |  `H` is an output hidden state (equal to :attr:`IRNNv2Layer.hidden_size` or 2x :attr:`IRNNv2Layer.hidden_size` )

            The second tensor is the final hidden state of the RNN across all layers, and if the RNN is an LSTM (i.e. :attr:`IRNNv2Layer.op` is :const:`RNNOperation.LSTM` ), then the third tensor is the final cell state of the RNN across all layers.  Both the second and third output tensors have dimensions `{N1, ..., Np, L, H}`:

            |  `N1..Np` are the index dimensions specified by the input tensor
            |  `L` is the number of layers in the RNN, equal to :attr:`IRNNv2Layer.num_layers`
            |  `H` is the hidden state for each layer, equal to :attr:`IRNNv2Layer.hidden_size` if getDirection is :const:`RNNDirection.UNIDIRECTION`, and 2x :attr:`IRNNv2Layer.hidden_size` otherwise.

            :returns: The new RNNv2 layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_plugin_ext = R"trtdoc(
            Add a plugin layer to the network using an :class:`IPluginExt` interface.
            See :class:`IPluginExt` for more information.

            :arg inputs: The input tensors to the layer.
            :arg plugin: The layer plugin.

            :returns: The new plugin layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_identity = R"trtdoc(
            Add an identity layer.
            See :class:`IIdentityLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new identity layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_parametric_relu = R"trtdoc(
             Add a parametric ReLU layer.
             See :class:`IParametricReLULayer` for more information.

             :arg input: The input tensor to the layer.
             :arg slopes: The slopes tensor (input elements are multiplied with the slopes where the input is negative).

             :returns: The new parametric ReLU layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_resize = R"trtdoc(
            Add a resize layer.
            See :class:`IResizeLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new resize layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_loop = R"trtdoc(
            Adds a loop to the network, whcih provides a way to specify a recurrent subgraph.
            See :class:`ILoop` for more information.

            :returns: The new loop layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_shape = R"trtdoc(
            Add a shape layer to the network.
            See :class:`IShapeLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new shape layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_select = R"trtdoc(
            Add a select layer.
            See :class:`ISelectLayer` for more information.

            :arg condition: The condition tensor to the layer.
            :arg then_input: The then input tensor to the layer.
            :arg else_input: The else input tensor to the layer.

            :returns: The new select layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* add_fill = R"trtdoc(
            Add a fill layer.
            See :class:`IFillLayer` for more information.

            :arg dimensions: The output tensor dimensions.
            :arg op: The fill operation that the layer applies.

            :returns: The new fill layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* remove_tensor = R"trtdoc(
             Remove a tensor from the network.

             :arg tensor: The tensor to remove

             It is illegal to remove a tensor that is the input or output of a layer.
             if this method is called with such a tensor, a warning will be emitted on the log
             and the call will be ignored.
        )trtdoc";

        constexpr const char* unmark_output = R"trtdoc(
             Unmark a tensor as a network output.

             :arg tensor: The tensor to unmark as an output tensor.
        )trtdoc";

        constexpr const char* mark_output_for_shapes = R"trtdoc(
            Enable tensor's value to be computed by :func:`IExecutionContext.get_shape_binding`.

            :arg tensor: The tensor to unmark as an output tensor. The tensor must be of type :class:`tensorrt.int32` and have no more than one dimension.

            :returns: :class:`True` if successful, :class:`False` if tensor is already marked as an output.
        )trtdoc";

        constexpr const char* unmark_output_for_shapes = R"trtdoc(
            Undo :func:`mark_output_for_shapes` .

            :arg tensor: The tensor to unmark as an output tensor.

            :returns: :class:`True` if successful, :class:`False` if tensor is not marked as an output.

        )trtdoc";

        constexpr const char* add_plugin_v2 = R"trtdoc(
            Add a plugin layer to the network using an :class:`IPluginV2` interface.
            See :class:`IPluginV2` for more information.

            :arg inputs: The input tensors to the layer.
            :arg plugin: The layer plugin.

            :returns: The new plugin layer, or :class:`None` if it could not be created.
        )trtdoc";

        constexpr const char* get_layer = R"trtdoc(
            Get the layer specified by the given index.

            :arg index: The index of the layer.

            :returns: The layer, or :class:`None` if it is out of range.
        )trtdoc";

        constexpr const char* get_input = R"trtdoc(
            Get the input tensor specified by the given index.

            :arg index: The index of the input tensor.

            :returns: The tensor, or :class:`None` if it is out of range.
        )trtdoc";

        constexpr const char* get_output = R"trtdoc(
            Get the output tensor specified by the given index.

            :arg index: The index of the output tensor.

            :returns: The tensor, or :class:`None` if it is out of range.
        )trtdoc";

        constexpr const char* serialize = R"trtdoc(
            Serialize the network to a stream.

            :returns: An :class:`IHostMemory` object containing the serialized :class:`INetworkDefinition` .
        )trtdoc";

    } // INetworkDefinitionDoc

} // tensorrt
