#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import tensorflow as tf
from tensorflow.python.util import tf_inspect
from tensorflow_quantization.quantize_wrapper_base import BaseQuantizeWrapper
import warnings

"""
Naming convention for keras `layer` quantize wrapper is
<layer.__class__.__name__>QuantizeWrapper
"""

DISABLED_LAYER_QUANTIZATION_DEFAULT = [
    "MaxPooling2D",
    "BatchNormalization",
    "Add",
    "Multiply",
    "Concatenate"
]


# ##############################################
# ############# Weighted Layers ################
# ##############################################


class WeightedBaseQuantizeWrapper(BaseQuantizeWrapper):
    """
    BaseQuantizeWrapper for weighted layers: Conv2D, DepthwiseConv2D, and Dense layer.
        These layers share a lot of the same code except for a few modifications. Conv2D and Dense share the same code.
        Layers that inherit this class support weight and input QDQ nodes.

    TRT Rule:
        One Q/DQ pair should be attached to the input activation, and another Q/DQ pair should be attached to weights.
        Weights tensor is per-channel quantized:
            For the Q/DQ attached to weight tensor, set axis=0 and axis=1 for Conv and ConvTransposed respectively.
        Input tensor is per-tensor quantized.
    """

    def __init__(
        self, layer: tf.keras.layers.Layer, kernel_type: str = "kernel", **kwargs
    ):
        """
        Creates a wrapper to emulate quantization for a keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          kernel_type (str): Options=['kernel' for Conv2D/Dense, 'depthwise_kernel' for DepthwiseConv2D]
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        self.kernel_type = kernel_type
        self.channel_axis = kwargs.get("axis", -1)
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        self._weight_vars = []
        self.input_vars = {}
        self.output_vars = {}
        self.channel_axis = -1
        if self.kernel_type == "depthwise_kernel":
            self.channel_axis = 2
        # quantize weights only applicable for weighted ops.
        # By default weights is per channel quantization
        if self.quantize_weights:
            # get kernel weights dims.
            kernel_weights = getattr(self.layer, self.kernel_type)
            min_weight = self.layer.add_weight(
                kernel_weights.name.split(":")[0] + "_min",
                shape=(kernel_weights.shape[self.channel_axis]),
                initializer=tf.keras.initializers.Constant(-6.0),
                trainable=False,
            )
            max_weight = self.layer.add_weight(
                kernel_weights.name.split(":")[0] + "_max",
                shape=(kernel_weights.shape[self.channel_axis]),
                initializer=tf.keras.initializers.Constant(6.0),
                trainable=False,
            )
            quantizer_vars = {"min_var": min_weight, "max_var": max_weight}
            self._weight_vars.append((kernel_weights, quantizer_vars))
            # Needed to ensure unquantized weights get trained as part of the wrapper.
            self._trainable_weights.append(kernel_weights)

        # By default input is per tensor quantization
        if self.quantize_inputs:
            input_min_weight = self.layer.add_weight(
                self.layer.name + "_ip_min",
                shape=None,
                initializer=tf.keras.initializers.Constant(-6.0),
                trainable=False,
            )
            input_max_weight = self.layer.add_weight(
                self.layer.name + "_ip_max",
                shape=None,
                initializer=tf.keras.initializers.Constant(6.0),
                trainable=False,
            )
            self.input_vars["min_var"] = input_min_weight
            self.input_vars["max_var"] = input_max_weight

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # Quantize all weights, and replace them in the underlying layer.
        if self.quantize_weights:
            quantized_weights = []
            quantized_weight = self._last_value_quantizer(
                self._weight_vars[0][0],
                training,
                self._weight_vars[0][1],
                per_channel=True,
                channel_axis=self.channel_axis,
            )
            quantized_weights.append(quantized_weight)
            # Replace the original weights with QDQ weights
            setattr(self.layer, self.kernel_type, quantized_weights[0])

        # Quantize inputs to the conv layer
        if self.quantize_inputs:
            quantized_inputs = self._last_value_quantizer(
                inputs, training, self.input_vars, per_channel=False
            )
        else:
            quantized_inputs = inputs

        args = tf_inspect.getfullargspec(self.layer.call).args
        if "training" in args:
            outputs = self.layer.call(quantized_inputs, training=training)
        else:
            outputs = self.layer.call(quantized_inputs)

        return outputs


class Conv2DQuantizeWrapper(WeightedBaseQuantizeWrapper):
    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the Conv2D keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        self.kernel_type = "kernel"
        super().__init__(layer, kernel_type=self.kernel_type, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class DenseQuantizeWrapper(WeightedBaseQuantizeWrapper):
    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the Dense keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        self.kernel_type = "kernel"
        super().__init__(layer, kernel_type=self.kernel_type, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class DepthwiseConv2DQuantizeWrapper(WeightedBaseQuantizeWrapper):
    """Requires TF >= 2.8.0"""

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the DepthwiseConv2D keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        self.kernel_type = "depthwise_kernel"
        super().__init__(layer, kernel_type=self.kernel_type, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


# ##############################################
# ########### Non-Weighted Layers ##############
# ######### with Single Input/Output ###########
# ##############################################


class NonWeightedBaseQuantizeWrapper(BaseQuantizeWrapper):
    """
    BaseQuantizeWrapper for non-weighted layers with Single Input/Output: AveragePooling2D, GlobalAveragePooling,
        MaxPooling2D and BatchNormalization.

    Supports 1 input and 1 output QDQ. Similar to Concat, except that Concat supports multiple inputs.
    NonWeightedBaseQuantizeWrapper can use WeightedBaseQuantizeWrapper by giving quantize_weigths=False.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for non-weighted keras layers.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        self.input_vars = {}

        # By default input is per tensor quantization
        if self.quantize_inputs:
            input_min_weight = self.layer.add_weight(
                self.layer.name + "_ip_min",
                shape=None,
                initializer=tf.keras.initializers.Constant(-6.0),
                trainable=False,
            )
            input_max_weight = self.layer.add_weight(
                self.layer.name + "_ip_max",
                shape=None,
                initializer=tf.keras.initializers.Constant(6.0),
                trainable=False,
            )
            self.input_vars["min_var"] = input_min_weight
            self.input_vars["max_var"] = input_max_weight

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # Quantize inputs to the conv layer
        if self.quantize_inputs:
            quantized_inputs = self._last_value_quantizer(
                inputs, training, self.input_vars, per_channel=False
            )
        else:
            quantized_inputs = inputs

        args = tf_inspect.getfullargspec(self.layer.call).args
        if "training" in args:
            outputs = self.layer.call(quantized_inputs, training=training)
        else:
            outputs = self.layer.call(quantized_inputs)

        return outputs


class AveragePooling2DQuantizeWrapper(NonWeightedBaseQuantizeWrapper):
    """
    TRT Rule:
        Add Q/DQ to its input if the ops follows is quantized.
        Quantize average pooling will introduce small variance compared to float because of the rounding change.
        TensorRT doesn’t have Int8 in and fp32 out average pool support.
        If the op follows average pooling is not quantized, it is users choice between running average pooling
        in int8 then convert to fp32 for the following op and run average pooling in fp32.

    Currently, we're adding QDQ to all AveragePooling2D layers.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the AveragePooling2D keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class GlobalAveragePooling2DQuantizeWrapper(NonWeightedBaseQuantizeWrapper):
    """
    TRT Rule:
        No explicit rule from the TRT team. Following the same as AveragePooling2D.

    Residual block v2: Add to MaxPool (branch1) and BN (branch2).

    Supports 1 input and 1 output QDQ. Same as AveragePooling2DQuantizeWrapper.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the GlobalAveragePooling2D keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class MaxPooling2DQuantizeWrapper(NonWeightedBaseQuantizeWrapper):
    """
    TRT Rule:
        Max pooling is precision-neutral. But unlike ReLU, input and output of max pooling will have different
            histograms which will lead to different calibration results.
        The recommendation is to let TensorRT optimize precision neutral ops.
        There are cases where adding Q/DQ before maxpool can enable additional optimization.

    Residual block v2: Add to MaxPool (branch1) and BN (branch2).

    Supports 1 input and 1 output QDQ. Same as AveragePooling2DQuantizeWrapper.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the MaxPooling2D keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class BatchNormalizationQuantizeWrapper(NonWeightedBaseQuantizeWrapper):
    """
    TRT Rule:
        Keep batch normalization untouched, don't add Q/DQ to its input and not necessary to fold it before exporting
            graph. TensorRT supports Batch normalization folding. It can take a graph with batch normalization, fold it
            into previous convolution and create a new graph.
        If batch normalization is folded before exporting the graph, TensorRT can still import and execute the graph as
            it becomes regular convolutions.

    Exception for Residual block v2:
        BN-ReLU-Conv2D -> need to add Q/DQ before BN in order to run in INT8.
        In order to do that, we add a check in 'quantize_model()' to check if BN's parent is a Conv layer. If it is, set
            quantize_inputs to False. The reason why we don't add this check here is to allow the user to add QDQ nodes
            before BN if they so wish.

    Supports 1 input and 1 output QDQ. Same as AveragePooling2DQuantizeWrapper.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the BatchNormalization keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


# ##############################################
# ########### Non-Weighted Layers ##############
# #### with Multiple Inputs, Single Output #####
# ##############################################


class NonWeightedBaseQuantizeWrapperForMultipleInputs(BaseQuantizeWrapper):
    """
    BaseQuantizeWrapper for non-weighted layers with Multiple Inputs: Concat, Add, and Multiply.
    Supports multiple inputs and 1 output QDQ. Similar to AveragePooling2D, except pooling supports only a single input.

    TRT Rule:
        Add Q/DQ to all inputs of the layer.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def _should_quantization_this_index(self, i):
        if not self.quantize_specific_input_indices:
            return True
        else:
            # This is a small list so iterating makes sense
            for e in self.quantize_specific_input_indices:
                if e == i:
                    return True
                elif e >= self.num_inputs:
                    warnings.warn(
                        "{layer_name} has {num_inputs} inputs but quantization index {e} is passed.".format(
                            layer_name=self.layer.name, num_inputs=self.num_inputs, e=e
                        )
                    )
            return False

    def build(self, input_shape):
        super().build(input_shape)

        self.input_vars = []  # list of dictionaries
        self.num_inputs = len(input_shape)

        # By default input is per tensor quantization
        if self.quantize_inputs:
            # for concat input is list of Tensors
            layer_name_key_idx = 0
            for i in range(self.num_inputs):
                if self._should_quantization_this_index(i):
                    input_min_weight = self.layer.add_weight(
                        self.layer.name + "_ip{}_min".format(layer_name_key_idx),
                        shape=None,
                        initializer=tf.keras.initializers.Constant(-6.0),
                        trainable=False,
                    )
                    input_max_weight = self.layer.add_weight(
                        self.layer.name + "_ip{}_max".format(layer_name_key_idx),
                        shape=None,
                        initializer=tf.keras.initializers.Constant(6.0),
                        trainable=False,
                    )
                    self.input_vars.append(
                        {"min_var": input_min_weight, "max_var": input_max_weight}
                    )
                    layer_name_key_idx += 1

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # Quantize inputs to the conv layer
        quantized_inputs = inputs[:]
        if self.quantize_inputs:
            input_vars_idx = 0
            for i in range(len(inputs)):
                if self._should_quantization_this_index(i):
                    quantized_inputs[i] = self._last_value_quantizer(
                        inputs[i],
                        training,
                        self.input_vars[input_vars_idx],
                        per_channel=False,
                    )
                    input_vars_idx += 1

        args = tf_inspect.getfullargspec(self.layer.call).args
        if "training" in args:
            outputs = self.layer.call(quantized_inputs, training=training)
        else:
            outputs = self.layer.call(quantized_inputs)

        return outputs


class MultiplyQuantizeWrapper(NonWeightedBaseQuantizeWrapperForMultipleInputs):
    """
    TRT Rule:
        Add Q/DQ to all inputs of Multiply layer in SE block.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the Multiply keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class ConcatenateQuantizeWrapper(NonWeightedBaseQuantizeWrapperForMultipleInputs):
    """
    TRT Rule:
        Add Q/DQ to all inputs.
        Alternative: If there is Q/DQ attached to the input of the op after concat, don't add Q/DQ to input to concat,
            let TensorRT pull from the next op.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the Concatenate keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class AddQuantizeWrapper(NonWeightedBaseQuantizeWrapperForMultipleInputs):
    """
    TRT Rule:
        If the add is NOT bias. Attach Q/DQ to all of its input.
        Exception: add in residual block. To trigger fusion, Attach Q/DQ to the residual being added to output of
            convolution.
    """

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """
        Creates a wrapper to emulate quantization for the Add keras layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)
