.. _bqw_api:

**tensorflow_quantization.BaseQuantizeWrapper**
==================================================

.. autoclass:: tensorflow_quantization.BaseQuantizeWrapper
   :members:

Example

`Conv2DTranspose` layer is a weighted layer used to perform transformations going in the opposite direction of `Convolution`.

.. note:: `Conv2DTranspose` is a Keras class, thus new wrapper class is `Conv2DTransposeQuantizeWrapper`. This follows toolkit naming conventions.

.. code:: python

    from tensorflow.python.util import tf_inspect
    from tensorflow_quantization.quantize_wrapper_base import BaseQuantizeWrapper

    class Conv2DTransposeQuantizeWrapper(BaseQuantizeWrapper):
        def __init__(self, layer, kernel_type="kernel", **kwargs):
            """
            Create a quantize emulate wrapper for a keras layer.
            This wrapper provides options to quantize inputs, outputs amd weights of a quantizable layer.
            Args:
            layer: The keras layer to be quantized.
            kernel_type: Options=['kernel' for Conv2D/Dense, 'depthwise_kernel' for DepthwiseConv2D]
            **kwargs: Additional keyword arguments to be passed to the keras layer.
            """
            self.kernel_type = kernel_type
            self.channel_axis = kwargs.get("axis", -1)
            super(Conv2DTransposeQuantizeWrapper, self).__init__(layer, **kwargs)

        def build(self, input_shape):
            super(Conv2DTransposeQuantizeWrapper, self).build(input_shape)

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
                    channel_axis=self.channel_axis
                )
                quantized_weights.append(quantized_weight)
                # Replace the original weights with QDQ weights
                setattr(self.layer, self.kernel_type, quantized_weights[0])

            # Quantize inputs to the conv layer
            if self.quantize_inputs:
                quantized_inputs = self._last_value_quantizer(
                    inputs, 
                    training,
                    self.input_vars,
                    per_channel=False)
            else:
                quantized_inputs = inputs

            args = tf_inspect.getfullargspec(self.layer.call).args
            if "training" in args:
                outputs = self.layer.call(quantized_inputs, training=training)
            else:
                outputs = self.layer.call(quantized_inputs)

            return outputs
