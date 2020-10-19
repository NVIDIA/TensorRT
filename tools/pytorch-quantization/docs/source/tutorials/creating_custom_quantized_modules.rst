Creating Custom Quantized Modules
=================================

There are several quantized modules provided by the quantization tool as follows:

- QuantConv1d, QuantConv2d, QuantConv3d, QuantConvTranspose1d, QuantConvTranspose2d, QuantConvTranspose3d
- QuantLinear
- QuantAvgPool1d, QuantAvgPool2d, QuantAvgPool3d, QuantMaxPool1d, QuantMaxPool2d, QuantMaxPool3d

To quantize a module, we need to quantize the input and weights if present. Following are 3 major use-cases:

#. Create quantized wrapper for modules that have only inputs
#. Create quantized wrapper for modules that have inputs as well as weights.
#. Directly add the ``TensorQuantizer`` module to the inputs of an operation in the model graph.

The first two methods are very useful if it's needed to automatically replace the original modules (nodes in the graph) with their quantized versions. The third method could be useful when it's required to manually add the quantization to the model graph at very specific places (more manual, more control).

Let's see each use-case with examples below.

Quantizing Modules With Only Inputs
-----------------------------------
A suitable example would be quantizing the ``pooling`` module variants.

Essentially, we need to provide a wrapper function that takes the original module and adds the ``TensorQuantizer`` module around it so that the input is first quantized and then fed into the original module. 

- Create the wrapper by subclassing the original module (``pooling.MaxPool2d``) along with the utilities module (``_utils.QuantInputMixin``).

.. code:: python

    class QuantMaxPool2d(pooling.MaxPool2d, _utils.QuantInputMixin):

- The ``__init__.py`` function would call the original module's init function and provide it with the corresponding arguments. There would be just one additional argument using ``**kwargs`` which contains the quantization configuration information. The ``QuantInputMixin`` utility contains the method ``pop_quant_desc_in_kwargs`` which extracts this configuration information from the input or returns a default if that input is ``None``. Finally the ``init_quantizer`` method is called that initializes the ``TensorQuantizer`` module which would quantize the inputs.

.. code:: python

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

- After the initialization, the ``forward`` function needs to be defined in our wrapper module that would actually quantize the inputs using the ``_input_quantizer`` that was initialized in the ``__init__`` function forwarding the inputs to the base module using ``super`` call.

.. code:: python

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantMaxPool2d, self).forward(quant_input)

- Finally, we need to define a getter method for the ``_input_quantizer``. This could, for example, be used  to disable the quantization for a particular module using ``module.input_quantizer.disable()`` which is helpful while experimenting with different layer quantization configuration.

.. code:: python

    @property
    def input_quantizer(self):
        return self._input_quantizer

A complete quantized pooling module would look like following:

.. code:: python

    class QuantMaxPool2d(pooling.MaxPool2d, _utils.QuantInputMixin):
        """Quantized 2D maxpool"""
        def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                    return_indices=False, ceil_mode=False, **kwargs):
            super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                                return_indices, ceil_mode)
            quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
            self.init_quantizer(quant_desc_input)

        def forward(self, input):
            quant_input = self._input_quantizer(input)
            return super(QuantMaxPool2d, self).forward(quant_input)

        @property
        def input_quantizer(self):
            return self._input_quantizer

Quantizing Modules With Weights and Inputs
------------------------------------------
We give an example of quantizing the ``torch.nn.Linear`` module. It follows that the only additional change from the previous example of quantizing pooling modules is that we'd need to accomodate the quantization of weights in the Linear module. 

- We create the quantized linear module as follows:

.. code:: python

    class QuantLinear(nn.Linear, _utils.QuantMixin):

- In the ``__init__`` function, we first use the ``pop_quant_desc_in_kwargs`` function to extract the quantization descriptors for both inputs and weights. Second, we initialize the ``TensorQuantizer`` modules for both inputs and weights using these quantization descriptors.

.. code:: python

    def __init__(self, in_features, out_features, bias=True, **kwargs):
            super(QuantLinear, self).__init__(in_features, out_features, bias)
            quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

            self.init_quantizer(quant_desc_input, quant_desc_weight)

- Also, override the ``forward`` function call and pass the inputs and weights through ``_input_quantizer`` and ``_weight_quantizer`` respectively before passing the quantized arguments to the actual ``F.Linear`` call. This step adds the actual input/weight ``TensorQuantizer`` to the module and eventually the model.

.. code:: python

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        output = F.linear(quant_input, quant_weight, bias=self.bias)

        return output

- Also similar to the ``Linear`` module, we add the getter methods for the ``TensorQuantizer`` modules associated with inputs/weights. This could be used to, for example, disable the quantization mechanism by calling ``module_obj.weight_quantizer.disable()``

.. code:: python

    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer

- With all of the above changes, the quantized Linear module would look like following:

.. code:: python

    class QuantLinear(nn.Linear, _utils.QuantMixin):

        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super(QuantLinear, self).__init__(in_features, out_features, bias)
            quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

            self.init_quantizer(quant_desc_input, quant_desc_weight)

        def forward(self, input):
            quant_input = self._input_quantizer(input)
            quant_weight = self._weight_quantizer(self.weight)

            output = F.linear(quant_input, quant_weight, bias=self.bias)

            return output

        @property
        def input_quantizer(self):
            return self._input_quantizer

        @property
        def weight_quantizer(self):
            return self._weight_quantizer


Directly Quantizing Inputs In Graph
-----------------------------------
It is also possible to directly quantize graph inputs without creating wrappers as explained above.

Here's an example:

.. code:: python

    test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

    quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

    quant_input = quantizer(test_input)

    out = F.adaptive_avg_pool2d(quant_input, 3)

Assume that there is a ``F.adaptive_avg_pool2d`` operation in the graph and we'd like to quantize this operation. In the example above, we use ``TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)`` to define a quantizer that we then use to actually quantize the ``test_input`` and then feed this quantized input to the ``F.adaptive_avg_pool2d`` operation. Note that this quantizer is the same as the ones we used earlier while created quantized versions of torch's modules.
