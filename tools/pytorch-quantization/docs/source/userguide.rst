Basic Functionalities
---------------------

Quantization function
~~~~~~~~~~~~~~~~~~~~~

``tensor_quant`` and ``fake_tensor_quant`` are 2 basic functions to
quantize a tensor. ``fake_tensor_quant`` returns fake quantized tensor
(float value). ``tensor_quant`` returns quantized tensor (integer value)
and scale.

.. code:: python

    tensor_quant(inputs, amax, num_bits=8, output_dtype=torch.float, unsigned=False)
    fake_tensor_quant(inputs, amax, num_bits=8, output_dtype=torch.float, unsigned=False)

Example:

.. code:: python

    from pytorch_quantization import tensor_quant

    # Generate random input. With fixed seed 12345, x should be 
    # tensor([0.9817, 0.8796, 0.9921, 0.4611, 0.0832, 0.1784, 0.3674, 0.5676, 0.3376, 0.2119])
    torch.manual_seed(12345)
    x = torch.rand(10)

    # fake quantize tensor x. fake_quant_x will be 
    # tensor([0.9843, 0.8828, 0.9921, 0.4609, 0.0859, 0.1797, 0.3672, 0.5703, 0.3359, 0.2109])
    fake_quant_x = tensor_quant.fake_tensor_quant(x, x.abs().max())

    # quantize tensor x. quant_x will be
    # tensor([126., 113., 127.,  59.,  11.,  23.,  47.,  73.,  43.,  27.])
    # with scale=128.0057
    quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())

Backward of both functions are defined as `Straight-Through Estimator (STE) <https://arxiv.org/abs/1308.3432>`_.

Descriptor and quantizer
~~~~~~~~~~~~~~~~~~~~~~~~

``QuantDescriptor`` defines how a tensor should be quantized. There are
also some predefined ``QuantDescriptor``, e.g.
``QUANT_DESC_8BIT_PER_TENSOR`` and
``QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL``.

``TensorQuantizer`` is the module for quantizing tensors and defined by
``QuantDescriptor``.

.. code:: python

    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

    quant_desc = QuantDescriptor(num_bits=4, fake_quant=False, axis=(0), unsigned=True)
    quantizer = TensorQuantizer(quant_desc)

    torch.manual_seed(12345)
    x = torch.rand(10, 9, 8, 7)

    quant_x = quantizer(x)

If ``amax`` is given in the :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`, :func:`TensorQuantizer <pytorch_quantization.nn.TensorQuantizer>` will use it to quantize. Otherwise, :func:`TensorQuantizer <pytorch_quantization.nn.TensorQuantizer>`  will compute amax then quantize. amax will be computed w.r.t ``axis`` specified. Note that ``axis`` of QuantDescriptor specify remaining axis as oppsed to axis of `max() <https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html>`_.

Quantized module
~~~~~~~~~~~~~~~~

There are 2 major types of module, ``Conv`` and ``Linear``. Both can
replace ``torch.nn`` version and apply quantization on both weight and
activation.

Both take ``quant_desc_input`` and ``quant_desc_weight`` in addition to
arguments of the original module.

.. code:: python

    from torch import nn

    from pytorch_quantization import tensor_quant
    import pytorch_quantization.nn as quant_nn

    # pytorch's module
    fc1 = nn.Linear(in_features, out_features, bias=True)
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)

    # quantized version
    quant_fc1 = quant_nn.Linear(
        in_features, out_features, bias=True,
        quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
        quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW)
    quant_conv1 = quant_nn.Conv2d(
        in_channels, out_channels, kernel_size,
        quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
        quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)

Post training quantization
--------------------------

A model can be post training quantized by simply by calling ``quant_modules.initialize()``

.. code:: python

    from pytorch_quantization import quant_modules
    model = torchvision.models.resnet50()

If a model is not entirely defined by module, than TensorQuantizer should be 
manually created and added to the right place in the model.

Calibration
~~~~~~~~~~~

Calibration is the TensorRT terminology of passing data samples to the
quantizer and deciding the best amax for activations. We support 3
calibration method:

-  ``max``: Simply use global maximum absolute value
-  ``entropy``: TensorRT's entropy calibration
-  ``percentile``: Get rid of outlier based on given percentile.
-  ``mse``: MSE(Mean Squared Error) based calibration

In above ResNet50 example, calibration method is set to ``mse``, it can
be used as the following example:

.. code:: python

    # Find the TensorQuantizer and enable calibration
    for name, module in model.named_modules():
        if name.endswith('_input_quantizer'):
            module.enable_calib()
            module.disable_quant()  # Use full precision data to calibrate
            
    # Feeding data samples
    model(x)
    # ...

    # Finalize calibration
    for name, module in model.named_modules():
        if name.endswith('_input_quantizer'):
            module.load_calib_amax()
            module.enable_quant()
            
    # If running on GPU, it needs to call .cuda() again because new tensors will be created by calibration process
    model.cuda()

    # Keep running the quantized model
    # ...

Quantization Aware Training
---------------------

Quantization Aware Training is based on Straight Through Estimator (STE)
derivative approximation. It is some time known as “quantization aware
training”. We don’t use the name because it doesn’t reflect the
underneath assumption. If anything, it makes training being “unaware” of
quantization because of the STE approximation.

After calibration is done, Quantization Aware Training is simply select a
training schedule and continue training the calibrated model. Usually,
it doesn’t need to fine tune very long. We usually use around 10% of the
original training schedule, starting at 1% of the initial training
learning rate, and a cosine annealing learning rate schedule that
follows the decreasing half of a cosine period, down to 1% of the
initial fine tuning learning rate (0.01% of the initial training
learning rate).

Some recommendations
~~~~~~~~~~~~~~~~~~~~

Quantization Aware Training (Essentially a discrete numerical optimization
problem) is not a solved problem mathematically. Based on our
experience, here are some recommendations:

-  For STE approximation to work well, it is better to use small
   learning rate. Large learning rate is more likely to enlarge the
   variance introduced by STE approximation and destroy the trained
   network.
-  Do not change quantization representation (scale) during training, at
   least not too frequently. Changing scale every step, it is
   effectively like changing data format (e8m7, e5m10, e3m4, et.al)
   every step, which will easily affect convergence.

Export to ONNX
--------------

The goal of exporting to ONNX is to deploy inference by TensorRT, not
ONNX runtime. So we only export fake quantized model into a form TensorRT will take. Fake
quantization will be broken into a pair of
QuantizeLinear/DequantizeLinear ONNX ops. In future, TensorRT will take
the graph, and execute it in int8 in the most optimized way to its
capability.

Pytorch doesn’t support exporting fake quantize ops to ONNX yet, but the
code is simple. Add the following code to
``torch/onnx/symbolic_opset10.py``

.. code:: python

   @parse_args('v', 't', 'i', 'i', 'i')
   def fake_quantize_per_tensor_affine(g, inputs, scale, zero_point, quant_min=-128, quant_max=127):
       if quant_min not in [0, -128] or quant_max not in [127, 255]:
           raise TypeError("ONNX defines [0, 255] for quint8 and [-128, 127] for int8, got [{}, {}]".format(
               quant_min, quant_max))
       scale = scale.float()  # Avoid exportor generating double type
       zero_point = torch.tensor(zero_point, dtype=torch.int8)  # ONNX requires zero_point to be tensor
       return g.op("DequantizeLinear", g.op("QuantizeLinear", inputs, scale, zero_point), scale, zero_point)

   @parse_args('v', 'v', 'v', 'i', 'i', 'i')
   def fake_quantize_per_channel_affine(g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
       if quant_min not in [0, -128] or quant_max not in [127, 255]:
           raise TypeError("ONNX defines [0, 255] for quint8 and [-128, 127] for int8, got [{}, {}]".format(
               quant_min, quant_max))
       # ONNX defines zero_point to be int8 or uint8
       if quant_min == 0:
           zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx['Byte'])
       else:
           zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx['Char'])
       return g.op(
           "DequantizeLinear",
           g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis),
           scale, zero_point, axis_i=axis)

Then set static member of TensorQuantizer to use Pytorch’s own fake
quantization functions

.. code:: python

   from pytorch_quantization import nn as quant_nn
   quant_nn.TensorQuantizer.use_fb_fake_quant = True

Fake quantized model can now be exported to ONNX as other models, follow
the instructions in
`torch.onnx <https://pytorch.org/docs/stable/onnx.html?highlight=onnx#module-torch.onnx>`__.
For example:

.. code:: python

   from pytorch_quantization import nn as quant_nn
   from pytorch_quantization import quant_modules
   quant_nn.TensorQuantizer.use_fb_fake_quant = True

   quant_modules.initialize()
   model = torchvision.models.resnet50()
   # load the calibrated model
   state_dict = torch.load("quant_resnet50-entropy-1024.pth", map_location="cpu")
   model.load_state_dict(state_dict)
   model.cuda()

   dummy_input = torch.randn(128, 3, 224, 224, device='cuda')

   input_names = [ "actual_input_1" ]
   output_names = [ "output1" ]

   # enable_onnx_checker needs to be disabled. See notes below.
   torch.onnx.export(
       model, dummy_input, "quant_resnet50.onnx", verbose=True, opset_version=10, enable_onnx_checker=False)

.. Note::

    Note that ``axis`` is added to ``QuantizeLinear`` and ``DequantizeLinear`` in opset13.
