.. _globals_api:

**tensorflow_quantization**
============================


:tensorflow_quantization.G_NUM_BITS:

    8 bit quantization is used by default. However, it can be changed by using ``G_NUM_BITS`` global variable.
    The following code snippet performs 4 bit quantization.

    .. code:: python

        import tensorflow_quantization
        # get pretrained model
        .....

        # perform 4 bit quantization
        tensorflow_quantization.G_NUM_BITS = 4
        q_model = quantize_model(nn_model_original)

        # fine-tune model
        .....
        
    Check ``test_end_to_end_workflow_4bit()`` test case from ``quantize_test.py`` test module.

:tensorflow_quantization.G_NARROW_RANGE:

    If True, the absolute value of quantized minimum is the same as the quantized maximum value. For example,
    minimum of -127 is used for 8 bit quantization instead of -128. TensorRT |tred| only supports G_NARROW_RANGE=True.

:tensorflow_quantization.G_SYMMETRIC:

    If True, 0.0 is always in the center of real min, max i.e. zero point is always 0. 
    TensorRT |tred| only supports G_SYMMETRIC=True.

.. attention:: When used, set global variables immediately before the ``quantize_model`` function call.

.. |tred|    unicode:: U+2122 .. TRADEMARK SIGN