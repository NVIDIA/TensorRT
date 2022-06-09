.. _qmodel_api:

**tensorflow_quantization.quantize_model**
============================================

.. automodule:: tensorflow_quantization.quantize
   :members: quantize_model

.. note:: Currently only Functional and Sequential models are supported.

Examples

.. code:: python

   import tensorflow as tf
   from tensorflow_quantization.quantize import quantize_model

   # Simple full model quantization.
   # 1. Create a simple network
   input_img = tf.keras.layers.Input(shape=(28, 28))
   r = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_img)
   x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3))(r)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3))(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Flatten()(x)
   model = tf.keras.Model(input_img, x)

   print(model.summary())

   # 2. Quantize the network
   q_model = quantize_model(model)
   print(q_model.summary())