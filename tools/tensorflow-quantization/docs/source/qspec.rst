.. _qspec_api:

**tensorflow_quantization.QuantizationSpec**
=============================================
.. autoclass:: tensorflow_quantization.QuantizationSpec
   :members:

Examples

Let's write a simple network to use in all examples.

.. code-block:: python

   import tensorflow as tf
   # Import necessary methods from the Quantization Toolkit
   from tensorflow_quantization.quantize import quantize_model, QuantizationSpec

   # 1. Create a small network
   input_img = tf.keras.layers.Input(shape=(28, 28))
   x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_img)
   x = tf.keras.layers.Conv2D(filters=126, kernel_size=(3, 3))(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3))(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
   x = tf.keras.layers.Flatten()(x)
   x = tf.keras.layers.Dense(100)(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Dense(10)(x)
   model = tf.keras.Model(input_img, x)


#. **Select layers based on layer names**  

   **Goal**: Quantize the 2nd Conv2D, 4th Conv2D and 1st Dense layer in the following network.

   .. code-block:: python

      # 1. Find out layer names
      print(model.summary())

      # 2. Create quantization spec and add layer names
      q_spec = QuantizationSpec()
      layer_name = ['conv2d_1', 'conv2d_3', 'dense']

      """
      # Alternatively, each layer configuration can be added one at a time:
      q_spec.add('conv2d_1')
      q_spec.add('conv2d_3')
      q_spec.add('dense')
      """

      q_spec.add(name=layer_name)

      # 3. Quantize model
      q_model = quantize_model(model, quantization_mode='partial', quantization_spec=q_spec)
      print(q_model.summary())

      tf.keras.backend.clear_session()


#. **Select layers based on layer class**
   
   **Goal**: Quantize all `Conv2D` layers.

   .. code-block:: python

      # 1. Create QuantizationSpec object and add layer class
      q_spec = QuantizationSpec()
      q_spec.add(name='Conv2D', is_keras_class=True)

      # 2. Quantize model
      q_model = quantize_model(model, quantization_mode='partial', quantization_spec=q_spec)
      q_model.summary()

      tf.keras.backend.clear_session()

#. **Select layers based both layer name and layer class**

   **Goal**: Quantize all `Dense` layers and the 3rd `Conv2D` layer.

   .. code-block:: python

      # 1. Create QuantizationSpec object and add layer information
      q_spec = QuantizationSpec()

      layer_name = ['Dense', 'conv2d_2']
      layer_is_keras_class = [True, False]

      """
      # Alternatively, each layer configuration can be added one at a time:
      q_spec.add(name='Dense', is_keras_class=True)
      q_spec.add(name='conv2d_2')
      """

      q_spec.add(name=layer_name, is_keras_class=layer_is_keras_class)

      # 2. Quantize model
      q_model = quantize_model(model, quantization_mode='partial', quantization_spec=q_spec)
      q_model.summary()

      tf.keras.backend.clear_session()

#. **Select inputs at specific index for multi-input layers**  

   For layers with multiple inputs, the user can choose which ones need to be quantized. Assume a network that has two layers of class `Add`.

   **Goal**: Quantize index 1 of `add` layer, index 0 of `add_1` layer and the 3rd `Conv2D` layer.

   .. code-block:: python

      # 1. Create QuantizationSpec object and add layer information
      q_spec = QuantizationSpec()

      layer_name = ['add', 'add_1', 'conv2d_2']
      layer_q_indices = [[1], [0], None]

      """
      # Alternatively, each layer configuration can be added one at a time:
      q_spec.add(name='add', quantization_index=[1])
      q_spec.add(name='add', quantization_index=[0])
      q_spec.add(name='conv2d_2')
      """

      q_spec.add(name=layer_name, quantization_index=layer_q_indices)

      # 2. Quantize model
      q_model = quantize_model(model, quantization_mode='partial', quantization_spec=q_spec)
      q_model.summary()

      tf.keras.backend.clear_session()

#. **Quantize only weight and NOT input**  

   **Goal**: Quantize the 2nd Conv2D, 4th Conv2D and 1st Dense layer in the following network. In addition to that, quantize only the weights of the 2nd Conv2D.

   .. code-block:: python

      # 1. Find out layer names
      print(model.summary())

      # 2. Create quantization spec and add layer names
      q_spec = QuantizationSpec()
      layer_name = ['conv2d_1', 'conv2d_3', 'dense']
      layer_q_input = [False, True, True]

      """
      # Alternatively, each layer configuration can be added one at a time:
      q_spec.add('conv2d_1', quantize_input=False)
      q_spec.add('conv2d_3')
      q_spec.add('dense')
      """

      q_spec.add(name=layer_name, quantize_input=layer_q_input)

      # 3. Quantize model
      q_model = quantize_model(model, quantization_mode='partial', quantization_spec=q_spec)
      print(q_model.summary())

      tf.keras.backend.clear_session()
