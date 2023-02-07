.. _cqdq_api:

**tensorflow_quantization.CustomQDQInsertionCase**
==================================================

.. autoclass:: tensorflow_quantization.CustomQDQInsertionCase
   :members:

Example

.. code:: python

   class EfficientNetQDQCase(CustomQDQInsertionCase):
    def __init__(self) -> None:
        super().__init__()

    def info(self):
        return "In Multiply operation quantize inputs at index 0 and 1."

    def case(self, keras_model: 'tf.keras.Model', qspec: 'QuantizationSpec') -> 'QuantizationSpec':
        se_block_qspec_object = QuantizationSpec()
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Multiply):
                se_block_qspec_object.add(layer.name, quantize_input=True, quantize_weight=False, quantization_index=[0, 1])
        return se_block_qspec_object

