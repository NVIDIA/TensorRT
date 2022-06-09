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
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#    
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow_quantization.quantizers as quantizers
import tensorflow_quantization.global_config as cfg
from abc import abstractmethod

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object

NO_WEIGHT_LAYERS = {
    "Concatenate",
    "Add",
    "AveragePooling2D",
    "GlobalAveragePooling2D",
    "MaxPooling2D",
    "BatchNormalization",
}


class BaseQuantizeWrapper(tf.keras.layers.Wrapper):
    """Base wrapper class which all layer wrappers inherit"""

    CHILD_WRAPPERS = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.CHILD_WRAPPERS[cls.__name__] = cls

    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        """Create a quantize emulate wrapper for a keras layer.
        This wrapper provides options to quantize inputs and weights of the layer.
        Args:
          layer (tf.keras.layers.Layer): The keras layer to be quantized.
          **kwargs: Additional keyword arguments to be passed to the keras layer.
        """
        if layer is None:
            raise ValueError("`layer` cannot be None.")

        # Check against keras.Model since it is an instance of keras.layers.Layer.
        if not isinstance(layer, tf.keras.layers.Layer) or isinstance(
            layer, tf.keras.Model
        ):
            raise ValueError(
                "`layer` can only be a `tf.keras.layers.Layer` instance. "
                "You passed an instance of type: {input}.".format(
                    input=layer.__class__.__name__
                )
            )
        if "name" not in kwargs:
            kwargs["name"] = self._make_layer_name(layer)
        super(BaseQuantizeWrapper, self).__init__(layer, **kwargs)

        # get quantize config object that holds all the information about how quantization should be performed.
        quantize_config_object = cfg.get_config_object()

        # set all initial quantization parameters to False/None
        self.quantize_inputs = False
        self.quantize_weights = False
        self.quantize_specific_input_indices = None

        layer_class_name_t = layer.__class__.__name__  # Layer class name
        layer_name_t = layer.name  # Actual layer name

        def _configure_singular_quantize():
            self.quantize_inputs = True
            if layer_class_name_t in NO_WEIGHT_LAYERS:
                self.quantize_weights = False
            else:
                self.quantize_weights = True

        def _configure_special_quantize(
            quantize_bool_list: list, layer_name_t: str, index_list_if_any: list = None
        ):
            assert (len(quantize_bool_list)) == 2, (
                "Three boolean values (representing whether to quantize [inputs, weights]) must be provided in "
                "quantize_config for layer: {layer_name_t}. If quantization does not apply for specific part, "
                "pass None. e.g. For layer ( e.g. Concatenate, Add) with no weights, `qbool_list` to quantize "
                "input can be [True, False]".format(layer_name_t=layer_name_t)
            )

            self.quantize_inputs = quantize_bool_list[0]
            if layer_class_name_t in NO_WEIGHT_LAYERS:
                self.quantize_weights = False
            else:
                self.quantize_weights = quantize_bool_list[1]

            if index_list_if_any:
                self.quantize_specific_input_indices = index_list_if_any

        if quantize_config_object.config_class_id == 0:
            # This is straight forward full network quantization
            _configure_singular_quantize()
        else:
            # Config class id 1 or 2.
            # User has provided layer (name) specific quantization information
            quantize_config_dict = quantize_config_object.get_layer_config()
            if layer_name_t in quantize_config_dict:
                # This layer needs to be quantized in specific way
                if "qindex_list" in quantize_config_dict[layer_name_t]:
                    _configure_special_quantize(
                        quantize_config_dict[layer_name_t]["qbool_list"],
                        layer_name_t,
                        quantize_config_dict[layer_name_t]["qindex_list"],
                    )
                else:
                    _configure_special_quantize(
                        quantize_config_dict[layer_name_t]["qbool_list"], layer_name_t
                    )
            else:
                _configure_singular_quantize()

        self._track_trackable(layer, name="layer")

    @staticmethod
    def _make_layer_name(layer):
        return "{}_{}".format("quant", layer.name)

    @staticmethod
    def _weight_name(name):
        """Extracts the weight name from the full TensorFlow variable name.
        For example, returns 'kernel' for 'dense_2/kernel:0'.
        Args:
          name: TensorFlow variable name.
        Returns:
          Extracted weight name.
        """
        return name.split(":")[0].split("/")[-1]

    def build(self, input_shape):
        super(BaseQuantizeWrapper, self).build(input_shape)

        self.optimizer_step = self.add_weight(
            "optimizer_step",
            initializer=tf.keras.initializers.Constant(-1),
            dtype=tf.dtypes.int32,
            trainable=False,
        )

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(self.layer.input_shape)

    def _last_value_quantizer(
        self, x, training, quantizer_vars, per_channel=False, channel_axis=-1
    ):
        """Use currying to return True/False specialized fns to the cond."""
        from tensorflow_quantization import G_NUM_BITS, G_SYMMETRIC, G_NARROW_RANGE

        return quantizers.LastValueQuantize(
            x,
            quantizer_vars["min_var"],
            quantizer_vars["max_var"],
            per_channel=per_channel,
            channel_axis=channel_axis,
            is_training=training,
            num_bits=G_NUM_BITS,
            narrow_range=G_NARROW_RANGE,
            symmetric=G_SYMMETRIC,
        )

    @abstractmethod
    def call(self, inputs, training=None):
        raise NotImplementedError

    def get_config(self):
        base_config = super(BaseQuantizeWrapper, self).get_config()
        config = {"quantize_config": None}
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        # BaseQuantizeWrapper may be constructed with any QuantizeConfig and the
        # wrapper itself cannot know all the possible config classes.
        # The deserialization code should ensure the QuantizeConfig is in keras
        # serialization scope.
        quantize_config = deserialize_keras_object(
            config.pop("quantize_config"), module_objects=globals(), custom_objects=None
        )

        layer = tf.keras.layers.deserialize(config.pop("layer"))

        return cls(layer=layer, quantize_config=quantize_config, **config)

    @property
    def trainable(self):
        return self.layer.trainable

    @trainable.setter
    def trainable(self, value):
        self.layer.trainable = value

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights + self._trainable_weights

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights + self._non_trainable_weights

    @property
    def updates(self):
        return self.layer.updates + self._updates

    @property
    def losses(self):
        return self.layer.losses + self._losses
