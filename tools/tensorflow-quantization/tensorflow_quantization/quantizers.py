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

"""
This module is borrowed from TFMOT repository and updated.
It implements QDQ insertion based on "Last Value Quantization".
"""

import tensorflow as tf


def LastValueQuantize(
    inputs,
    min_var,
    max_var,
    per_channel=False,
    channel_axis=-1,
    name_prefix="LastValueQuant",
    is_training=True,
    num_bits=8,
    narrow_range=False,
    symmetric=False,
):
    """Adds a layer that collects quantization ranges as last input ranges.
    LastValueQuantize creates variables called 'min' and 'max', representing the
    interval used for quantization and clamping.
    Args:
      inputs: a tensor containing values to be quantized.
      per_channel: (Optional) a boolean specifying whether to use different
        quantization ranges per output channel.
      init_min: a float scalar, the initial value for variable min.
      init_max: a float scalar, the initial value for variable max.
      name_prefix: name_prefix for created nodes.
      is_training: Whether the op is applied to a training or eval graph.
      num_bits: Number of bits to use for quantization, must be between 2 and 8.
      narrow_range: Whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
    Returns:
      a tensor containing quantized values.
    """
    with tf.name_scope(name_prefix):
        input_shape = inputs.get_shape()
        input_dim = len(input_shape)
        if channel_axis == -1:
            channel_axis += input_dim

        if not is_training:
            return _QuantizeAndDequantize(
                inputs,
                min_var,
                max_var,
                per_channel=per_channel,
                channel_axis=channel_axis,
                num_bits=num_bits,
                narrow_range=narrow_range,
            )

        if per_channel:
            if input_dim == 2:
                reduce_dims = [0]
            elif input_dim == 4:
                reduce_dims = [i for i in range(input_dim) if i != channel_axis]

        if per_channel:
            if input_dim >= 2:
                batch_min = tf.math.reduce_min(
                    inputs, axis=reduce_dims, name="BatchMin"
                )
            else:
                batch_min = inputs
        else:
            batch_min = tf.math.reduce_min(inputs, name="BatchMin")

        if per_channel:
            if input_dim >= 2:
                batch_max = tf.math.reduce_max(
                    inputs, axis=reduce_dims, name="BatchMax"
                )
            else:
                batch_max = inputs
        else:
            batch_max = tf.math.reduce_max(inputs, name="BatchMax")

        if symmetric:
            if narrow_range:
                min_max_ratio = -1
            else:
                # In two's complement notation, the negative range is slightly larger
                # than the positive range.
                min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)

            # TFLite requires that 0.0 if always in the [min; max] range. Because
            # batch_min <= batch_max, it follows that range_min <= 0 <= range_max.
            range_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
            range_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)
        else:
            # TFLite requires that 0.0 if always in the [min; max] range.
            range_min = tf.math.minimum(batch_min, 0.0)
            range_max = tf.math.maximum(batch_max, 0.0)

        assign_min = min_var.assign(range_min, name="AssignMinLast")
        assign_max = max_var.assign(range_max, name="AssignMaxLast")

        return _QuantizeAndDequantize(
            inputs,
            assign_min,
            assign_max,
            per_channel=per_channel,
            channel_axis=channel_axis,
            num_bits=num_bits,
            narrow_range=narrow_range,
        )


def _QuantizeAndDequantize(
    inputs, min_var, max_var, per_channel, channel_axis, num_bits, narrow_range
):
    """Adds a fake quantization operation.
    Depending on value of per_channel, this operation may do global quantization
    or per channel quantization.  min_var and max_var should have corresponding
    shapes: [1] when per_channel == False and [d] when per_channel == True.
    Args:
      inputs: a tensor containing values to be quantized.
      min_var: a variable containing quantization range lower end(s).
      max_var: a variable containing quantization range upper end(s).
      per_channel: a boolean specifying whether to use per-channel quantization.
      num_bits: Number of bits to use for quantization, must be between 2 and 8.
      narrow_range: Whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    Returns:
      a tensor containing quantized values.
    """

    if per_channel:

        return tf.quantization.quantize_and_dequantize_v2(
            inputs,
            min_var,
            max_var,
            num_bits=num_bits,
            narrow_range=narrow_range,
            axis=channel_axis,
            range_given=True,
        )
    else:
        assert min_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
        assert max_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison

        return tf.quantization.quantize_and_dequantize_v2(
            inputs,
            min_var,
            max_var,
            num_bits=num_bits,
            narrow_range=narrow_range,
            range_given=True,
        )
