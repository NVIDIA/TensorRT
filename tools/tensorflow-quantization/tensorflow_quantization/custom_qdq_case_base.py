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


from abc import ABC


class CustomQDQInsertionCase(ABC):
    """
    This class helps user to programatically decide toolkit behavior to quantize specific layers.
    Based on the output of this class 'case' function, toolkit deviates from its standard behavior.
    """

    def info(self) -> str:
        return ""

    def case(
        self, keras_model: "tf.keras.Model", qspec: "QuantizationSpec"
    ) -> "QuantizationSpec":
        """
        This function is called internally by the framework.
        Given keras model is passed as an argument and object of QuantizationSpec class
        is expcted in return.
        Returned QuantzaionSpec class object should contain information about the layers that needs
        to be treated specially/differently from default framework behavior.

        Args:
            keras_model (tf.keras.Model): Keras functional or sequentail model
            qspec (QuantizationSpec): User passed QuantizationSpec object. It is important to note that
            new special qdq might or might not use quantizations specs user has provided.
        Returns:
            A new QuantizationSpec object.
        """
        raise NotImplementedError("case method must be overridden by user")
