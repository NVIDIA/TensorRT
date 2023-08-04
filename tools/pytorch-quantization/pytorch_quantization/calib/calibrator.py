#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Abstract base class for calibrators"""


class _Calibrator():
    """Abstract base class of calibrators
    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        axis:
    """
    def __init__(self, num_bits, axis, unsigned):
        self._num_bits = num_bits
        self._axis = axis
        self._unsigned = unsigned

    def collect(self, x):
        """Abstract method: collect tensor statistics used to compute amax

        Args:
            x: A tensor
        """
        raise NotImplementedError

    def reset(self):
        """Abstract method: reset calibrator to initial state"""
        raise NotImplementedError

    def compute_amax(self, *args, **kwargs):
        """Abstract method: compute the amax from the collected data

        Returns:
            amax: a tensor
        """
        raise NotImplementedError

    def __repr__(self):
        s = "num_bits={_num_bits}"
        s += " axis={_axis}"
        s += " unsigned={_unsigned}"
        return s.format(**self.__dict__)
