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


"""Function to get absolute maximum of a tensor
Follow numpy fashion, which is more generic as pytorch's
"""

import torch

def reduce_amax(input, axis=None, keepdims=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computeation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True
        granularity: DEPRECTED. specifies if the statistic has to be calculated at tensor or channel granularity

    Returns:
        The reduced tensor.

    Raises:
        ValueError: Any axis which doesn't make sense or is not supported
        ValueError: If unknown granularity is passed in.
    """
    with torch.no_grad():
        output = input.abs()
        if axis is None:
            output = torch.max(output)
        else:
            if isinstance(axis, int):
                output, _ = torch.max(output, dim=axis, keepdim=keepdims)
            else:
                if isinstance(axis, tuple) and len(axis) > input.dim():
                    raise ValueError("Cannot reduce more axes than tensor's dim.")
                for i in axis:
                    output, _ = torch.max(output, dim=i, keepdim=True)
                if not keepdims or output.numel() == 1:
                    output.squeeze_()
        return output
